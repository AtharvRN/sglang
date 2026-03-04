import logging
import math
import time
from copy import deepcopy
from typing import Optional, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    can_dflash_use_fused_qkv_proj,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import get_bool_env_var, is_cuda

logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self._warned_sampling_fallback = False
        self._logged_first_verify = False
        self._report_timing = get_bool_env_var("SGLANG_DFLASH_REPORT_TIMING")
        self._report_cycle_trace = bool(server_args.speculative_dflash_cycle_trace)
        self._last_draft_time_s = 0.0
        self._warned_mixed_runtime_block_size = False
        self._warned_invalid_runtime_block_size = False
        self._logged_runtime_block_size_override = False

        # Draft runner (separate KV cache + attention backend).
        # Share req_to_token_pool + token_to_kv_pool_allocator with the target worker (EAGLE3-style),
        # while keeping a separate draft KV cache pool (the draft model has different KV values).
        shared_req_to_token_pool, shared_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"
        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' yet; "
                "falling back to 'flashinfer'."
            )
            draft_backend = "flashinfer"
        elif draft_backend not in ("flashinfer", "fa3"):
            logger.warning(
                "DFLASH draft worker only supports attention_backend in {'flashinfer', 'fa3'} for now, "
                "but got %r. Falling back to 'flashinfer'.",
                draft_backend,
            )
            draft_backend = "flashinfer"

        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=shared_token_to_kv_pool_allocator,
        )
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        self._adaptive_block_size_enabled = bool(
            server_args.speculative_dflash_adaptive_block_size
        )
        self._adaptive_algo = str(
            server_args.speculative_dflash_adaptive_algo
        ).lower()
        self._adaptive_rho = float(server_args.speculative_dflash_adaptive_rho)
        self._adaptive_delta = float(server_args.speculative_dflash_adaptive_delta)
        self._adaptive_reward_mode = str(
            server_args.speculative_dflash_adaptive_reward_mode
        ).lower()
        self._adaptive_ucb_c = float(server_args.speculative_dflash_adaptive_ucb_c)
        self._adaptive_ucb_delta = float(
            server_args.speculative_dflash_adaptive_ucb_delta
        )
        self._adaptive_k_min = (
            int(server_args.speculative_dflash_adaptive_k_min)
            if server_args.speculative_dflash_adaptive_k_min is not None
            else 1
        )
        self._adaptive_k_max = (
            int(server_args.speculative_dflash_adaptive_k_max)
            if server_args.speculative_dflash_adaptive_k_max is not None
            else int(self.block_size)
        )
        self._adaptive_k_start = (
            int(server_args.speculative_dflash_adaptive_k_start)
            if server_args.speculative_dflash_adaptive_k_start is not None
            else int(self.block_size)
        )
        self._adaptive_low_accept_threshold = float(
            server_args.speculative_dflash_adaptive_low_accept_threshold
        )
        self._adaptive_low_accept_streak = int(
            server_args.speculative_dflash_adaptive_low_accept_streak
        )
        self._adaptive_high_accept_threshold = float(
            server_args.speculative_dflash_adaptive_high_accept_threshold
        )
        self._adaptive_high_accept_streak = int(
            server_args.speculative_dflash_adaptive_high_accept_streak
        )
        self._adaptive_cooldown_cycles = int(
            server_args.speculative_dflash_adaptive_cooldown_cycles
        )
        self._adaptive_block_buckets: list[int] = []
        if self._adaptive_block_size_enabled:
            self._adaptive_block_buckets = self._build_adaptive_block_buckets()
        self._last_runtime_block_size = int(self.block_size)

        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
            )
            if self._adaptive_block_size_enabled:
                logger.info(
                    "DFLASH adaptive block size enabled. "
                    "algo=%s reward_mode=%s k_min=%d k_max=%d k_start=%d "
                    "rho=%.3f delta=%.3f ucb_c=%.3f ucb_delta=%.4f "
                    "low_accept_threshold=%.3f low_accept_streak=%d "
                    "high_accept_threshold=%.3f high_accept_streak=%d cooldown_cycles=%d",
                    self._adaptive_algo,
                    self._adaptive_reward_mode,
                    self._adaptive_k_min,
                    self._adaptive_k_max,
                    self._adaptive_k_start,
                    self._adaptive_rho,
                    self._adaptive_delta,
                    self._adaptive_ucb_c,
                    self._adaptive_ucb_delta,
                    self._adaptive_low_accept_threshold,
                    self._adaptive_low_accept_streak,
                    self._adaptive_high_accept_threshold,
                    self._adaptive_high_accept_streak,
                    self._adaptive_cooldown_cycles,
                )
                if self._adaptive_block_buckets:
                    logger.info(
                        "DFLASH adaptive block-size buckets enabled: %s",
                        self._adaptive_block_buckets,
                    )
            if self._report_cycle_trace:
                logger.info("DFLASH per-cycle trace enabled.")
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0

        self._use_fused_kv_materialize = is_cuda()
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

    def _build_adaptive_block_buckets(self) -> list[int]:
        upper = int(min(int(self._adaptive_k_max), int(self.block_size)))
        lower = int(max(1, int(self._adaptive_k_min)))
        if upper < lower:
            upper = lower
        configured_buckets = getattr(
            self.server_args, "speculative_dflash_adaptive_block_buckets", None
        )
        if configured_buckets:
            buckets = sorted(
                {
                    int(v)
                    for v in configured_buckets
                    if int(v) >= lower and int(v) <= upper
                }
            )
            if len(buckets) > 0:
                return buckets

        # Fallback to contiguous runtime block sizes when buckets are not configured.
        buckets = list(range(lower, upper + 1))
        if len(buckets) == 0:
            buckets = [int(min(max(int(self._adaptive_k_start), 1), int(self.block_size)))]
        return buckets

    def _snap_to_adaptive_bucket(self, value: int, mode: str = "nearest") -> int:
        if not self._adaptive_block_buckets:
            return int(value)
        val = int(value)
        buckets = self._adaptive_block_buckets
        if mode == "floor":
            for bucket in reversed(buckets):
                if bucket <= val:
                    return int(bucket)
            return int(buckets[0])
        if mode == "ceil":
            for bucket in buckets:
                if bucket >= val:
                    return int(bucket)
            return int(buckets[-1])
        # nearest (tie -> smaller bucket)
        return int(min(buckets, key=lambda b: (abs(int(b) - val), int(b))))

    def _clamp_runtime_block_size(self, value: int) -> int:
        return int(min(max(1, int(value)), int(self.block_size)))

    def _adaptive_arms(self) -> list[int]:
        if self._adaptive_block_buckets:
            return [int(v) for v in self._adaptive_block_buckets]
        lower = int(max(1, int(self._adaptive_k_min)))
        upper = int(min(int(self._adaptive_k_max), int(self.block_size)))
        if upper < lower:
            upper = lower
        return list(range(lower, upper + 1))

    def _ensure_req_ucb_state(self, req) -> tuple[dict[int, int], dict[int, float]]:
        arms = self._adaptive_arms()
        counts = getattr(req, "dflash_adaptive_ucb_counts", None)
        reward_sums = getattr(req, "dflash_adaptive_ucb_reward_sums", None)
        if not isinstance(counts, dict):
            counts = {}
        if not isinstance(reward_sums, dict):
            reward_sums = {}

        # Keep state aligned with current arm space.
        normalized_counts: dict[int, int] = {}
        normalized_sums: dict[int, float] = {}
        for arm in arms:
            arm_i = int(arm)
            normalized_counts[arm_i] = int(counts.get(arm_i, 0) or 0)
            normalized_sums[arm_i] = float(reward_sums.get(arm_i, 0.0) or 0.0)

        req.dflash_adaptive_ucb_counts = normalized_counts
        req.dflash_adaptive_ucb_reward_sums = normalized_sums
        if getattr(req, "dflash_adaptive_ucb_rounds", None) is None:
            req.dflash_adaptive_ucb_rounds = 0
        if getattr(req, "dflash_adaptive_ucb_reward_norm_max", None) is None:
            req.dflash_adaptive_ucb_reward_norm_max = 1.0
        return normalized_counts, normalized_sums

    def _compute_ucb_reward(
        self,
        *,
        req,
        accepted_draft_tokens: int,
        draft_time_s: float,
        verify_time_s: float,
        num_active_reqs: int,
    ) -> tuple[float, str, float]:
        # Include the target bonus token to match cycle-level accept length.
        accept_length = float(max(0, int(accepted_draft_tokens)) + 1)

        if self._adaptive_reward_mode != "throughput":
            return accept_length, "accept_length", accept_length

        if (
            not self._report_timing
            or draft_time_s <= 0.0
            or verify_time_s <= 0.0
            or num_active_reqs <= 0
        ):
            # Timing is not always enabled in deployment; fall back to token-based reward.
            return accept_length, "accept_length_fallback_no_timing", accept_length

        per_req_cycle_time_s = (float(draft_time_s) + float(verify_time_s)) / float(
            max(num_active_reqs, 1)
        )
        raw = accept_length / max(per_req_cycle_time_s, 1e-6)

        # Normalize to [0, 1] using running max to keep UCB numerically stable.
        reward_norm_max = float(
            max(getattr(req, "dflash_adaptive_ucb_reward_norm_max", 1.0), raw, 1e-6)
        )
        req.dflash_adaptive_ucb_reward_norm_max = reward_norm_max
        reward = raw / reward_norm_max
        return reward, "throughput", raw

    def _init_req_adaptive_state(self, req) -> int:
        if (
            getattr(req, "dflash_adaptive_current_bs", None) is not None
            and int(req.dflash_adaptive_current_bs) >= 1
        ):
            resolved = self._clamp_runtime_block_size(req.dflash_adaptive_current_bs)
            if self._adaptive_block_buckets:
                resolved = self._snap_to_adaptive_bucket(resolved, mode="nearest")
            if self._adaptive_algo == "ucb":
                self._ensure_req_ucb_state(req)
            return resolved

        init_bs = int(self._adaptive_k_start if self._adaptive_block_size_enabled else self.block_size)
        sampling_params = getattr(req, "sampling_params", None)
        custom_params = (
            getattr(sampling_params, "custom_params", None)
            if sampling_params is not None
            else None
        )
        if isinstance(custom_params, dict) and "dflash_block_size" in custom_params:
            raw_val = custom_params["dflash_block_size"]
            try:
                init_bs = int(raw_val)
            except Exception:
                if not self._warned_invalid_runtime_block_size and self.tp_rank == 0:
                    logger.warning(
                        "Ignoring invalid runtime dflash_block_size=%r. "
                        "Expected an integer in [1, %d].",
                        raw_val,
                        int(self.block_size),
                    )
                    self._warned_invalid_runtime_block_size = True

        init_bs = self._clamp_runtime_block_size(init_bs)
        init_bs = int(min(max(init_bs, int(self._adaptive_k_min)), int(self._adaptive_k_max)))
        if self._adaptive_block_buckets:
            init_bs = self._snap_to_adaptive_bucket(init_bs, mode="nearest")

        req.dflash_adaptive_current_bs = init_bs
        req.dflash_adaptive_lgen_hat = None
        req.dflash_adaptive_lacc_hat = None
        req.dflash_adaptive_accept_ratio_ewma = None
        req.dflash_adaptive_low_accept_count = 0
        req.dflash_adaptive_high_accept_count = 0
        req.dflash_adaptive_cooldown_remaining = 0
        req.dflash_adaptive_last_decision = None
        if self._adaptive_algo == "ucb":
            self._ensure_req_ucb_state(req)
            req.dflash_adaptive_ucb_rounds = 0
            req.dflash_adaptive_ucb_reward_norm_max = 1.0
            req.dflash_adaptive_ucb_last_scores = None
        return init_bs

    def _resolve_runtime_block_size(self, batch: ScheduleBatch) -> int:
        """Resolve runtime DFLASH block size for the current decode step.

        Modes:
        - Adaptive mode ON: use per-request adaptive state and choose an effective
          batch block size as the minimum desired block size among active requests.
        - Adaptive mode OFF: fallback to optional request custom param
          (`sampling_params.custom_params['dflash_block_size']`) with min-on-mixed policy.
        """
        max_block_size = int(self.block_size)

        if self._adaptive_block_size_enabled:
            desired: list[int] = []
            for req in batch.reqs:
                desired.append(int(self._init_req_adaptive_state(req)))

            if len(desired) == 0:
                return max_block_size

            effective_block_size = int(min(desired))

            if (
                len(set(desired)) > 1
                and not self._warned_mixed_runtime_block_size
                and self.tp_rank == 0
            ):
                logger.info(
                    "DFLASH adaptive per-request desired block sizes are mixed (%s); "
                    "using min=%d for this step.",
                    sorted(set(desired)),
                    effective_block_size,
                )
                self._warned_mixed_runtime_block_size = True

            return effective_block_size

        runtime_block_sizes: list[int] = []
        for req in batch.reqs:
            sampling_params = getattr(req, "sampling_params", None)
            custom_params = (
                getattr(sampling_params, "custom_params", None)
                if sampling_params is not None
                else None
            )
            if not isinstance(custom_params, dict):
                continue
            if "dflash_block_size" not in custom_params:
                continue
            raw_val = custom_params["dflash_block_size"]
            try:
                runtime_block_sizes.append(int(raw_val))
            except Exception:
                if not self._warned_invalid_runtime_block_size and self.tp_rank == 0:
                    logger.warning(
                        "Ignoring invalid runtime dflash_block_size=%r. "
                        "Expected an integer in [1, %d].",
                        raw_val,
                        max_block_size,
                    )
                    self._warned_invalid_runtime_block_size = True

        if len(runtime_block_sizes) == 0:
            return max_block_size

        clamped = [self._clamp_runtime_block_size(v) for v in runtime_block_sizes]
        effective_block_size = int(min(clamped))
        if (
            len(set(clamped)) > 1
            and not self._warned_mixed_runtime_block_size
            and self.tp_rank == 0
        ):
            logger.info(
                "DFLASH mixed runtime block sizes in one batch (%s); using min=%d for this step.",
                sorted(set(clamped)),
                effective_block_size,
            )
            self._warned_mixed_runtime_block_size = True
        if (
            effective_block_size != max_block_size
            and not self._logged_runtime_block_size_override
            and self.tp_rank == 0
        ):
            logger.info(
                "DFLASH runtime block-size override active: max=%d, effective=%d.",
                max_block_size,
                effective_block_size,
            )
            self._logged_runtime_block_size_override = True
        return effective_block_size

    def _update_req_adaptive_state(
        self,
        req,
        *,
        accepted_draft_tokens: int,
        runtime_block_size: int,
        draft_time_s: float = 0.0,
        verify_time_s: float = 0.0,
        num_active_reqs: int = 1,
    ) -> None:
        if not self._adaptive_block_size_enabled:
            return

        current_bs = self._init_req_adaptive_state(req)
        proposed = max(0, int(runtime_block_size) - 1)
        accepted = max(0, int(accepted_draft_tokens))
        accept_ratio = (
            max(0.0, min(1.0, float(accepted) / float(proposed)))
            if proposed > 0
            else 1.0
        )

        if self._adaptive_algo == "ucb":
            counts, reward_sums = self._ensure_req_ucb_state(req)
            arms = sorted(int(a) for a in counts.keys())

            pulled_arm = int(runtime_block_size)
            pulled_arm = int(
                min(max(pulled_arm, int(self._adaptive_k_min)), int(self._adaptive_k_max))
            )
            pulled_arm = self._clamp_runtime_block_size(pulled_arm)
            if self._adaptive_block_buckets:
                pulled_arm = self._snap_to_adaptive_bucket(pulled_arm, mode="nearest")
            if pulled_arm not in counts:
                counts[pulled_arm] = 0
                reward_sums[pulled_arm] = 0.0
                arms = sorted(int(a) for a in counts.keys())

            reward, reward_source, reward_raw = self._compute_ucb_reward(
                req=req,
                accepted_draft_tokens=accepted,
                draft_time_s=float(draft_time_s),
                verify_time_s=float(verify_time_s),
                num_active_reqs=int(num_active_reqs),
            )

            counts[pulled_arm] = int(counts[pulled_arm]) + 1
            reward_sums[pulled_arm] = float(reward_sums[pulled_arm]) + float(reward)
            rounds = int(getattr(req, "dflash_adaptive_ucb_rounds", 0) or 0) + 1
            req.dflash_adaptive_ucb_rounds = int(rounds)

            unseen_arms = [int(a) for a in arms if int(counts.get(int(a), 0)) == 0]
            means: dict[int, float] = {}
            bonuses: dict[int, float] = {}
            scores: dict[int, float] = {}
            if unseen_arms:
                next_bs = int(unseen_arms[0])
                reason = "ucb_warmup"
            else:
                num_arms = max(len(arms), 1)
                if self._adaptive_reward_mode == "accept_length":
                    l_bound = float(max(int(self.block_size), 1))
                    delta = max(float(self._adaptive_ucb_delta), 1e-12)
                    for arm in arms:
                        n = float(max(int(counts.get(arm, 0)), 1))
                        mean = float(reward_sums.get(arm, 0.0)) / n
                        log_arg = (
                            float(num_arms)
                            * float(max(rounds, 1) ** 2)
                            * math.sqrt(1.0 + n)
                            / delta
                        )
                        inner = 1.0 + 2.0 * math.log(max(log_arg, 1.0000001))
                        bonus = (l_bound / 2.0) * math.sqrt(
                            ((1.0 + n) / (n * n)) * max(inner, 0.0)
                        )
                        means[arm] = float(mean)
                        bonuses[arm] = float(bonus)
                        scores[arm] = float(mean + bonus)
                else:
                    c = max(float(self._adaptive_ucb_c), 0.0)
                    log_t = math.log(max(float(rounds), 2.0))
                    for arm in arms:
                        n = float(max(int(counts.get(arm, 0)), 1))
                        mean = float(reward_sums.get(arm, 0.0)) / n
                        bonus = c * math.sqrt((2.0 * log_t) / n)
                        means[arm] = float(mean)
                        bonuses[arm] = float(bonus)
                        scores[arm] = float(mean + bonus)
                next_bs = int(max(arms, key=lambda a: (scores[a], -int(a))))
                reason = "ucb_score"

            action = "hold"
            if next_bs > int(current_bs):
                action = "up"
            elif next_bs < int(current_bs):
                action = "down"

            unclamped_next_bs = int(next_bs)
            next_bs = int(
                min(max(int(next_bs), int(self._adaptive_k_min)), int(self._adaptive_k_max))
            )
            next_bs = self._clamp_runtime_block_size(next_bs)
            if self._adaptive_block_buckets:
                next_bs = self._snap_to_adaptive_bucket(next_bs, mode="nearest")
            if next_bs != unclamped_next_bs:
                reason = "clamped_or_bucketed"
                if next_bs > int(current_bs):
                    action = "up"
                elif next_bs < int(current_bs):
                    action = "down"
                else:
                    action = "hold"

            old_ratio_ewma = getattr(req, "dflash_adaptive_accept_ratio_ewma", None)
            if old_ratio_ewma is None:
                accept_ratio_ewma = float(accept_ratio)
            else:
                accept_ratio_ewma = float(
                    (1.0 - self._adaptive_rho) * float(old_ratio_ewma)
                    + self._adaptive_rho * float(accept_ratio)
                )

            req.dflash_adaptive_current_bs = int(next_bs)
            req.dflash_adaptive_lgen_hat = None
            req.dflash_adaptive_lacc_hat = None
            req.dflash_adaptive_accept_ratio_ewma = float(accept_ratio_ewma)
            req.dflash_adaptive_low_accept_count = 0
            req.dflash_adaptive_high_accept_count = 0
            req.dflash_adaptive_cooldown_remaining = 0
            req.dflash_adaptive_ucb_last_scores = (
                {int(a): float(scores[a]) for a in scores} if scores else None
            )
            req.dflash_adaptive_last_decision = {
                "algo": "ucb",
                "reward_mode": self._adaptive_reward_mode,
                "reward_source": reward_source,
                "reward": float(reward),
                "reward_raw": float(reward_raw),
                "ucb_round": int(rounds),
                "ucb_pulled_arm": int(pulled_arm),
                "ucb_pulled_arm_count": int(counts.get(pulled_arm, 0)),
                "ucb_pulled_arm_mean_reward": float(
                    float(reward_sums.get(pulled_arm, 0.0))
                    / float(max(int(counts.get(pulled_arm, 0)), 1))
                ),
                "ucb_selected_mean_reward": (
                    float(means[next_bs]) if next_bs in means else None
                ),
                "ucb_selected_bonus": (
                    float(bonuses[next_bs]) if next_bs in bonuses else None
                ),
                "ucb_selected_score": (
                    float(scores[next_bs]) if next_bs in scores else None
                ),
                "prev_bs": int(current_bs),
                "next_bs": int(next_bs),
                "action": action,
                "reason": reason,
                "accept_ratio": float(accept_ratio),
                "accept_ratio_ewma": float(accept_ratio_ewma),
                "accepted_draft_tokens": int(accepted),
                "proposed_draft_tokens": int(proposed),
                "low_accept_count": 0,
                "high_accept_count": 0,
                "cooldown_remaining": 0,
                "lgen_hat": None,
                "lacc_hat": None,
            }
            return

        action = "hold"
        reason = "ewma_hold"

        # Original EWMA proposal controller:
        # - lgen_hat tracks proposed draft length (k-1)
        # - lacc_hat tracks accepted draft length
        # - if acceptance keeps up (lacc_hat >= lgen_hat), allow +delta growth
        old_lgen = getattr(req, "dflash_adaptive_lgen_hat", None)
        old_lacc = getattr(req, "dflash_adaptive_lacc_hat", None)
        if old_lgen is None:
            lgen_hat = float(proposed)
        else:
            lgen_hat = float(
                (1.0 - self._adaptive_rho) * float(old_lgen)
                + self._adaptive_rho * float(proposed)
            )
        if old_lacc is None:
            lacc_hat = float(accepted)
        else:
            lacc_hat = float(
                (1.0 - self._adaptive_rho) * float(old_lacc)
                + self._adaptive_rho * float(accepted)
            )
        req.dflash_adaptive_lgen_hat = float(lgen_hat)
        req.dflash_adaptive_lacc_hat = float(lacc_hat)

        growth = float(self._adaptive_delta) if lacc_hat >= lgen_hat else 0.0
        next_proposed = int(math.ceil(lgen_hat + growth))
        next_bs = int(next_proposed + 1)
        if next_bs > current_bs:
            action = "up"
            reason = "ewma_growth"
        elif next_bs < current_bs:
            action = "down"
            reason = "ewma_shrink"

        # Conservative fallback on persistent low acceptance.
        low_count = int(getattr(req, "dflash_adaptive_low_accept_count", 0) or 0)
        if proposed == 0:
            # k=1 has no drafted tokens; probe upward to avoid k=1 deadlock.
            accept_ratio = 1.0
            next_bs = max(int(next_bs), int(current_bs) + 1)
            low_count = 0
            action = "up"
            reason = "k1_probe_recover"
        else:
            if accept_ratio < float(self._adaptive_low_accept_threshold):
                low_count += 1
            else:
                low_count = 0

            if low_count >= int(self._adaptive_low_accept_streak):
                fallback_bs = max(int(self._adaptive_k_min), int(current_bs) - 1)
                next_bs = min(int(next_bs), int(fallback_bs))
                low_count = 0
                if next_bs < current_bs:
                    action = "down"
                    reason = "low_accept_streak"

        req.dflash_adaptive_low_accept_count = int(low_count)
        req.dflash_adaptive_high_accept_count = 0
        req.dflash_adaptive_cooldown_remaining = 0

        unclamped_next_bs = int(next_bs)
        next_bs = int(
            min(max(int(next_bs), int(self._adaptive_k_min)), int(self._adaptive_k_max))
        )
        next_bs = self._clamp_runtime_block_size(next_bs)
        if self._adaptive_block_buckets:
            if next_bs > current_bs:
                next_bs = self._snap_to_adaptive_bucket(next_bs, mode="ceil")
            elif next_bs < current_bs:
                next_bs = self._snap_to_adaptive_bucket(next_bs, mode="floor")
            else:
                next_bs = self._snap_to_adaptive_bucket(next_bs, mode="nearest")

        if next_bs != unclamped_next_bs:
            reason = "clamped_or_bucketed"

        req.dflash_adaptive_current_bs = next_bs
        accept_ratio_ewma = (
            float(lacc_hat) / float(max(lgen_hat, 1e-6)) if lgen_hat > 0.0 else 1.0
        )
        req.dflash_adaptive_accept_ratio_ewma = float(accept_ratio_ewma)
        req.dflash_adaptive_last_decision = {
            "algo": "ewma",
            "reward_mode": "accept_length",
            "prev_bs": int(current_bs),
            "next_bs": int(next_bs),
            "action": action,
            "reason": reason,
            "accept_ratio": float(accept_ratio),
            "accept_ratio_ewma": float(accept_ratio_ewma),
            "accepted_draft_tokens": int(accepted),
            "proposed_draft_tokens": int(proposed),
            "low_accept_count": int(low_count),
            "high_accept_count": 0,
            "cooldown_remaining": 0,
            "lgen_hat": float(lgen_hat),
            "lacc_hat": float(lacc_hat),
        }

    def _record_runtime_block_size_usage(
        self, batch: ScheduleBatch, runtime_block_size: int
    ) -> None:
        """Accumulate per-request runtime block-size usage (one count per verify cycle)."""
        bs = int(runtime_block_size)
        for req in batch.reqs:
            hist = getattr(req, "dflash_runtime_bs_hist", None)
            if not isinstance(hist, dict):
                hist = {}
                req.dflash_runtime_bs_hist = hist
            hist[bs] = int(hist.get(bs, 0)) + 1

    def _measure_forward_s(self, fn) -> tuple[object, float]:
        if not self._report_timing:
            return fn(), 0.0

        if is_cuda():
            with torch.cuda.device(self.model_runner.device):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                out = fn()
                end_event.record()
                end_event.synchronize()
                return out, float(start_event.elapsed_time(end_event)) / 1000.0

        start_t = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - start_t

    @staticmethod
    def _accumulate_req_shared_time(
        reqs: list, attr_name: str, total_time_s: float
    ) -> None:
        if total_time_s <= 0.0 or len(reqs) == 0:
            return

        # One verify/draft forward pass is shared by all active requests.
        per_req_time_s = total_time_s / float(len(reqs))
        for req in reqs:
            setattr(req, attr_name, float(getattr(req, attr_name, 0.0)) + per_req_time_s)

    def _iter_req_value_pairs(self, reqs: list, values, *, tag: str):
        n_req = len(reqs)
        n_val = len(values)
        if n_req != n_val and self.tp_rank == 0:
            logger.warning(
                "DFLASH %s length mismatch (reqs=%d, values=%d); truncating to min length.",
                tag,
                n_req,
                n_val,
            )
        n = min(n_req, n_val)
        for i in range(n):
            yield reqs[i], values[i]

    def _record_cycle_trace(
        self,
        *,
        batch: ScheduleBatch,
        accept_length_per_req_cpu: list[int],
        runtime_block_size: int,
        verify_time_s: float,
    ) -> None:
        if not self._report_cycle_trace:
            return
        if len(batch.reqs) == 0:
            return

        proposed_draft_tokens = max(0, int(runtime_block_size) - 1)
        per_req_draft_time_s = (
            float(self._last_draft_time_s) / float(len(batch.reqs))
            if self._report_timing and self._last_draft_time_s > 0.0
            else None
        )
        per_req_verify_time_s = (
            float(verify_time_s) / float(len(batch.reqs))
            if self._report_timing and verify_time_s > 0.0
            else None
        )

        for req, accepted_draft_tokens in self._iter_req_value_pairs(
            batch.reqs, accept_length_per_req_cpu, tag="cycle_trace"
        ):
            trace = getattr(req, "spec_cycle_trace", None)
            if not isinstance(trace, list):
                trace = []
                req.spec_cycle_trace = trace

            accepted_draft_tokens = int(max(0, accepted_draft_tokens))
            accept_rate = (
                float(accepted_draft_tokens) / float(proposed_draft_tokens)
                if proposed_draft_tokens > 0
                else None
            )
            adaptive_decision = getattr(req, "dflash_adaptive_last_decision", None)
            trace.append(
                {
                    "cycle_idx": int(getattr(req, "spec_verify_ct", 0)),
                    "runtime_block_size": int(runtime_block_size),
                    "accepted_draft_tokens": accepted_draft_tokens,
                    # Include the target bonus token for cycle-level tau.
                    "accept_length": int(accepted_draft_tokens + 1),
                    "accept_rate": accept_rate,
                    "draft_time_s": per_req_draft_time_s,
                    "verify_time_s": per_req_verify_time_s,
                    "adaptive_decision": adaptive_decision,
                }
            )

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        try:
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
            )
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # allocator and req_to_token_pool are shared with target worker
        pass

    def on_req_finished(self, req):
        # allocator and req_to_token_pool are shared with the target worker;
        # there is no separate draft allocation to release here.
        if hasattr(req, "dflash_draft_seq_len"):
            req.dflash_draft_seq_len = 0
        if hasattr(req, "dflash_adaptive_current_bs"):
            req.dflash_adaptive_current_bs = None
            req.dflash_adaptive_lgen_hat = None
            req.dflash_adaptive_lacc_hat = None
            req.dflash_adaptive_accept_ratio_ewma = None
            req.dflash_adaptive_low_accept_count = 0
            req.dflash_adaptive_high_accept_count = 0
            req.dflash_adaptive_cooldown_remaining = 0
            req.dflash_adaptive_last_decision = None
            req.dflash_adaptive_ucb_counts = {}
            req.dflash_adaptive_ucb_reward_sums = {}
            req.dflash_adaptive_ucb_rounds = 0
            req.dflash_adaptive_ucb_reward_norm_max = 1.0
            req.dflash_adaptive_ucb_last_scores = None
            req.dflash_runtime_bs_hist = {}
        if hasattr(req, "spec_cycle_trace"):
            req.spec_cycle_trace = None

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise ValueError(
                "DFLASH does not support grammar-constrained decoding yet."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if (
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                logger.warning(
                    "DFLASH non-greedy verification is unavailable on this build/device; "
                    "falling back to greedy argmax verification."
                )
                self._warned_sampling_fallback = True

        bs = batch.batch_size()
        device = self.model_runner.device
        runtime_block_size = self._resolve_runtime_block_size(batch)
        self._last_runtime_block_size = int(runtime_block_size)
        self._record_runtime_block_size_usage(batch, runtime_block_size)

        # --- 1) Append any newly committed tokens into the draft KV cache.
        self._append_target_hidden_to_draft_kv(batch, draft_input)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFLASH requires the target model to expose a vocab-parallel `lm_head` with `weight` and "
                "`shard_indices` attributes."
            )

        # --- 2) Draft a non-causal block with the draft model.
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs, :runtime_block_size]
        block_ids.fill_(int(self._mask_token_id))
        block_ids[:, 0].copy_(draft_input.verified_id.to(torch.long))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        # For spec-v1, the draft KV cache is always materialized to the current target
        # prefix before drafting the next block.
        prefix_lens = batch.seq_lens  # int32, device

        positions_2d = self._draft_block_positions_buf[:bs, :runtime_block_size]
        torch.add(
            prefix_lens.unsqueeze(1),
            self._block_pos_offsets[:runtime_block_size],
            out=positions_2d,
        )
        # runtime_block_size can be < max block_size, making this view non-contiguous.
        # The fused RoPE kernel requires contiguous position tensors.
        positions = positions_2d.reshape(-1).contiguous()

        block_start = prefix_lens
        block_end = self._draft_block_end_buf[:bs]
        torch.add(block_start, runtime_block_size, out=block_end)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if batch.seq_lens_cpu.dtype == torch.int32:
            seq_lens_cpu.copy_(batch.seq_lens_cpu)
        else:
            seq_lens_cpu.copy_(batch.seq_lens_cpu.to(torch.int32))
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        token_to_kv_pool_state_backup = allocator.backup_state()
        try:
            if self.page_size == 1:
                block_cache_loc = allocator.alloc(bs * runtime_block_size)
            else:
                block_end_cpu = seq_lens_cpu + runtime_block_size
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    block_start,
                    seq_lens_cpu,
                    block_end,
                    block_end_cpu,
                    last_loc,
                    bs * runtime_block_size,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    "DFLASH draft OOM when allocating "
                    f"{bs * runtime_block_size} block tokens."
                )

            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                block_start,
                block_end,
                block_cache_loc,
                bs,
            )

            # Use TARGET_VERIFY mode for draft forwarding. In this mode, `seq_lens`
            # stores prefix lengths; attention backends derive kv_len by adding
            # `draft_token_num` (runtime block size for this step).
            draft_spec_info = self._draft_block_spec_info
            draft_spec_info.draft_token_num = int(runtime_block_size)
            draft_spec_info.num_tokens_per_batch = int(runtime_block_size)
            draft_spec_info.custom_mask = None
            seq_lens = prefix_lens
            seq_lens_sum = int(batch.seq_lens_sum)
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=bs,
                input_ids=block_ids.reshape(-1).contiguous(),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=seq_lens_cpu,
                positions=positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            with torch.inference_mode():
                draft_out, draft_time_s = self._measure_forward_s(
                    lambda: self.draft_model_runner.forward(forward_batch)
                )
                draft_hidden = draft_out.logits_output
                self._last_draft_time_s = float(draft_time_s)
        finally:
            # Drop the speculative block from the shared allocator (EAGLE3-style).
            allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_hidden.view(bs, runtime_block_size, -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, runtime_block_size - 1)
        draft_tokens = self._draft_block_tokens_buf[:bs, :runtime_block_size]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        if runtime_block_size > 1:
            draft_tokens[:, 1:].copy_(draft_next)
        positions = positions_2d.reshape(-1).contiguous()

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1).contiguous(),
            positions=positions,
            draft_token_num=runtime_block_size,
        )
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        verify_input.prepare_for_verify(
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        shard = lm_head.shard_indices
        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    out_token_ids[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start
                    )
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                out_token_ids[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_token_ids[start:end].copy_(selected_ids.view(-1))

        return out_token_ids

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        """

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError(
                "DFLASH draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            return

        req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token

        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens
        draft_seq_lens = draft_input.draft_seq_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if draft_seq_lens.dtype != torch.int32:
            draft_seq_lens = draft_seq_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        if draft_seq_lens.device != device:
            draft_seq_lens = draft_seq_lens.to(device, non_blocking=True)

        if bs == 1:
            # Fast path for single request.
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = draft_seq_lens.to(torch.int64)[:, None] + r[None, :]  # [1, ctx]
            cache2d = req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx]
            ctx_positions = pos2d.reshape(-1)  # [ctx]
        else:
            # In decode mode, ctx_lens <= block_size so we can skip the .item() sync.
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(self.block_size)
            if max_ctx <= 0:
                raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]  # [1, max_ctx]
            pos2d = draft_seq_lens.to(torch.int64)[:, None] + r  # [bs, max_ctx]
            mask = r < ctx_lens[:, None]

            # Batched gather of cache locations and positions.
            cache2d = req_to_token[req_pool_indices[:, None], pos2d]  # [bs, max_ctx]
            ctx_cache_loc = cache2d[mask].to(torch.int64)  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)]

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(
                draft_input.target_hidden
            )  # [sum(ctx), hidden]
            if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                raise RuntimeError(
                    f"DFLASH ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                )

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
                except Exception as e:
                    logger.warning(
                        "DFLASH fused KV append failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )
            else:
                self._append_target_hidden_sequential(
                    ctx_hidden, ctx_positions, ctx_cache_loc
                )

        draft_input.draft_seq_lens = draft_seq_lens + ctx_lens
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        for layer in self.draft_model.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
                attn.k_scale,
                attn.v_scale,
            )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return

        accepted_steps = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=_to_int32_device_tensor(model_worker_batch.extend_seq_lens),
                draft_seq_lens=_to_int32_device_tensor(
                    model_worker_batch.extend_prefix_lens
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input
            for req, draft_len in self._iter_req_value_pairs(
                batch.reqs, batch.seq_lens_cpu, tag="prefill_seq_lens"
            ):
                req.dflash_draft_seq_len = int(draft_len)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)
        if self._report_timing and self.tp_rank == 0:
            self._accumulate_req_shared_time(
                batch.reqs, "spec_draft_time_s", float(self._last_draft_time_s)
            )

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        batch_result, verify_time_s = self._measure_forward_s(
            lambda: self.target_worker.forward_batch_generation(
                model_worker_batch, is_verify=True, **kwargs
            )
        )
        if self._report_timing and self.tp_rank == 0:
            self._accumulate_req_shared_time(
                batch.reqs, "spec_verify_time_s", float(verify_time_s)
            )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        runtime_bs = int(getattr(self, "_last_runtime_block_size", self.block_size))
        if self._adaptive_block_size_enabled:
            for req, accepted_draft_tokens in self._iter_req_value_pairs(
                batch.reqs, accept_length_per_req_cpu, tag="adaptive_update"
            ):
                self._update_req_adaptive_state(
                    req,
                    accepted_draft_tokens=int(accepted_draft_tokens),
                    runtime_block_size=runtime_bs,
                    draft_time_s=float(self._last_draft_time_s),
                    verify_time_s=float(verify_time_s),
                    num_active_reqs=len(batch.reqs),
                )
        self._record_cycle_trace(
            batch=batch,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            runtime_block_size=runtime_bs,
            verify_time_s=float(verify_time_s),
        )

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DFLASH verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
