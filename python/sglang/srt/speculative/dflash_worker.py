import logging
import math
import random
import time
import copy
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
from sglang.srt.speculative.predictor_dataset import PredictorDatasetShardWriter
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
        self._adaptive_proxy_cycle_ms_raw = str(
            getattr(server_args, "speculative_dflash_adaptive_proxy_cycle_ms", "")
            or ""
        ).strip()
        (
            self._adaptive_proxy_cycle_ms_by_bs,
            self._adaptive_proxy_cycle_ms_by_ck,
        ) = self._parse_adaptive_proxy_cycle_ms(self._adaptive_proxy_cycle_ms_raw)
        self._adaptive_proxy_powerlaw_a = float(
            getattr(
                server_args,
                "speculative_dflash_adaptive_proxy_powerlaw_a",
                0.0,
            )
            or 0.0
        )
        self._adaptive_proxy_powerlaw_c_exp = float(
            getattr(
                server_args,
                "speculative_dflash_adaptive_proxy_powerlaw_c_exp",
                0.430,
            )
        )
        self._adaptive_proxy_powerlaw_k_exp = float(
            getattr(
                server_args,
                "speculative_dflash_adaptive_proxy_powerlaw_k_exp",
                0.160,
            )
        )
        self._adaptive_proxy_tau_exp = float(
            getattr(
                server_args,
                "speculative_dflash_adaptive_proxy_tau_exp",
                1.0,
            )
        )
        self._adaptive_proxy_time_exp = float(
            getattr(
                server_args,
                "speculative_dflash_adaptive_proxy_time_exp",
                1.0,
            )
        )
        self._adaptive_ucb_c = float(server_args.speculative_dflash_adaptive_ucb_c)
        self._adaptive_ucb_delta = float(
            server_args.speculative_dflash_adaptive_ucb_delta
        )
        self._adaptive_linucb_alpha = float(
            server_args.speculative_dflash_adaptive_linucb_alpha
        )
        self._adaptive_linucb_lambda = float(
            server_args.speculative_dflash_adaptive_linucb_lambda
        )
        self._adaptive_linucb_dim = 7
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
        self._confidence_gate_enabled = bool(
            getattr(server_args, "speculative_dflash_confidence_gate", False)
        )
        self._confidence_gate_mode = str(
            getattr(server_args, "speculative_dflash_confidence_gate_mode", "threshold")
        ).lower()
        self._confidence_gate_threshold = float(
            getattr(server_args, "speculative_dflash_confidence_threshold", 0.2)
        )
        self._confidence_gate_score_metric = str(
            getattr(
                server_args,
                "speculative_dflash_confidence_gate_score_metric",
                "neg_log_max_prob",
            )
        ).lower()
        self._confidence_gate_score_budget = float(
            getattr(
                server_args,
                "speculative_dflash_confidence_gate_score_budget",
                1.5,
            )
        )
        self._confidence_gate_aggregate = str(
            getattr(server_args, "speculative_dflash_confidence_gate_aggregate", "q10")
        ).lower()
        self._confidence_gate_min_verify_tokens = int(
            getattr(
                server_args,
                "speculative_dflash_confidence_gate_min_verify_tokens",
                1,
            )
        )
        self._confidence_gate_mab_enabled = bool(
            getattr(server_args, "speculative_dflash_confidence_gate_mab", False)
        )
        self._confidence_gate_mab_algo = str(
            getattr(server_args, "speculative_dflash_confidence_gate_mab_algo", "ucb")
        ).lower()
        self._confidence_gate_mab_ucb_c = float(
            getattr(server_args, "speculative_dflash_confidence_gate_mab_ucb_c", 1.0)
        )
        raw_conf_mab_arms = getattr(
            server_args, "speculative_dflash_confidence_gate_mab_arms", None
        )
        if raw_conf_mab_arms:
            self._confidence_gate_mab_arms = sorted(
                {float(v) for v in raw_conf_mab_arms}
            )
        else:
            self._confidence_gate_mab_arms = [float(self._confidence_gate_threshold)]
        self._confidence_gate_mab_counts = {
            float(arm): 0 for arm in self._confidence_gate_mab_arms
        }
        self._confidence_gate_mab_reward_sums = {
            float(arm): 0.0 for arm in self._confidence_gate_mab_arms
        }
        self._confidence_gate_mab_alpha = {
            float(arm): 1.0 for arm in self._confidence_gate_mab_arms
        }
        self._confidence_gate_mab_beta = {
            float(arm): 1.0 for arm in self._confidence_gate_mab_arms
        }
        self._confidence_gate_mab_rounds = 0
        self._confidence_gate_mab_reward_norm_max = 1.0
        self._confidence_gate_grouped_verify_enabled = bool(
            getattr(
                server_args,
                "speculative_dflash_confidence_gate_grouped_verify",
                False,
            )
        )
        raw_group_buckets = getattr(
            server_args,
            "speculative_dflash_confidence_gate_grouped_verify_buckets",
            None,
        )
        if raw_group_buckets:
            self._confidence_gate_grouped_verify_buckets = sorted(
                {
                    int(v)
                    for v in raw_group_buckets
                    if 1 <= int(v) <= int(self.block_size)
                }
            )
        else:
            self._confidence_gate_grouped_verify_buckets = []
        self._last_confidence_gate_decision = None
        self._last_verify_token_num = int(self.block_size)
        self._last_per_req_verify_tokens: Optional[torch.Tensor] = None
        self._last_draft_tokens_2d: Optional[torch.Tensor] = None
        self._last_verify_positions_2d: Optional[torch.Tensor] = None
        self._last_grouped_verify_active = False
        self._last_grouped_verify_plan: list[tuple[int, list[int]]] = []
        self._warned_grouped_verify_mamba_fallback = False
        self._adaptive_block_buckets: list[int] = []
        self._predictor_dataset_writer: Optional[PredictorDatasetShardWriter] = None
        self._last_predictor_draft_hidden_3d: Optional[torch.Tensor] = None
        self._last_predictor_verify_tokens_by_req: Optional[list[int]] = None
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
                    "linucb_alpha=%.3f linucb_lambda=%.3f "
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
                    self._adaptive_linucb_alpha,
                    self._adaptive_linucb_lambda,
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
                if self._adaptive_proxy_cycle_ms_by_bs:
                    logger.info(
                        "DFLASH adaptive throughput-proxy cycle-ms map (bs->ms): %s",
                        self._adaptive_proxy_cycle_ms_by_bs,
                    )
                if self._adaptive_proxy_cycle_ms_by_ck:
                    logger.info(
                        "DFLASH adaptive throughput-proxy cycle-ms map ((c,k)->ms): %s",
                        self._adaptive_proxy_cycle_ms_by_ck,
                    )
                if self._adaptive_proxy_powerlaw_a > 0.0:
                    logger.info(
                        "DFLASH adaptive throughput-proxy power-law enabled: "
                        "a=%.6f c_exp=%.6f k_exp=%.6f",
                        self._adaptive_proxy_powerlaw_a,
                        self._adaptive_proxy_powerlaw_c_exp,
                        self._adaptive_proxy_powerlaw_k_exp,
                    )
                if (
                    self._adaptive_reward_mode == "throughput_proxy"
                    and not self._adaptive_proxy_cycle_ms_by_bs
                    and not self._adaptive_proxy_cycle_ms_by_ck
                    and self._adaptive_proxy_powerlaw_a <= 0.0
                ):
                    logger.warning(
                        "DFLASH throughput_proxy has no cycle-time map and no power-law estimator. "
                        "Falling back to tau/k proxy units (k=runtime block size). "
                        "Provide --speculative-dflash-adaptive-proxy-cycle-ms or enable "
                        "--speculative-dflash-adaptive-proxy-powerlaw-a for cost-aware rewards."
                    )
                if (
                    abs(self._adaptive_proxy_tau_exp - 1.0) > 1e-12
                    or abs(self._adaptive_proxy_time_exp - 1.0) > 1e-12
                ):
                    logger.info(
                        "DFLASH adaptive throughput-proxy reward exponents: tau_exp=%.6f time_exp=%.6f",
                        self._adaptive_proxy_tau_exp,
                        self._adaptive_proxy_time_exp,
                    )
            if self._confidence_gate_enabled:
                logger.info(
                    "DFLASH confidence-gated verify enabled. mode=%s threshold=%.4f score_metric=%s score_budget=%.4f aggregate=%s min_verify_tokens=%d mab=%s mab_algo=%s mab_ucb_c=%.4f mab_arms=%s grouped_verify=%s grouped_buckets=%s",
                    self._confidence_gate_mode,
                    self._confidence_gate_threshold,
                    self._confidence_gate_score_metric,
                    self._confidence_gate_score_budget,
                    self._confidence_gate_aggregate,
                    self._confidence_gate_min_verify_tokens,
                    self._confidence_gate_mab_enabled,
                    self._confidence_gate_mab_algo,
                    self._confidence_gate_mab_ucb_c,
                    self._confidence_gate_mab_arms,
                    self._confidence_gate_grouped_verify_enabled,
                    self._confidence_gate_grouped_verify_buckets,
                )
            if self._report_cycle_trace:
                logger.info("DFLASH per-cycle trace enabled.")
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )
            predictor_output_dir = getattr(
                server_args,
                "speculative_dflash_predictor_dataset_output_dir",
                None,
            )
            if predictor_output_dir:
                shard_rows = int(
                    getattr(
                        server_args,
                        "speculative_dflash_predictor_dataset_shard_rows",
                        100000,
                    )
                )
                self._predictor_dataset_writer = PredictorDatasetShardWriter(
                    output_dir=str(predictor_output_dir),
                    worker_tag=(
                        f"gpu{int(self.gpu_id)}_tp{int(self.tp_rank)}_dp"
                        f"{int(self.dp_rank) if self.dp_rank is not None else 0}"
                    ),
                    shard_max_rows=shard_rows,
                )
                logger.info(
                    "DFLASH predictor dataset dump enabled. output_dir=%s shard_rows=%d",
                    str(predictor_output_dir),
                    shard_rows,
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

    def _select_confidence_gate_threshold(self) -> tuple[float, str, Optional[dict[float, float]]]:
        if not self._confidence_gate_enabled:
            return float(self._confidence_gate_threshold), "disabled", None
        if not self._confidence_gate_mab_enabled:
            return float(self._confidence_gate_threshold), "fixed", None

        arms = [float(v) for v in self._confidence_gate_mab_arms]
        unseen = [
            float(a)
            for a in arms
            if int(self._confidence_gate_mab_counts.get(float(a), 0)) <= 0
        ]
        if unseen:
            return float(unseen[0]), "mab_warmup", None

        if self._confidence_gate_mab_algo == "thompson":
            samples: dict[float, float] = {}
            for arm in arms:
                a = float(max(self._confidence_gate_mab_alpha.get(float(arm), 1.0), 1e-6))
                b = float(max(self._confidence_gate_mab_beta.get(float(arm), 1.0), 1e-6))
                samples[float(arm)] = float(random.betavariate(a, b))
            selected = float(max(arms, key=lambda arm: (samples[float(arm)], -float(arm))))
            return selected, "mab_thompson", samples

        # Default UCB-style selection.
        rounds = int(max(self._confidence_gate_mab_rounds, 1))
        c = float(max(self._confidence_gate_mab_ucb_c, 0.0))
        log_t = math.log(max(float(rounds), 2.0))
        scores: dict[float, float] = {}
        for arm in arms:
            arm_f = float(arm)
            n = float(max(int(self._confidence_gate_mab_counts.get(arm_f, 0)), 1))
            mean = float(self._confidence_gate_mab_reward_sums.get(arm_f, 0.0)) / n
            bonus = c * math.sqrt((2.0 * log_t) / n)
            scores[arm_f] = float(mean + bonus)
        selected = float(max(arms, key=lambda arm: (scores[float(arm)], -float(arm))))
        return selected, "mab_ucb", scores

    def _aggregate_confidence_gate_verify_tokens(
        self,
        *,
        per_req_verify_tokens: torch.Tensor,
        runtime_block_size: int,
    ) -> int:
        vals = per_req_verify_tokens.to(dtype=torch.float32)
        agg = self._confidence_gate_aggregate
        if agg == "min":
            out = float(torch.min(vals).item())
        elif agg == "max":
            out = float(torch.max(vals).item())
        elif agg == "mean":
            out = float(torch.mean(vals).item())
        elif agg == "median":
            out = float(torch.quantile(vals, q=0.5).item())
        elif agg == "q25":
            out = float(torch.quantile(vals, q=0.25).item())
        elif agg == "q50":
            out = float(torch.quantile(vals, q=0.50).item())
        elif agg == "q75":
            out = float(torch.quantile(vals, q=0.75).item())
        elif agg == "q90":
            out = float(torch.quantile(vals, q=0.90).item())
        else:
            # Default to conservative q10 aggregation.
            out = float(torch.quantile(vals, q=0.10).item())
        out_i = int(round(out))
        out_i = int(max(out_i, int(self._confidence_gate_min_verify_tokens)))
        out_i = int(min(out_i, int(runtime_block_size)))
        return out_i

    def _compute_confidence_gated_verify_tokens(
        self,
        *,
        draft_max_probs: Optional[torch.Tensor],
        draft_entropies: Optional[torch.Tensor],
        runtime_block_size: int,
    ) -> tuple[int, dict, torch.Tensor]:
        if self._confidence_gate_mode == "score":
            return self._compute_score_gated_verify_tokens(
                draft_max_probs=draft_max_probs,
                draft_entropies=draft_entropies,
                runtime_block_size=runtime_block_size,
            )
        return self._compute_threshold_gated_verify_tokens(
            draft_max_probs=draft_max_probs,
            runtime_block_size=runtime_block_size,
        )

    def _compute_threshold_gated_verify_tokens(
        self,
        *,
        draft_max_probs: torch.Tensor,
        runtime_block_size: int,
    ) -> tuple[int, dict, torch.Tensor]:
        threshold, select_reason, mab_scores = self._select_confidence_gate_threshold()

        # draft_max_probs: [bs, runtime_block_size-1], aligned with drafted positions [1..k-1].
        bs = int(draft_max_probs.shape[0]) if draft_max_probs is not None else 0
        per_req_verify_tokens = torch.full(
            (bs,),
            int(runtime_block_size),
            dtype=torch.int32,
            device=self.device,
        )
        if (
            draft_max_probs is not None
            and draft_max_probs.numel() > 0
            and int(runtime_block_size) > 1
        ):
            below = draft_max_probs < float(threshold)
            has_below = torch.any(below, dim=1)
            first_idx = torch.argmax(below.to(torch.int32), dim=1) + 1
            per_req_verify_tokens = torch.where(
                has_below,
                first_idx.to(torch.int32),
                per_req_verify_tokens,
            )

        per_req_verify_tokens = torch.clamp(
            per_req_verify_tokens,
            min=int(self._confidence_gate_min_verify_tokens),
            max=int(runtime_block_size),
        )
        verify_token_num = self._aggregate_confidence_gate_verify_tokens(
            per_req_verify_tokens=per_req_verify_tokens,
            runtime_block_size=int(runtime_block_size),
        )

        decision = {
            "enabled": True,
            "mode": "threshold",
            "threshold": float(threshold),
            "selection_reason": str(select_reason),
            "aggregate": str(self._confidence_gate_aggregate),
            "min_verify_tokens": int(self._confidence_gate_min_verify_tokens),
            "runtime_block_size": int(runtime_block_size),
            "verify_token_num": int(verify_token_num),
            "per_req_verify_tokens": [int(v) for v in per_req_verify_tokens.tolist()],
            "mab_enabled": bool(self._confidence_gate_mab_enabled),
            "mab_algo": str(self._confidence_gate_mab_algo),
            "mab_round": int(self._confidence_gate_mab_rounds),
            "mab_scores": (
                {float(k): float(v) for k, v in mab_scores.items()}
                if isinstance(mab_scores, dict)
                else None
            ),
        }
        return int(verify_token_num), decision, per_req_verify_tokens

    def _compute_score_gated_verify_tokens(
        self,
        *,
        draft_max_probs: Optional[torch.Tensor],
        draft_entropies: Optional[torch.Tensor],
        runtime_block_size: int,
    ) -> tuple[int, dict, torch.Tensor]:
        score_values = draft_max_probs
        if self._confidence_gate_score_metric == "entropy":
            score_values = draft_entropies

        # score_values: [bs, runtime_block_size-1], aligned with drafted positions [1..k-1].
        bs = int(score_values.shape[0]) if score_values is not None else 0
        per_req_verify_tokens = torch.full(
            (bs,),
            int(runtime_block_size),
            dtype=torch.int32,
            device=self.device,
        )
        score_budget = float(self._confidence_gate_score_budget)
        per_req_score_budget_used = torch.zeros(
            (bs,),
            dtype=torch.float32,
            device=self.device,
        )

        if (
            score_values is not None
            and score_values.numel() > 0
            and int(runtime_block_size) > 1
        ):
            if self._confidence_gate_score_metric == "neg_log_max_prob":
                safe_probs = torch.clamp(
                    score_values.to(dtype=torch.float32),
                    min=1e-6,
                    max=1.0,
                )
                per_token_scores = -torch.log(safe_probs)
            elif self._confidence_gate_score_metric == "entropy":
                per_token_scores = torch.clamp(
                    score_values.to(dtype=torch.float32),
                    min=0.0,
                )
            else:
                raise RuntimeError(
                    "Unsupported confidence-gate score metric: "
                    f"{self._confidence_gate_score_metric!r}."
                )
            cumulative_scores = torch.cumsum(per_token_scores, dim=1)
            prefix_keep_counts = torch.sum(
                (cumulative_scores <= score_budget).to(torch.int32),
                dim=1,
            )
            per_req_verify_tokens = prefix_keep_counts.to(torch.int32) + 1
            clamped_keep_counts = torch.clamp(
                prefix_keep_counts,
                min=0,
                max=max(int(runtime_block_size) - 1, 0),
            ).to(torch.long)
            has_kept_prefix = clamped_keep_counts > 0
            if cumulative_scores.shape[1] > 0:
                gather_idx = torch.clamp(clamped_keep_counts - 1, min=0)
                gathered = torch.gather(cumulative_scores, 1, gather_idx.unsqueeze(1))
                per_req_score_budget_used = torch.where(
                    has_kept_prefix,
                    gathered.squeeze(1),
                    per_req_score_budget_used,
                )

        per_req_verify_tokens = torch.clamp(
            per_req_verify_tokens,
            min=int(self._confidence_gate_min_verify_tokens),
            max=int(runtime_block_size),
        )
        verify_token_num = self._aggregate_confidence_gate_verify_tokens(
            per_req_verify_tokens=per_req_verify_tokens,
            runtime_block_size=int(runtime_block_size),
        )

        decision = {
            "enabled": True,
            "mode": "score",
            "score_metric": str(self._confidence_gate_score_metric),
            "score_budget": float(score_budget),
            "selection_reason": "score_budget",
            "aggregate": str(self._confidence_gate_aggregate),
            "min_verify_tokens": int(self._confidence_gate_min_verify_tokens),
            "runtime_block_size": int(runtime_block_size),
            "verify_token_num": int(verify_token_num),
            "per_req_verify_tokens": [int(v) for v in per_req_verify_tokens.tolist()],
            "per_req_score_budget_used": [
                float(v) for v in per_req_score_budget_used.tolist()
            ],
            "mab_enabled": False,
            "mab_algo": str(self._confidence_gate_mab_algo),
            "mab_round": int(self._confidence_gate_mab_rounds),
            "mab_scores": None,
        }
        return int(verify_token_num), decision, per_req_verify_tokens

    def _build_grouped_verify_plan(
        self,
        *,
        per_req_verify_tokens: torch.Tensor,
        runtime_block_size: int,
    ) -> list[tuple[int, list[int]]]:
        """Build grouped verify plan: (verify_k, request_indices)."""
        if per_req_verify_tokens is None or per_req_verify_tokens.numel() == 0:
            return []

        runtime_k = int(max(1, runtime_block_size))
        raw_vals = [
            int(min(max(int(v), 1), runtime_k))
            for v in per_req_verify_tokens.tolist()
        ]

        buckets = [
            int(v)
            for v in self._confidence_gate_grouped_verify_buckets
            if 1 <= int(v) <= runtime_k
        ]
        buckets = sorted(set(buckets))

        mapped_vals: list[int] = []
        if buckets:
            for k in raw_vals:
                selected = None
                for b in buckets:
                    if b >= k:
                        selected = int(b)
                        break
                if selected is None:
                    selected = int(buckets[-1])
                mapped_vals.append(int(selected))
        else:
            mapped_vals = raw_vals

        grouped: dict[int, list[int]] = {}
        for req_idx, k in enumerate(mapped_vals):
            grouped.setdefault(int(k), []).append(int(req_idx))

        # Smaller verify-k first keeps short paths responsive and limits KV transient pressure.
        return sorted(grouped.items(), key=lambda x: int(x[0]))

    def _build_verify_sub_batch(
        self,
        *,
        batch: ScheduleBatch,
        keep_indices: list[int],
        keep_indices_device: torch.Tensor,
    ) -> ScheduleBatch:
        """Build a lightweight per-group view of the running batch for grouped verify."""
        sub_batch = copy.copy(batch)
        sub_batch.reqs = [batch.reqs[i] for i in keep_indices]
        sub_batch.req_pool_indices = batch.req_pool_indices[keep_indices_device]
        sub_batch.seq_lens = batch.seq_lens[keep_indices_device]
        if isinstance(batch.seq_lens_cpu, torch.Tensor):
            sub_batch.seq_lens_cpu = batch.seq_lens_cpu[keep_indices]
        else:
            sub_batch.seq_lens_cpu = [batch.seq_lens_cpu[i] for i in keep_indices]
        sub_batch.orig_seq_lens = batch.orig_seq_lens[keep_indices_device]
        sub_batch.seq_lens_sum = int(sub_batch.seq_lens.sum().item())
        sub_batch.out_cache_loc = None
        sub_batch.forward_mode = ForwardMode.TARGET_VERIFY
        sub_batch.spec_info = None

        if batch.output_ids is not None:
            sub_batch.output_ids = batch.output_ids[keep_indices_device]
        else:
            sub_batch.output_ids = None

        if batch.multimodal_inputs is not None:
            sub_batch.multimodal_inputs = [batch.multimodal_inputs[i] for i in keep_indices]
        else:
            sub_batch.multimodal_inputs = None

        sub_batch.return_logprob = any(req.return_logprob for req in sub_batch.reqs)
        if sub_batch.return_logprob:
            sub_batch.top_logprobs_nums = (
                [batch.top_logprobs_nums[i] for i in keep_indices]
                if batch.top_logprobs_nums is not None
                else None
            )
            sub_batch.token_ids_logprobs = [
                batch.token_ids_logprobs[i] for i in keep_indices
            ] if batch.token_ids_logprobs is not None else None
        else:
            sub_batch.top_logprobs_nums = None
            sub_batch.token_ids_logprobs = None

        sub_batch.has_stream = any(req.stream for req in sub_batch.reqs)
        sub_batch.has_grammar = any(req.grammar for req in sub_batch.reqs)

        if batch.sampling_info is not None:
            sub_sampling_info = copy.deepcopy(batch.sampling_info)
            sub_sampling_info.filter_batch(keep_indices, keep_indices_device)
            sub_batch.sampling_info = sub_sampling_info
        else:
            sub_batch.sampling_info = None

        sub_batch.mamba_track_indices = None
        sub_batch.mamba_track_mask = None
        sub_batch.mamba_track_seqlens = None
        return sub_batch

    def _update_confidence_gate_state(
        self,
        *,
        decision: Optional[dict],
        accept_length_per_req_cpu: list[int],
        cycle_e2e_s: float,
    ) -> None:
        if not self._confidence_gate_enabled:
            self._last_confidence_gate_decision = None
            return
        if not isinstance(decision, dict):
            self._last_confidence_gate_decision = {
                "enabled": True,
                "selection_reason": "missing_decision",
            }
            return

        proposed = max(int(decision.get("verify_token_num", 1)) - 1, 0)
        per_req_verify_tokens = decision.get("per_req_verify_tokens", None)
        accepted_sum = float(sum(max(0, int(v)) for v in accept_length_per_req_cpu))
        # Cycle-level tau includes target bonus token.
        tau_sum = float(
            sum(max(0, int(v)) + 1 for v in accept_length_per_req_cpu)
        )
        proposed_total = 0
        if isinstance(per_req_verify_tokens, list) and len(per_req_verify_tokens) > 0:
            proposed_total = int(
                sum(max(0, int(v) - 1) for v in per_req_verify_tokens)
            )
        elif proposed > 0 and len(accept_length_per_req_cpu) > 0:
            proposed_total = int(len(accept_length_per_req_cpu) * proposed)

        if proposed_total > 0:
            accept_ratio = accepted_sum / float(proposed_total)
        else:
            accept_ratio = 1.0

        if cycle_e2e_s > 0.0:
            reward = tau_sum / max(float(cycle_e2e_s), 1e-6)
            reward_source = "tau_per_cycle_e2e_s"
        else:
            reward = float(accept_ratio)
            reward_source = "accept_ratio_fallback"

        decision["reward"] = float(reward)
        decision["reward_source"] = str(reward_source)
        decision["accepted_draft_tokens_sum"] = float(accepted_sum)
        decision["tau_sum"] = float(tau_sum)
        decision["proposed_draft_tokens_sum"] = float(proposed_total)
        decision["accept_ratio"] = float(accept_ratio)
        decision["cycle_e2e_s"] = (
            float(cycle_e2e_s) if cycle_e2e_s > 0.0 else None
        )

        if self._confidence_gate_mab_enabled:
            selected = float(decision.get("threshold", self._confidence_gate_threshold))
            counts = self._confidence_gate_mab_counts
            reward_sums = self._confidence_gate_mab_reward_sums
            if selected not in counts:
                selected = float(
                    min(
                        self._confidence_gate_mab_arms,
                        key=lambda v: abs(float(v) - float(selected)),
                    )
                )
            if self._confidence_gate_mab_algo == "thompson":
                reward_norm_max = float(
                    max(self._confidence_gate_mab_reward_norm_max, float(reward), 1e-6)
                )
                self._confidence_gate_mab_reward_norm_max = reward_norm_max
                reward_norm = float(min(max(float(reward) / reward_norm_max, 0.0), 1.0))
                self._confidence_gate_mab_alpha[selected] = float(
                    self._confidence_gate_mab_alpha.get(selected, 1.0)
                ) + reward_norm
                self._confidence_gate_mab_beta[selected] = float(
                    self._confidence_gate_mab_beta.get(selected, 1.0)
                ) + (1.0 - reward_norm)
                decision["mab_reward_norm"] = float(reward_norm)
                decision["mab_reward_norm_max"] = float(reward_norm_max)
                decision["mab_selected_alpha"] = float(
                    self._confidence_gate_mab_alpha.get(selected, 1.0)
                )
                decision["mab_selected_beta"] = float(
                    self._confidence_gate_mab_beta.get(selected, 1.0)
                )

            counts[selected] = int(counts.get(selected, 0)) + 1
            reward_sums[selected] = float(reward_sums.get(selected, 0.0)) + float(reward)
            self._confidence_gate_mab_rounds = int(self._confidence_gate_mab_rounds) + 1
            decision["mab_selected_count"] = int(counts.get(selected, 0))
            decision["mab_selected_mean_reward"] = float(
                float(reward_sums.get(selected, 0.0))
                / float(max(int(counts.get(selected, 0)), 1))
            )
            decision["mab_round"] = int(self._confidence_gate_mab_rounds)

        self._last_confidence_gate_decision = decision

    def _ensure_req_ucb_state(
        self, req
    ) -> tuple[dict[int, int], dict[int, float], dict[int, float], dict[int, float]]:
        arms = self._adaptive_arms()
        counts = getattr(req, "dflash_adaptive_ucb_counts", None)
        reward_sums = getattr(req, "dflash_adaptive_ucb_reward_sums", None)
        gain_sums = getattr(req, "dflash_adaptive_ucb_gain_sums", None)
        cost_sums = getattr(req, "dflash_adaptive_ucb_cost_sums", None)
        if not isinstance(counts, dict):
            counts = {}
        if not isinstance(reward_sums, dict):
            reward_sums = {}
        if not isinstance(gain_sums, dict):
            gain_sums = {}
        if not isinstance(cost_sums, dict):
            cost_sums = {}

        # Keep state aligned with current arm space.
        normalized_counts: dict[int, int] = {}
        normalized_sums: dict[int, float] = {}
        normalized_gain_sums: dict[int, float] = {}
        normalized_cost_sums: dict[int, float] = {}
        for arm in arms:
            arm_i = int(arm)
            normalized_counts[arm_i] = int(counts.get(arm_i, 0) or 0)
            normalized_sums[arm_i] = float(reward_sums.get(arm_i, 0.0) or 0.0)
            normalized_gain_sums[arm_i] = float(gain_sums.get(arm_i, 0.0) or 0.0)
            normalized_cost_sums[arm_i] = float(cost_sums.get(arm_i, 0.0) or 0.0)

        req.dflash_adaptive_ucb_counts = normalized_counts
        req.dflash_adaptive_ucb_reward_sums = normalized_sums
        req.dflash_adaptive_ucb_gain_sums = normalized_gain_sums
        req.dflash_adaptive_ucb_cost_sums = normalized_cost_sums
        if getattr(req, "dflash_adaptive_ucb_rounds", None) is None:
            req.dflash_adaptive_ucb_rounds = 0
        if getattr(req, "dflash_adaptive_ucb_reward_norm_max", None) is None:
            req.dflash_adaptive_ucb_reward_norm_max = 1.0
        return normalized_counts, normalized_sums, normalized_gain_sums, normalized_cost_sums

    def _ensure_req_linucb_state(self, req):
        arms = self._adaptive_arms()
        dim = int(self._adaptive_linucb_dim)
        lam = float(max(self._adaptive_linucb_lambda, 1e-9))

        counts = getattr(req, "dflash_adaptive_linucb_counts", None)
        a_inv = getattr(req, "dflash_adaptive_linucb_A_inv", None)
        b_vec = getattr(req, "dflash_adaptive_linucb_b", None)
        if not isinstance(counts, dict):
            counts = {}
        if not isinstance(a_inv, dict):
            a_inv = {}
        if not isinstance(b_vec, dict):
            b_vec = {}

        eye = torch.eye(dim, dtype=torch.float64, device="cpu") * (1.0 / lam)
        normalized_counts: dict[int, int] = {}
        normalized_a_inv: dict[int, torch.Tensor] = {}
        normalized_b: dict[int, torch.Tensor] = {}

        for arm in arms:
            arm_i = int(arm)
            normalized_counts[arm_i] = int(counts.get(arm_i, 0) or 0)

            cur_a_inv = a_inv.get(arm_i, None)
            if (
                isinstance(cur_a_inv, torch.Tensor)
                and cur_a_inv.dim() == 2
                and int(cur_a_inv.shape[0]) == dim
                and int(cur_a_inv.shape[1]) == dim
            ):
                normalized_a_inv[arm_i] = cur_a_inv.to(
                    device="cpu", dtype=torch.float64
                )
            else:
                normalized_a_inv[arm_i] = eye.clone()

            cur_b = b_vec.get(arm_i, None)
            if (
                isinstance(cur_b, torch.Tensor)
                and cur_b.dim() == 1
                and int(cur_b.shape[0]) == dim
            ):
                normalized_b[arm_i] = cur_b.to(device="cpu", dtype=torch.float64)
            else:
                normalized_b[arm_i] = torch.zeros(dim, dtype=torch.float64, device="cpu")

        req.dflash_adaptive_linucb_counts = normalized_counts
        req.dflash_adaptive_linucb_A_inv = normalized_a_inv
        req.dflash_adaptive_linucb_b = normalized_b
        if getattr(req, "dflash_adaptive_linucb_rounds", None) is None:
            req.dflash_adaptive_linucb_rounds = 0
        return normalized_counts, normalized_a_inv, normalized_b

    def _ensure_req_thompson_state(
        self, req
    ) -> tuple[dict[int, int], dict[int, float], dict[int, float]]:
        arms = self._adaptive_arms()
        counts = getattr(req, "dflash_adaptive_thompson_counts", None)
        alpha = getattr(req, "dflash_adaptive_thompson_alpha", None)
        beta = getattr(req, "dflash_adaptive_thompson_beta", None)
        if not isinstance(counts, dict):
            counts = {}
        if not isinstance(alpha, dict):
            alpha = {}
        if not isinstance(beta, dict):
            beta = {}

        normalized_counts: dict[int, int] = {}
        normalized_alpha: dict[int, float] = {}
        normalized_beta: dict[int, float] = {}
        for arm in arms:
            arm_i = int(arm)
            normalized_counts[arm_i] = int(counts.get(arm_i, 0) or 0)
            # Beta(1,1) prior by default.
            normalized_alpha[arm_i] = float(alpha.get(arm_i, 1.0) or 1.0)
            normalized_beta[arm_i] = float(beta.get(arm_i, 1.0) or 1.0)

        req.dflash_adaptive_thompson_counts = normalized_counts
        req.dflash_adaptive_thompson_alpha = normalized_alpha
        req.dflash_adaptive_thompson_beta = normalized_beta
        if getattr(req, "dflash_adaptive_thompson_rounds", None) is None:
            req.dflash_adaptive_thompson_rounds = 0
        if getattr(req, "dflash_adaptive_thompson_reward_norm_max", None) is None:
            req.dflash_adaptive_thompson_reward_norm_max = 1.0
        return normalized_counts, normalized_alpha, normalized_beta

    def _build_linucb_context(
        self,
        *,
        req,
        current_bs: int,
        accept_ratio: float,
        accepted_draft_tokens: int,
        proposed_draft_tokens: int,
        num_active_reqs: int,
    ) -> torch.Tensor:
        max_bs = float(max(int(self.block_size), 1))
        max_running = float(max(int(self.server_args.max_running_requests or 1), 1))
        cur_bs = float(max(1, int(current_bs)))
        proposed = float(max(0, int(proposed_draft_tokens)))
        accepted = float(max(0, int(accepted_draft_tokens)))
        acc_ewma = getattr(req, "dflash_adaptive_accept_ratio_ewma", None)
        if acc_ewma is None:
            acc_ewma = float(accept_ratio)
        acc_ewma = float(max(0.0, min(1.0, float(acc_ewma))))
        context_vals = [
            1.0,
            cur_bs / max_bs,
            float(max(1, int(num_active_reqs))) / max_running,
            float(max(0.0, min(1.0, float(accept_ratio)))),
            acc_ewma,
            proposed / max(max_bs - 1.0, 1.0),
            accepted / max(max_bs - 1.0, 1.0),
        ]
        return torch.tensor(context_vals, dtype=torch.float64, device="cpu")

    def _parse_adaptive_proxy_cycle_ms(
        self, raw: str
    ) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
        # Supports both:
        # - '<k>:<ms>' (legacy)
        # - '<c>x<k>:<ms>' (concurrency-specific)
        out_by_bs: dict[int, float] = {}
        out_by_ck: dict[tuple[int, int], float] = {}
        if not raw:
            return out_by_bs, out_by_ck
        for token in str(raw).split(","):
            item = token.strip()
            if not item:
                continue
            if ":" not in item:
                if self.tp_rank == 0:
                    logger.warning(
                        "Ignoring malformed throughput-proxy cycle-ms entry '%s'. "
                        "Expected '<block_size>:<ms>'.",
                        item,
                    )
                continue
            k_str, v_str = item.split(":", 1)
            try:
                key = k_str.strip().lower()
                v = float(v_str.strip())
                if v <= 0.0:
                    raise ValueError()
                if "x" in key:
                    # concurrency-specific entry: '<c>x<k>:<ms>'
                    c_raw, k_raw = key.split("x", 1)
                    c = int(c_raw.strip())
                    k = int(k_raw.strip())
                    if c < 1 or k < 1:
                        raise ValueError()
                    out_by_ck[(c, k)] = v
                else:
                    k = int(key)
                    if k < 1:
                        raise ValueError()
                    out_by_bs[k] = v
            except Exception:
                if self.tp_rank == 0:
                    logger.warning(
                        "Ignoring invalid throughput-proxy cycle-ms entry '%s'. "
                        "Expected '<k>:<ms>' or '<c>x<k>:<ms>' with positive values.",
                        item,
                    )
        return out_by_bs, out_by_ck

    def _estimate_proxy_cycle_ms(
        self, *, runtime_block_size: int, num_active_reqs: int
    ) -> tuple[float, str]:
        bs = int(max(int(runtime_block_size), 1))
        c = int(max(int(num_active_reqs), 1))

        if (c, bs) in self._adaptive_proxy_cycle_ms_by_ck:
            return (
                float(self._adaptive_proxy_cycle_ms_by_ck[(c, bs)]),
                "throughput_proxy_ck_map",
            )

        if bs in self._adaptive_proxy_cycle_ms_by_bs:
            return (
                float(self._adaptive_proxy_cycle_ms_by_bs[bs]),
                "throughput_proxy_bs_map",
            )

        if self._adaptive_proxy_powerlaw_a > 0.0:
            est = (
                float(self._adaptive_proxy_powerlaw_a)
                * (float(c) ** float(self._adaptive_proxy_powerlaw_c_exp))
                * (float(bs) ** float(self._adaptive_proxy_powerlaw_k_exp))
            )
            return (float(max(est, 1e-6)), "throughput_proxy_powerlaw")

        return float(bs), "throughput_proxy_fallback_bs_units"

    def _compute_ucb_reward(
        self,
        *,
        req,
        accepted_draft_tokens: int,
        runtime_block_size: int,
        draft_time_s: float,
        verify_time_s: float,
        cycle_e2e_s: float,
        num_active_reqs: int,
    ) -> tuple[float, str, float]:
        # Include the target bonus token to match cycle-level accept length.
        accept_length = float(max(0, int(accepted_draft_tokens)) + 1)

        if self._adaptive_reward_mode == "accept_length":
            return accept_length, "accept_length", accept_length
        if self._adaptive_reward_mode == "throughput_cycle_e2e":
            if cycle_e2e_s <= 0.0:
                return (
                    accept_length,
                    "throughput_cycle_e2e_fallback_no_timing",
                    accept_length,
                )
            raw = accept_length / max(float(cycle_e2e_s), 1e-6)
            return raw, "throughput_cycle_e2e", raw
        if self._adaptive_reward_mode == "throughput_cycle_rate":
            if cycle_e2e_s <= 0.0:
                return (
                    accept_length,
                    "throughput_cycle_rate_fallback_no_timing",
                    accept_length,
                )
            raw = accept_length / max(float(cycle_e2e_s), 1e-6)
            return raw, "throughput_cycle_rate", raw
        if self._adaptive_reward_mode == "throughput_proxy":
            # Cost-aware proxy that avoids runtime timing sync:
            # reward ~ tau^tau_exp / est_cycle_ms^time_exp.
            est_ms, source = self._estimate_proxy_cycle_ms(
                runtime_block_size=int(runtime_block_size),
                num_active_reqs=int(num_active_reqs),
            )
            tau_exp = float(max(self._adaptive_proxy_tau_exp, 1e-6))
            time_exp = float(max(self._adaptive_proxy_time_exp, 1e-6))
            raw = (accept_length ** tau_exp) / (max(est_ms, 1e-6) ** time_exp)
            return raw, source, raw

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
            elif self._adaptive_algo == "linucb":
                self._ensure_req_linucb_state(req)
            elif self._adaptive_algo == "thompson":
                self._ensure_req_thompson_state(req)
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
        elif self._adaptive_algo == "linucb":
            self._ensure_req_linucb_state(req)
            req.dflash_adaptive_linucb_rounds = 0
            req.dflash_adaptive_linucb_last_scores = None
        elif self._adaptive_algo == "thompson":
            self._ensure_req_thompson_state(req)
            req.dflash_adaptive_thompson_rounds = 0
            req.dflash_adaptive_thompson_reward_norm_max = 1.0
            req.dflash_adaptive_thompson_last_scores = None
        return init_bs

    def _resolve_runtime_block_size(self, batch: ScheduleBatch) -> int:
        """Resolve runtime DFLASH block size for the current decode step.

        Modes:
        - Adaptive mode ON: use per-request adaptive state and choose an effective
          batch block size as the maximum desired block size among active requests.
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

            # Use the largest requested runtime block size for the batch.
            # This favors higher speculative length when requests disagree.
            effective_block_size = int(max(desired))

            if (
                len(set(desired)) > 1
                and not self._warned_mixed_runtime_block_size
                and self.tp_rank == 0
            ):
                logger.info(
                    "DFLASH adaptive per-request desired block sizes are mixed (%s); "
                    "using max=%d for this step.",
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
        cycle_e2e_s: float = 0.0,
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
            counts, reward_sums, gain_sums, cost_sums = self._ensure_req_ucb_state(req)
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
                runtime_block_size=int(runtime_block_size),
                draft_time_s=float(draft_time_s),
                verify_time_s=float(verify_time_s),
                cycle_e2e_s=float(cycle_e2e_s),
                num_active_reqs=int(num_active_reqs),
            )

            # Rate objective for throughput optimization: mean_k = sum(gain)/sum(cost),
            # where gain is accepted length and cost is measured end-to-end cycle time.
            gain = float(max(0, int(accepted_draft_tokens)) + 1)
            cost = float(max(float(cycle_e2e_s), 1e-6))

            counts[pulled_arm] = int(counts[pulled_arm]) + 1
            reward_sums[pulled_arm] = float(reward_sums[pulled_arm]) + float(reward)
            gain_sums[pulled_arm] = float(gain_sums[pulled_arm]) + float(gain)
            cost_sums[pulled_arm] = float(cost_sums[pulled_arm]) + float(cost)
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
                elif self._adaptive_reward_mode == "throughput_cycle_rate":
                    c = max(float(self._adaptive_ucb_c), 0.0)
                    log_t = math.log(max(float(rounds), 2.0))
                    for arm in arms:
                        n = float(max(int(counts.get(arm, 0)), 1))
                        arm_cost = float(max(float(cost_sums.get(arm, 0.0)), 1e-6))
                        arm_gain = float(gain_sums.get(arm, 0.0))
                        mean = arm_gain / arm_cost
                        bonus = c * math.sqrt((2.0 * log_t) / n)
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
                "ucb_pulled_arm_gain_sum": float(gain_sums.get(pulled_arm, 0.0)),
                "ucb_pulled_arm_cost_sum_s": float(cost_sums.get(pulled_arm, 0.0)),
                "ucb_pulled_arm_rate": float(
                    float(gain_sums.get(pulled_arm, 0.0))
                    / float(max(float(cost_sums.get(pulled_arm, 0.0)), 1e-6))
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

        if self._adaptive_algo == "linucb":
            counts, a_inv_map, b_map = self._ensure_req_linucb_state(req)
            arms = sorted(int(a) for a in counts.keys())

            pulled_arm = int(runtime_block_size)
            pulled_arm = int(
                min(max(pulled_arm, int(self._adaptive_k_min)), int(self._adaptive_k_max))
            )
            pulled_arm = self._clamp_runtime_block_size(pulled_arm)
            if self._adaptive_block_buckets:
                pulled_arm = self._snap_to_adaptive_bucket(pulled_arm, mode="nearest")
            if pulled_arm not in counts:
                dim = int(self._adaptive_linucb_dim)
                lam = float(max(self._adaptive_linucb_lambda, 1e-9))
                counts[pulled_arm] = 0
                a_inv_map[pulled_arm] = (
                    torch.eye(dim, dtype=torch.float64, device="cpu") * (1.0 / lam)
                )
                b_map[pulled_arm] = torch.zeros(dim, dtype=torch.float64, device="cpu")
                arms = sorted(int(a) for a in counts.keys())

            context = self._build_linucb_context(
                req=req,
                current_bs=int(current_bs),
                accept_ratio=float(accept_ratio),
                accepted_draft_tokens=int(accepted),
                proposed_draft_tokens=int(proposed),
                num_active_reqs=int(num_active_reqs),
            )
            reward, reward_source, reward_raw = self._compute_ucb_reward(
                req=req,
                accepted_draft_tokens=accepted,
                runtime_block_size=int(runtime_block_size),
                draft_time_s=float(draft_time_s),
                verify_time_s=float(verify_time_s),
                cycle_e2e_s=float(cycle_e2e_s),
                num_active_reqs=int(num_active_reqs),
            )

            # Online ridge update via Sherman-Morrison on A^{-1}.
            a_inv = a_inv_map[pulled_arm]
            b_vec = b_map[pulled_arm]
            ax = torch.mv(a_inv, context)
            denom = float(1.0 + torch.dot(context, ax))
            if denom > 1e-12:
                a_inv = a_inv - torch.ger(ax, ax) / denom
            a_inv_map[pulled_arm] = a_inv
            b_map[pulled_arm] = b_vec + float(reward) * context
            counts[pulled_arm] = int(counts[pulled_arm]) + 1
            rounds = int(getattr(req, "dflash_adaptive_linucb_rounds", 0) or 0) + 1
            req.dflash_adaptive_linucb_rounds = int(rounds)

            unseen_arms = [int(a) for a in arms if int(counts.get(int(a), 0)) == 0]
            means: dict[int, float] = {}
            bonuses: dict[int, float] = {}
            scores: dict[int, float] = {}
            if unseen_arms:
                next_bs = int(unseen_arms[0])
                reason = "linucb_warmup"
            else:
                alpha = float(max(self._adaptive_linucb_alpha, 0.0))
                for arm in arms:
                    arm_a_inv = a_inv_map[arm]
                    arm_b = b_map[arm]
                    theta = torch.mv(arm_a_inv, arm_b)
                    mean = float(torch.dot(theta, context))
                    quad = float(torch.dot(context, torch.mv(arm_a_inv, context)))
                    bonus = alpha * math.sqrt(max(quad, 0.0))
                    means[arm] = mean
                    bonuses[arm] = bonus
                    scores[arm] = mean + bonus
                next_bs = int(max(arms, key=lambda a: (scores[a], -int(a))))
                reason = "linucb_score"

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
            req.dflash_adaptive_linucb_last_scores = (
                {int(a): float(scores[a]) for a in scores} if scores else None
            )
            req.dflash_adaptive_last_decision = {
                "algo": "linucb",
                "reward_mode": self._adaptive_reward_mode,
                "reward_source": reward_source,
                "reward": float(reward),
                "reward_raw": float(reward_raw),
                "linucb_round": int(rounds),
                "linucb_pulled_arm": int(pulled_arm),
                "linucb_pulled_arm_count": int(counts.get(pulled_arm, 0)),
                "linucb_selected_mean_reward": (
                    float(means[next_bs]) if next_bs in means else None
                ),
                "linucb_selected_bonus": (
                    float(bonuses[next_bs]) if next_bs in bonuses else None
                ),
                "linucb_selected_score": (
                    float(scores[next_bs]) if next_bs in scores else None
                ),
                "linucb_alpha": float(self._adaptive_linucb_alpha),
                "linucb_lambda": float(self._adaptive_linucb_lambda),
                "linucb_context": [float(v) for v in context.tolist()],
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

        if self._adaptive_algo == "thompson":
            counts, alpha_map, beta_map = self._ensure_req_thompson_state(req)
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
                alpha_map[pulled_arm] = 1.0
                beta_map[pulled_arm] = 1.0
                arms = sorted(int(a) for a in counts.keys())

            reward, reward_source, reward_raw = self._compute_ucb_reward(
                req=req,
                accepted_draft_tokens=accepted,
                runtime_block_size=int(runtime_block_size),
                draft_time_s=float(draft_time_s),
                verify_time_s=float(verify_time_s),
                cycle_e2e_s=float(cycle_e2e_s),
                num_active_reqs=int(num_active_reqs),
            )

            reward_norm_max = float(
                max(
                    getattr(req, "dflash_adaptive_thompson_reward_norm_max", 1.0),
                    float(reward),
                    1e-6,
                )
            )
            req.dflash_adaptive_thompson_reward_norm_max = reward_norm_max
            reward_norm = float(float(reward) / reward_norm_max)
            reward_norm = float(min(max(reward_norm, 0.0), 1.0))

            alpha_map[pulled_arm] = float(alpha_map.get(pulled_arm, 1.0)) + float(
                reward_norm
            )
            beta_map[pulled_arm] = float(beta_map.get(pulled_arm, 1.0)) + float(
                1.0 - reward_norm
            )
            counts[pulled_arm] = int(counts[pulled_arm]) + 1
            rounds = int(getattr(req, "dflash_adaptive_thompson_rounds", 0) or 0) + 1
            req.dflash_adaptive_thompson_rounds = int(rounds)

            unseen_arms = [int(a) for a in arms if int(counts.get(int(a), 0)) == 0]
            samples: dict[int, float] = {}
            if unseen_arms:
                next_bs = int(unseen_arms[0])
                reason = "thompson_warmup"
            else:
                for arm in arms:
                    a = float(max(alpha_map.get(arm, 1.0), 1e-6))
                    b = float(max(beta_map.get(arm, 1.0), 1e-6))
                    samples[arm] = float(random.betavariate(a, b))
                next_bs = int(max(arms, key=lambda a: (samples[a], -int(a))))
                reason = "thompson_sample"

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
            req.dflash_adaptive_thompson_last_scores = (
                {int(a): float(samples[a]) for a in samples} if samples else None
            )
            req.dflash_adaptive_last_decision = {
                "algo": "thompson",
                "reward_mode": self._adaptive_reward_mode,
                "reward_source": reward_source,
                "reward": float(reward_norm),
                "reward_raw": float(reward_raw),
                "thompson_round": int(rounds),
                "thompson_pulled_arm": int(pulled_arm),
                "thompson_pulled_arm_count": int(counts.get(pulled_arm, 0)),
                "thompson_pulled_arm_alpha": float(alpha_map.get(pulled_arm, 1.0)),
                "thompson_pulled_arm_beta": float(beta_map.get(pulled_arm, 1.0)),
                "thompson_reward_norm_max": float(reward_norm_max),
                "thompson_selected_sample": (
                    float(samples[next_bs]) if next_bs in samples else None
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

    def _record_runtime_verify_token_usage(
        self, batch: ScheduleBatch, verify_token_num: int
    ) -> None:
        """Accumulate per-request verify-token usage after confidence gating."""
        n = int(verify_token_num)
        for req in batch.reqs:
            hist = getattr(req, "dflash_runtime_verify_token_hist", None)
            if not isinstance(hist, dict):
                hist = {}
                req.dflash_runtime_verify_token_hist = hist
            hist[n] = int(hist.get(n, 0)) + 1

    @staticmethod
    def _record_runtime_verify_token_usage_by_req(
        reqs: list, verify_token_num_by_req: list[int]
    ) -> None:
        """Accumulate per-request verify-token usage when grouped verify is enabled."""
        n = min(len(reqs), len(verify_token_num_by_req))
        for i in range(n):
            req = reqs[i]
            k = int(verify_token_num_by_req[i])
            hist = getattr(req, "dflash_runtime_verify_token_hist", None)
            if not isinstance(hist, dict):
                hist = {}
                req.dflash_runtime_verify_token_hist = hist
            hist[k] = int(hist.get(k, 0)) + 1

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
        cycle_e2e_s: float,
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
        per_req_cycle_e2e_s = (
            float(cycle_e2e_s) / float(len(batch.reqs))
            if cycle_e2e_s > 0.0 and len(batch.reqs) > 0
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
            confidence_gate_decision = getattr(
                req, "dflash_confidence_gate_last_decision", None
            )
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
                    "cycle_e2e_batch_s": (
                        float(cycle_e2e_s) if cycle_e2e_s > 0.0 else None
                    ),
                    "cycle_e2e_s": per_req_cycle_e2e_s,
                    "adaptive_decision": adaptive_decision,
                    "confidence_gate_decision": confidence_gate_decision,
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
            req.dflash_adaptive_ucb_gain_sums = {}
            req.dflash_adaptive_ucb_cost_sums = {}
            req.dflash_adaptive_ucb_rounds = 0
            req.dflash_adaptive_ucb_reward_norm_max = 1.0
            req.dflash_adaptive_ucb_last_scores = None
            req.dflash_adaptive_linucb_counts = {}
            req.dflash_adaptive_linucb_rounds = 0
            req.dflash_adaptive_linucb_A_inv = {}
            req.dflash_adaptive_linucb_b = {}
            req.dflash_adaptive_linucb_last_scores = None
            req.dflash_adaptive_thompson_counts = {}
            req.dflash_adaptive_thompson_alpha = {}
            req.dflash_adaptive_thompson_beta = {}
            req.dflash_adaptive_thompson_rounds = 0
            req.dflash_adaptive_thompson_reward_norm_max = 1.0
            req.dflash_adaptive_thompson_last_scores = None
            req.dflash_runtime_bs_hist = {}
            req.dflash_runtime_verify_token_hist = {}
            req.dflash_confidence_gate_last_decision = None
        if hasattr(req, "spec_cycle_trace"):
            req.spec_cycle_trace = None

    def _cache_predictor_cycle_features(
        self,
        *,
        draft_hidden: torch.Tensor,
        runtime_block_size: int,
        per_req_verify_tokens: Optional[torch.Tensor],
    ) -> None:
        if self._predictor_dataset_writer is None or int(runtime_block_size) <= 1:
            self._last_predictor_draft_hidden_3d = None
            self._last_predictor_verify_tokens_by_req = None
            return
        self._last_predictor_draft_hidden_3d = (
            draft_hidden[:, 1:int(runtime_block_size), :].detach().contiguous()
        )
        if per_req_verify_tokens is not None:
            self._last_predictor_verify_tokens_by_req = [
                int(v) for v in per_req_verify_tokens.detach().cpu().tolist()
            ]
        else:
            bs = int(draft_hidden.shape[0])
            self._last_predictor_verify_tokens_by_req = [
                int(runtime_block_size) for _ in range(bs)
            ]

    def _write_predictor_cycle_features(
        self,
        *,
        batch: ScheduleBatch,
        runtime_block_size: int,
        accept_length_per_req_cpu: list[int],
    ) -> None:
        if self._predictor_dataset_writer is None or int(runtime_block_size) <= 1:
            return
        draft_hidden = self._last_predictor_draft_hidden_3d
        draft_tokens = self._last_draft_tokens_2d
        if draft_hidden is None or draft_tokens is None:
            return
        proposed = int(runtime_block_size) - 1
        if proposed <= 0:
            return
        draft_token_ids = draft_tokens[:, 1:int(runtime_block_size)]
        verify_token_nums = self._last_predictor_verify_tokens_by_req
        if verify_token_nums is None or len(verify_token_nums) != len(batch.reqs):
            verify_token_nums = [int(runtime_block_size) for _ in batch.reqs]
        cycle_indices = [int(getattr(req, "spec_verify_ct", 0)) for req in batch.reqs]
        rids = [str(getattr(req, "rid", f"req_{i}")) for i, req in enumerate(batch.reqs)]
        self._predictor_dataset_writer.add_cycle_batch(
            rids=rids,
            cycle_indices=cycle_indices,
            runtime_block_size=int(runtime_block_size),
            verify_token_nums=verify_token_nums,
            accepted_draft_tokens=[int(v) for v in accept_length_per_req_cpu],
            draft_token_ids_2d=draft_token_ids,
            draft_hidden_3d=draft_hidden,
        )
        self._last_predictor_draft_hidden_3d = None
        self._last_predictor_verify_tokens_by_req = None

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
        draft_next = torch.empty(
            (bs, max(runtime_block_size - 1, 0)),
            dtype=torch.long,
            device=device,
        )
        need_draft_max_probs = bool(
            self._confidence_gate_enabled
            and (
                self._confidence_gate_mode == "threshold"
                or self._confidence_gate_score_metric == "neg_log_max_prob"
            )
        )
        need_draft_entropies = bool(
            self._confidence_gate_enabled
            and self._confidence_gate_mode == "score"
            and self._confidence_gate_score_metric == "entropy"
        )
        draft_next_max_probs: Optional[torch.Tensor] = None
        draft_next_entropies: Optional[torch.Tensor] = None
        if runtime_block_size > 1:
            next_ids, next_max_probs, next_entropies = self._greedy_sample_from_vocab_parallel_head(
                hidden_states=draft_hidden[:, 1:, :].reshape(
                    -1, draft_hidden.shape[-1]
                ),
                lm_head=lm_head,
                return_max_probs=need_draft_max_probs,
                return_entropies=need_draft_entropies,
            )
            draft_next = next_ids.view(bs, runtime_block_size - 1)
            if next_max_probs is not None:
                draft_next_max_probs = next_max_probs.view(bs, runtime_block_size - 1)
            if next_entropies is not None:
                draft_next_entropies = next_entropies.view(
                    bs, runtime_block_size - 1
                )
        draft_tokens = self._draft_block_tokens_buf[:bs, :runtime_block_size]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        if runtime_block_size > 1:
            draft_tokens[:, 1:].copy_(draft_next)
        verify_token_num = int(runtime_block_size)
        confidence_gate_decision = None
        per_req_verify_tokens = torch.full(
            (bs,),
            int(verify_token_num),
            dtype=torch.int32,
            device=device,
        )
        if self._confidence_gate_enabled and runtime_block_size > 1:
            verify_token_num, confidence_gate_decision, per_req_verify_tokens = (
                self._compute_confidence_gated_verify_tokens(
                    draft_max_probs=draft_next_max_probs,
                    draft_entropies=draft_next_entropies,
                    runtime_block_size=int(runtime_block_size),
                )
            )
            if self._confidence_gate_grouped_verify_enabled:
                confidence_gate_decision["grouped_verify_enabled"] = True
                confidence_gate_decision["grouped_verify_buckets"] = [
                    int(v) for v in self._confidence_gate_grouped_verify_buckets
                ]
        self._last_verify_token_num = int(verify_token_num)
        if (
            self._confidence_gate_enabled
            and self._confidence_gate_grouped_verify_enabled
            and runtime_block_size > 1
            and not hasattr(
                self.target_worker.model_runner.attn_backend,
                "update_mamba_state_after_mtp_verify",
            )
        ):
            self._record_runtime_verify_token_usage_by_req(
                batch.reqs,
                [int(v) for v in per_req_verify_tokens.tolist()],
            )
        else:
            self._record_runtime_verify_token_usage(batch, verify_token_num)
        self._last_confidence_gate_decision = confidence_gate_decision
        self._last_per_req_verify_tokens = per_req_verify_tokens
        self._last_draft_tokens_2d = draft_tokens[:, :runtime_block_size]
        self._last_verify_positions_2d = positions_2d[:, :runtime_block_size]
        self._cache_predictor_cycle_features(
            draft_hidden=draft_hidden,
            runtime_block_size=int(runtime_block_size),
            per_req_verify_tokens=per_req_verify_tokens,
        )
        self._last_grouped_verify_active = False
        self._last_grouped_verify_plan = []
        if (
            self._confidence_gate_enabled
            and self._confidence_gate_grouped_verify_enabled
            and runtime_block_size > 1
        ):
            self._last_grouped_verify_plan = self._build_grouped_verify_plan(
                per_req_verify_tokens=per_req_verify_tokens,
                runtime_block_size=int(runtime_block_size),
            )
            self._last_grouped_verify_active = len(self._last_grouped_verify_plan) > 1
            if isinstance(confidence_gate_decision, dict):
                confidence_gate_decision["grouped_verify_candidate"] = bool(
                    self._last_grouped_verify_active
                )
                confidence_gate_decision["grouped_verify_plan"] = [
                    {
                        "verify_token_num": int(k),
                        "req_count": int(len(idxs)),
                    }
                    for k, idxs in self._last_grouped_verify_plan
                ]
            if self._last_grouped_verify_active:
                batch.spec_info = draft_input
                batch.return_hidden_states = False
                return

        verify_tokens_2d = draft_tokens[:, :verify_token_num]
        verify_positions_2d = positions_2d[:, :verify_token_num]
        positions = verify_positions_2d.reshape(-1).contiguous()

        verify_input = DFlashVerifyInput(
            draft_token=verify_tokens_2d.reshape(-1).contiguous(),
            positions=positions,
            draft_token_num=verify_token_num,
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
        return_max_probs: bool = False,
        return_entropies: bool = False,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            empty_ids = torch.empty(
                (0,), dtype=torch.long, device=hidden_states.device
            )
            empty_probs = (
                torch.empty((0,), dtype=torch.float32, device=hidden_states.device)
                if return_max_probs
                else None
            )
            empty_entropies = (
                torch.empty((0,), dtype=torch.float32, device=hidden_states.device)
                if return_entropies
                else None
            )
            return empty_ids, empty_probs, empty_entropies

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
        out_max_probs = (
            torch.empty((num_tokens,), dtype=torch.float32, device=hidden_states.device)
            if return_max_probs
            else None
        )
        out_entropies = (
            torch.empty((num_tokens,), dtype=torch.float32, device=hidden_states.device)
            if return_entropies
            else None
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        if tp_size == 1 and num_added == 0 and not return_max_probs and not return_entropies:
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
            return out_token_ids, None, None

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
                base_logits = None

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
            else:
                added_logits = None

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

            gathered_max_view = None
            if return_max_probs or return_entropies:
                if tp_size == 1:
                    global_max = local_max.to(torch.float32)
                else:
                    # Gather per-rank maxima to build numerically stable global log-sum-exp.
                    needed_for_max = tp_size * chunk_len
                    chunk_cap_for_max = int(chunk_size)
                    if (
                        self._draft_greedy_gather_cap < needed_for_max
                        or self._draft_greedy_gathered_max_buf is None
                        or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                        or self._draft_greedy_gathered_max_buf.device != hs.device
                    ):
                        cap = tp_size * chunk_cap_for_max
                        self._draft_greedy_gathered_max_buf = torch.empty(
                            (cap,), dtype=local_max.dtype, device=hs.device
                        )
                        if (
                            self._draft_greedy_gathered_ids_buf is None
                            or self._draft_greedy_gathered_ids_buf.device != hs.device
                            or int(self._draft_greedy_gathered_ids_buf.numel()) < cap
                        ):
                            self._draft_greedy_gathered_ids_buf = torch.empty(
                                (cap,), dtype=torch.int64, device=hs.device
                            )
                        self._draft_greedy_gather_cap = cap
                    gathered_max_for_prob = self._draft_greedy_gathered_max_buf[
                        :needed_for_max
                    ]
                    tp_group.all_gather_into_tensor(
                        gathered_max_for_prob, local_max.contiguous()
                    )
                    gathered_max_view = gathered_max_for_prob.view(tp_size, chunk_len)
                    global_max = gathered_max_view.max(dim=0).values.to(torch.float32)

                local_sumexp = torch.zeros(
                    (chunk_len,), dtype=torch.float32, device=hs.device
                )
                local_weighted_logit_sum = (
                    torch.zeros((chunk_len,), dtype=torch.float32, device=hs.device)
                    if return_entropies
                    else None
                )
                if base_logits is not None:
                    base_logits_fp32 = base_logits.to(torch.float32)
                    base_exp = torch.exp(base_logits_fp32 - global_max.unsqueeze(1))
                    local_sumexp += base_exp.sum(dim=-1)
                    if local_weighted_logit_sum is not None:
                        local_weighted_logit_sum += (
                            base_exp * base_logits_fp32
                        ).sum(dim=-1)
                if added_logits is not None:
                    added_logits_fp32 = added_logits.to(torch.float32)
                    added_exp = torch.exp(added_logits_fp32 - global_max.unsqueeze(1))
                    local_sumexp += added_exp.sum(dim=-1)
                    if local_weighted_logit_sum is not None:
                        local_weighted_logit_sum += (
                            added_exp * added_logits_fp32
                        ).sum(dim=-1)
                if tp_size > 1:
                    local_sumexp = tp_group.all_reduce(local_sumexp)
                    if local_weighted_logit_sum is not None:
                        local_weighted_logit_sum = tp_group.all_reduce(
                            local_weighted_logit_sum
                        )
                max_probs = torch.where(
                    local_sumexp > 0,
                    1.0 / torch.clamp(local_sumexp, min=1e-12),
                    torch.zeros_like(local_sumexp),
                )
                if out_max_probs is not None:
                    out_max_probs[start:end] = max_probs
                if out_entropies is not None:
                    assert local_weighted_logit_sum is not None
                    safe_sumexp = torch.clamp(local_sumexp, min=1e-12)
                    log_z = global_max + torch.log(safe_sumexp)
                    entropies = torch.where(
                        local_sumexp > 0,
                        log_z - (local_weighted_logit_sum / safe_sumexp),
                        torch.zeros_like(local_sumexp),
                    )
                    out_entropies[start:end] = torch.clamp(entropies, min=0.0)

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

            if gathered_max_view is None:
                tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
                gathered_max = gathered_max.view(tp_size, chunk_len)
            else:
                gathered_max = gathered_max_view
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_token_ids[start:end].copy_(selected_ids.view(-1))

        return out_token_ids, out_max_probs, out_entropies

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

        cycle_start_t = time.perf_counter()
        self._prepare_for_speculative_decoding(batch, draft_input)
        if self._report_timing and self.tp_rank == 0:
            self._accumulate_req_shared_time(
                batch.reqs, "spec_draft_time_s", float(self._last_draft_time_s)
            )

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        runtime_bs = int(getattr(self, "_last_runtime_block_size", self.block_size))
        grouped_verify_plan = list(getattr(self, "_last_grouped_verify_plan", []))
        grouped_verify_enabled = bool(getattr(self, "_last_grouped_verify_active", False))
        if grouped_verify_enabled and need_mamba_verify_commit:
            if not self._warned_grouped_verify_mamba_fallback and self.tp_rank == 0:
                logger.warning(
                    "DFLASH grouped confidence verify is disabled for Mamba verify-commit path. Falling back to single verify batch."
                )
                self._warned_grouped_verify_mamba_fallback = True
            grouped_verify_enabled = False

        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        verify_time_s = 0.0
        can_run_cuda_graph = True
        logits_output = None
        grouped_verify_consumed = False

        if grouped_verify_enabled:
            assert self._last_draft_tokens_2d is not None
            assert self._last_verify_positions_2d is not None
            bs = batch.batch_size()
            device = self.model_runner.device
            new_verified_id = torch.empty((bs,), dtype=torch.int64, device=device)
            commit_lens = torch.zeros((bs,), dtype=torch.int32, device=device)
            accept_length_per_req_cpu = [0 for _ in range(bs)]
            draft_seq_lens_work = draft_input.draft_seq_lens.clone()

            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )

            for verify_k, keep_indices in grouped_verify_plan:
                keep_indices_device = torch.tensor(
                    keep_indices, dtype=torch.int64, device=device
                )
                sub_batch = self._build_verify_sub_batch(
                    batch=batch,
                    keep_indices=keep_indices,
                    keep_indices_device=keep_indices_device,
                )

                verify_tokens_2d = self._last_draft_tokens_2d[
                    keep_indices_device, : int(verify_k)
                ]
                verify_positions_2d = self._last_verify_positions_2d[
                    keep_indices_device, : int(verify_k)
                ]
                verify_input = DFlashVerifyInput(
                    draft_token=verify_tokens_2d.reshape(-1).contiguous(),
                    positions=verify_positions_2d.reshape(-1).contiguous(),
                    draft_token_num=int(verify_k),
                )
                verify_input.prepare_for_verify(
                    sub_batch,
                    self.page_size,
                    build_custom_mask=build_custom_mask,
                )
                sub_batch.spec_info = verify_input
                sub_batch.return_hidden_states = False

                sub_model_worker_batch = sub_batch.get_model_worker_batch()
                assert sub_model_worker_batch.forward_mode.is_target_verify()
                sub_batch_result, sub_verify_time_s = self._measure_forward_s(
                    lambda: self.target_worker.forward_batch_generation(
                        sub_model_worker_batch, is_verify=True, **kwargs
                    )
                )
                verify_time_s += float(sub_verify_time_s)
                if self._report_timing and self.tp_rank == 0:
                    self._accumulate_req_shared_time(
                        sub_batch.reqs, "spec_verify_time_s", float(sub_verify_time_s)
                    )

                logits_output = sub_batch_result.logits_output
                can_run_cuda_graph = bool(
                    can_run_cuda_graph and sub_batch_result.can_run_cuda_graph
                )

                (
                    sub_new_verified_id,
                    sub_commit_lens,
                    sub_next_target_hidden,
                    sub_accept_length_per_req_cpu,
                ) = verify_input.verify(
                    batch=sub_batch,
                    logits_output=logits_output,
                    page_size=self.page_size,
                )

                # Sync per-group batch lens updates back to the original running batch.
                batch.seq_lens[keep_indices_device] = sub_batch.seq_lens
                if isinstance(batch.seq_lens_cpu, torch.Tensor):
                    batch.seq_lens_cpu[keep_indices] = sub_batch.seq_lens_cpu
                else:
                    for local_i, global_i in enumerate(keep_indices):
                        batch.seq_lens_cpu[global_i] = int(sub_batch.seq_lens_cpu[local_i])

                new_verified_id[keep_indices_device] = sub_new_verified_id
                commit_lens[keep_indices_device] = sub_commit_lens
                for local_i, global_i in enumerate(keep_indices):
                    accept_length_per_req_cpu[global_i] = int(
                        sub_accept_length_per_req_cpu[local_i]
                    )

                # Materialize committed tokens into draft KV for this group immediately.
                group_draft_input = DFlashDraftInput(
                    verified_id=sub_new_verified_id,
                    target_hidden=sub_next_target_hidden,
                    ctx_lens=sub_commit_lens,
                    draft_seq_lens=draft_seq_lens_work[keep_indices_device],
                )
                self._append_target_hidden_to_draft_kv(sub_batch, group_draft_input)
                draft_seq_lens_work[keep_indices_device] = group_draft_input.draft_seq_lens

            batch.seq_lens_sum = int(batch.seq_lens.sum().item())
            draft_input.verified_id = new_verified_id
            draft_input.target_hidden = draft_input.target_hidden[:0]
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.draft_seq_lens = draft_seq_lens_work
            grouped_verify_consumed = True

            if isinstance(self._last_confidence_gate_decision, dict):
                self._last_confidence_gate_decision["grouped_verify_applied"] = True
                self._last_confidence_gate_decision["grouped_verify_plan"] = [
                    {
                        "verify_token_num": int(k),
                        "req_count": int(len(idxs)),
                    }
                    for k, idxs in grouped_verify_plan
                ]
        else:
            model_worker_batch = batch.get_model_worker_batch()
            assert model_worker_batch.forward_mode.is_target_verify()
            verify_input = model_worker_batch.spec_info
            assert isinstance(verify_input, DFlashVerifyInput)

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
        if logits_output is None:
            raise RuntimeError(
                "DFLASH verify produced no logits output; this should never happen."
            )
        cycle_e2e_s = max(time.perf_counter() - cycle_start_t, 0.0)
        if self.tp_rank == 0:
            self._accumulate_req_shared_time(
                batch.reqs, "spec_cycle_e2e_s", float(cycle_e2e_s)
            )
        verify_token_num = int(getattr(self, "_last_verify_token_num", runtime_bs))
        conf_decision = self._last_confidence_gate_decision
        if self._confidence_gate_enabled:
            if not isinstance(conf_decision, dict):
                conf_decision = {
                    "enabled": True,
                    "selection_reason": "missing_prepare_decision",
                    "runtime_block_size": int(runtime_bs),
                    "verify_token_num": int(verify_token_num),
                }
            self._update_confidence_gate_state(
                decision=conf_decision,
                accept_length_per_req_cpu=accept_length_per_req_cpu,
                cycle_e2e_s=float(cycle_e2e_s),
            )
            for req in batch.reqs:
                req.dflash_confidence_gate_last_decision = self._last_confidence_gate_decision
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
                    cycle_e2e_s=float(cycle_e2e_s),
                    num_active_reqs=len(batch.reqs),
                )
        self._record_cycle_trace(
            batch=batch,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            runtime_block_size=runtime_bs,
            verify_time_s=float(verify_time_s),
            cycle_e2e_s=float(cycle_e2e_s),
        )
        self._write_predictor_cycle_features(
            batch=batch,
            runtime_block_size=int(runtime_bs),
            accept_length_per_req_cpu=accept_length_per_req_cpu,
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
        if not grouped_verify_consumed:
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
