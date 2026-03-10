from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

_DFLASH_VERIFY_MASK_TEMPLATE_CACHE: Dict[
    Tuple[str, int, int, int], torch.Tensor
] = {}
_DFLASH_COMPACT_PATH_INDEX_CACHE: Dict[
    Tuple[str, int, int, int], torch.Tensor
] = {}
_DFLASH_VERIFY_MASK_BUILD_CACHE: Dict[
    Tuple[str, int], Dict[str, torch.Tensor]
] = {}


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    draft_token_num: int,
    page_size: int,
) -> torch.Tensor:
    """Compute how many draft slots per request must remain allocated.

    The allocator frees at page granularity for paged mode, so we can only release
    full pages from the tail after verify.
    """

    if page_size <= 1:
        raise ValueError(f"Expected page_size > 1, got {page_size}.")

    seq_dtype = prefix_lens.dtype
    extended_lens = prefix_lens + int(draft_token_num)
    new_lens = prefix_lens + commit_lens.to(seq_dtype)
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)
    keep_slots.clamp_(min=0, max=int(draft_token_num))
    return keep_slots


def _get_dflash_verify_suffix_template(
    *,
    num_candidates: int,
    candidate_block_size: int,
    shared_prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a cached per-request [q_len, q_len] branch-isolation mask template."""
    num_candidates_i = int(max(1, num_candidates))
    block_len = int(candidate_block_size)
    shared_len = int(max(0, min(shared_prefix_len, block_len)))
    cache_key = (str(device), num_candidates_i, block_len, shared_len)
    cached = _DFLASH_VERIFY_MASK_TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    compact_tree = bool(num_candidates_i > 1 and 0 < shared_len < block_len)
    if compact_tree:
        suffix_len = block_len - shared_len
        q_len = shared_len + num_candidates_i * suffix_len
        template = torch.zeros((q_len, q_len), dtype=torch.bool, device=device)
        if shared_len > 0:
            template[:shared_len, :shared_len] = torch.tril(
                torch.ones((shared_len, shared_len), dtype=torch.bool, device=device)
            )
            template[shared_len:, :shared_len] = True
        suffix_tril = torch.tril(
            torch.ones((suffix_len, suffix_len), dtype=torch.bool, device=device)
        )
        for cand_idx in range(num_candidates_i):
            base = shared_len + cand_idx * suffix_len
            template[base : base + suffix_len, base : base + suffix_len] = suffix_tril
    elif num_candidates_i <= 1:
        q_len = block_len
        q_idx = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(1)
        k_idx = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(0)
        template = k_idx <= q_idx
    else:
        q_len = block_len * num_candidates_i
        template = torch.zeros((q_len, q_len), dtype=torch.bool, device=device)
        block_tril = torch.tril(
            torch.ones((block_len, block_len), dtype=torch.bool, device=device)
        )
        for cand_idx in range(num_candidates_i):
            base = cand_idx * block_len
            template[base : base + block_len, base : base + block_len] = block_tril

    _DFLASH_VERIFY_MASK_TEMPLATE_CACHE[cache_key] = template
    return template


def _get_compact_candidate_path_indices(
    *,
    num_candidates: int,
    candidate_block_size: int,
    shared_prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return [num_candidates, block_size] token indices for compact-tree chosen paths."""
    num_candidates_i = int(max(1, num_candidates))
    block_len = int(candidate_block_size)
    shared_len = int(max(0, min(shared_prefix_len, block_len)))
    cache_key = (str(device), num_candidates_i, block_len, shared_len)
    cached = _DFLASH_COMPACT_PATH_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not (num_candidates_i > 1 and 0 < shared_len < block_len):
        path = torch.arange(
            block_len,
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0)
        _DFLASH_COMPACT_PATH_INDEX_CACHE[cache_key] = path
        return path

    suffix_len = block_len - shared_len
    shared_idx = torch.arange(shared_len, dtype=torch.int64, device=device)
    suffix_arange = torch.arange(suffix_len, dtype=torch.int64, device=device)
    rows: List[torch.Tensor] = []
    for cand_idx in range(num_candidates_i):
        suffix_idx = shared_len + cand_idx * suffix_len + suffix_arange
        rows.append(torch.cat([shared_idx, suffix_idx], dim=0))
    table = torch.stack(rows, dim=0)
    _DFLASH_COMPACT_PATH_INDEX_CACHE[cache_key] = table
    return table


def build_dflash_verify_allow_mask(
    *,
    prefix_len: int,
    draft_token_num: int,
    num_candidates: int,
    candidate_block_size: int,
    shared_prefix_len: int = 0,
    device: torch.device,
) -> torch.Tensor:
    """Build the per-request bool allow mask for DFLASH target verify.

    Semantics:
    - single candidate: standard causal verify over the drafted block
    - multi candidate: each query attends to the full shared prefix and only its
      own candidate branch up to the current local position
    """

    num_candidates_i = int(max(1, num_candidates))
    block_len = int(candidate_block_size)
    shared_len = int(max(0, min(shared_prefix_len, block_len)))
    template = _get_dflash_verify_suffix_template(
        num_candidates=num_candidates_i,
        candidate_block_size=block_len,
        shared_prefix_len=shared_len,
        device=device,
    )
    q_len = int(template.shape[0])
    prefix_len_i = int(prefix_len)
    kv_len = prefix_len_i + q_len
    allow = torch.zeros((q_len, kv_len), dtype=torch.bool, device=device)
    if prefix_len_i > 0:
        allow[:, :prefix_len_i] = True
    allow[:, prefix_len_i : prefix_len_i + q_len] = template
    return allow


def _get_or_create_dflash_verify_mask_build_buffers(
    *,
    device: torch.device,
    bs: int,
    q_len: int,
    max_prefix_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return reusable staging buffers for batched DFLASH verify-mask assembly."""

    width = int(max_prefix_len + q_len)
    cache_key = (str(device), int(q_len))
    cached = _DFLASH_VERIFY_MASK_BUILD_CACHE.get(cache_key)
    cap_bs = 0 if cached is None else int(cached["dense_mask"].shape[0])
    cap_width = 0 if cached is None else int(cached["dense_mask"].shape[2])

    if cap_bs < bs or cap_width < width:
        new_cap_bs = max(int(bs), cap_bs * 2 if cap_bs > 0 else int(bs))
        new_cap_width = max(int(width), cap_width * 2 if cap_width > 0 else int(width))
        cached = {
            "dense_mask": torch.empty(
                (new_cap_bs, q_len, new_cap_width),
                dtype=torch.bool,
                device=device,
            ),
            "all_cols": torch.arange(
                new_cap_width, dtype=torch.int64, device=device
            ),
            "q_cols": torch.arange(q_len, dtype=torch.int64, device=device),
        }
        _DFLASH_VERIFY_MASK_BUILD_CACHE[cache_key] = cached

    assert cached is not None
    return (
        cached["dense_mask"][:bs, :, :width],
        cached["all_cols"][:width],
        cached["q_cols"],
    )


def build_dflash_verify_allow_mask_batched(
    *,
    prefix_lens: torch.Tensor,
    draft_token_num: int,
    num_candidates: int,
    candidate_block_size: int,
    shared_prefix_len: int = 0,
    device: torch.device,
) -> torch.Tensor:
    """Build the flattened per-batch allow mask for DFLASH target verify on GPU."""

    if prefix_lens.numel() == 0:
        return torch.empty((0,), dtype=torch.bool, device=device)

    num_candidates_i = int(max(1, num_candidates))
    block_len = int(candidate_block_size)
    shared_len = int(max(0, min(shared_prefix_len, block_len)))
    template = _get_dflash_verify_suffix_template(
        num_candidates=num_candidates_i,
        candidate_block_size=block_len,
        shared_prefix_len=shared_len,
        device=device,
    )
    q_len = int(template.shape[0])
    if q_len <= 0 or int(draft_token_num) <= 0:
        return torch.empty((0,), dtype=torch.bool, device=device)

    prefix_lens_i64 = prefix_lens.to(device=device, dtype=torch.int64)
    bs = int(prefix_lens_i64.numel())
    max_prefix_len = int(prefix_lens_i64.max().item()) if bs > 0 else 0

    dense_mask, all_cols, q_cols = _get_or_create_dflash_verify_mask_build_buffers(
        device=device,
        bs=bs,
        q_len=q_len,
        max_prefix_len=max_prefix_len,
    )
    dense_mask.zero_()

    if max_prefix_len > 0:
        dense_mask[:, :, :max_prefix_len] = (
            all_cols[:max_prefix_len].view(1, 1, -1)
            < prefix_lens_i64.view(bs, 1, 1)
        )

    template_indices = prefix_lens_i64.view(bs, 1, 1) + q_cols.view(1, 1, q_len)
    dense_mask.scatter_(
        2,
        template_indices.expand(bs, q_len, q_len),
        template.unsqueeze(0).expand(bs, -1, -1),
    )

    valid_cols = all_cols.view(1, 1, -1) < (
        prefix_lens_i64 + int(q_len)
    ).view(bs, 1, 1)
    return dense_mask[valid_cols.expand(bs, q_len, valid_cols.shape[-1])].contiguous()


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    Invariant (per request):
      - `draft_seq_len + ctx_len == batch.seq_lens[i]`
        where `ctx_len` is the number of target context-feature tokens carried in
        `target_hidden` for that request.
    """

    # Current token to start the next DFlash block (one per request).
    verified_id: torch.Tensor

    # Flattened context features for tokens that need to be appended into the draft cache.
    # Shape: [sum(ctx_lens), K * hidden_size], where K is the number of target-layer
    # hidden-state features concatenated per token (len(dflash_config.target_layer_ids),
    # or default K == draft_num_layers for existing checkpoints).
    target_hidden: torch.Tensor

    # Context lengths per request, used to slice `target_hidden`. Device tensor (int32).
    ctx_lens: torch.Tensor

    # How many tokens are already in the draft KV cache per request.
    # The next draft step appends ctx_lens[i] tokens starting at draft_seq_lens[i].
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Draft state does not change token accounting.
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]

        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        # Rebuild target_hidden for the filtered batch using vectorized indexing.
        old_bs = int(old_ctx_lens.shape[0])
        offsets = torch.zeros(
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))

        start = offsets[:-1]
        seg_start = start[new_indices]
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)

        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[
            None, :
        ]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        flat_pos = pos2d[mask]
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1).

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1
    num_candidates: int = 1
    candidate_block_size: int | None = None
    # Number of shared deterministic tokens (from the start of each candidate block)
    # packed once when compact tree verify is enabled.
    shared_prefix_len: int = 0
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it (e.g. triton).
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
        if self.candidate_block_size is None:
            self.candidate_block_size = int(self.draft_token_num)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.tokens_per_req)

    @property
    def tokens_per_req(self) -> int:
        block_len = int(self.candidate_block_size)
        num_candidates = int(max(1, self.num_candidates))
        shared_len = int(max(0, min(self.shared_prefix_len, block_len)))
        if num_candidates > 1 and 0 < shared_len < block_len:
            return int(shared_len + num_candidates * (block_len - shared_len))
        return int(block_len * num_candidates)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.tokens_per_req, self.tokens_per_req

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + self.tokens_per_req
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.tokens_per_req
            end_offset_cpu = prefix_lens_cpu + self.tokens_per_req
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 0:
            raise ValueError(
                f"DFLASH draft_token_num must be positive, got {self.draft_token_num}."
            )
        self.custom_mask = build_dflash_verify_allow_mask_batched(
            prefix_lens=batch.seq_lens,
            draft_token_num=int(self.draft_token_num),
            num_candidates=int(self.num_candidates),
            candidate_block_size=int(self.candidate_block_size),
            shared_prefix_len=int(self.shared_prefix_len),
            device=batch.device,
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.tokens_per_req,
            step=self.tokens_per_req,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.tokens_per_req
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.tokens_per_req * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.tokens_per_req
                + (self.tokens_per_req**2) * bs
            )
            if mask.numel() < mask_numel:
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def _verify_multi_candidate(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if page_size != 1:
            raise RuntimeError(
                "DFLASH multi-candidate packed verify currently supports only page_size=1."
            )

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise RuntimeError(
                "DFLASH multi-candidate packed verify currently supports only greedy target verification."
            )

        num_candidates = int(self.num_candidates)
        block_len = int(self.candidate_block_size)
        shared_prefix_len = int(max(0, min(int(self.shared_prefix_len), block_len)))
        compact_tree = bool(num_candidates > 1 and 0 < shared_prefix_len < block_len)

        if compact_tree:
            suffix_len = block_len - shared_prefix_len
            compact_tokens_per_req = int(self.tokens_per_req)
            compact_tokens = self.draft_token.view(bs, compact_tokens_per_req)
            compact_target_predict = torch.argmax(
                logits_output.next_token_logits, dim=-1
            ).view(bs, compact_tokens_per_req)

            candidates = torch.empty(
                (bs, num_candidates, block_len), dtype=compact_tokens.dtype, device=device
            )
            target_predict = torch.empty(
                (bs, num_candidates, block_len),
                dtype=compact_target_predict.dtype,
                device=device,
            )
            if shared_prefix_len > 0:
                shared_tokens = compact_tokens[:, :shared_prefix_len]
                shared_target_predict = compact_target_predict[:, :shared_prefix_len]
                candidates[:, :, :shared_prefix_len] = shared_tokens.unsqueeze(1).expand(
                    -1, num_candidates, -1
                )
                target_predict[:, :, :shared_prefix_len] = shared_target_predict.unsqueeze(
                    1
                ).expand(-1, num_candidates, -1)

            suffix_tokens = compact_tokens[:, shared_prefix_len:].view(
                bs, num_candidates, suffix_len
            )
            suffix_target_predict = compact_target_predict[:, shared_prefix_len:].view(
                bs, num_candidates, suffix_len
            )
            candidates[:, :, shared_prefix_len:] = suffix_tokens
            target_predict[:, :, shared_prefix_len:] = suffix_target_predict
        else:
            candidates = self.draft_token.view(bs, num_candidates, block_len)
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, num_candidates, block_len
            )

        chosen_candidate_idx = None
        chosen_tau = None
        chosen_next_token = None
        if block_len <= 1:
            accept_len = torch.zeros(
                (bs, num_candidates), dtype=torch.int64, device=device
            )
        else:
            accept_len = (
                candidates[:, :, 1:].eq(target_predict[:, :, :-1]).cumprod(dim=2).sum(dim=2)
            )
        tau = accept_len + 1
        candidate_idx = torch.arange(
            num_candidates, dtype=torch.int64, device=device
        ).unsqueeze(0)
        score = tau.to(torch.int64) * int(num_candidates + 1) - candidate_idx
        chosen_candidate_idx = torch.argmax(score, dim=1)
        chosen_tau = tau.gather(1, chosen_candidate_idx.unsqueeze(1)).squeeze(1)
        next_tokens_all = target_predict.gather(
            2, accept_len.unsqueeze(-1)
        ).squeeze(-1)
        chosen_next_token = next_tokens_all.gather(
            1, chosen_candidate_idx.unsqueeze(1)
        ).squeeze(1)
        chosen_tokens = candidates[
            torch.arange(bs, dtype=torch.int64, device=device), chosen_candidate_idx
        ]
        # Single batched D2H transfer avoids repeated per-cycle GPU sync from .item().
        packed_choice = torch.cat(
            [
                chosen_tokens.to(torch.int64),
                chosen_next_token.unsqueeze(1).to(torch.int64),
                chosen_tau.unsqueeze(1).to(torch.int64),
            ],
            dim=1,
        ).cpu()
        chosen_candidate_idx_cpu = chosen_candidate_idx.to(torch.int64).cpu()

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        if compact_tree:
            compact_tokens_per_req = int(self.tokens_per_req)
            out_cache_loc_compact = batch.out_cache_loc.view(bs, compact_tokens_per_req)
            hidden_compact = hidden.view(bs, compact_tokens_per_req, -1)
            hidden_feature_dim = int(hidden_compact.shape[-1])
            candidate_path_indices = _get_compact_candidate_path_indices(
                num_candidates=num_candidates,
                candidate_block_size=block_len,
                shared_prefix_len=shared_prefix_len,
                device=device,
            )
        else:
            out_cache_loc = batch.out_cache_loc.view(bs, num_candidates, block_len)
            hidden = hidden.view(bs, num_candidates, block_len, -1)
            hidden_feature_dim = int(hidden.shape[-1])

        free_segments: List[torch.Tensor] = []
        kept_segments: List[torch.Tensor] = []
        target_hidden_segments: List[torch.Tensor] = []
        commit_lens_cpu: List[int] = []
        accept_length_per_req_cpu: List[int] = []
        new_verified_list: List[int] = []

        for i, req in enumerate(batch.reqs):
            chosen_idx = int(chosen_candidate_idx_cpu[i].item())
            acc_len = int(packed_choice[i, block_len + 1].item()) - 1
            chosen_tokens_cpu = packed_choice[i, :block_len]
            proposed = chosen_tokens_cpu[1 : 1 + acc_len].tolist() + [
                int(packed_choice[i, block_len].item())
            ]

            appended = 0
            if (
                req.grammar is None
                and not req.sampling_params.stop_strs
                and not req.sampling_params.stop_regex_strs
            ):
                remaining = int(req.sampling_params.max_new_tokens) - len(req.output_ids)
                if remaining > 0:
                    tokens = proposed[:remaining]
                    if not req.sampling_params.ignore_eos:
                        stop_token_ids = req.sampling_params.stop_token_ids
                        eos_token_ids = req.eos_token_ids
                        tokenizer = req.tokenizer
                        tokenizer_eos = (
                            tokenizer.eos_token_id if tokenizer is not None else None
                        )
                        additional_stop = (
                            tokenizer.additional_stop_token_ids
                            if tokenizer is not None
                            else None
                        )
                        vocab_size = getattr(req, "vocab_size", None)

                        for j, token_id in enumerate(tokens):
                            if vocab_size is not None and (
                                int(token_id) > int(vocab_size) or int(token_id) < 0
                            ):
                                tokens = tokens[: j + 1]
                                break
                            if stop_token_ids and token_id in stop_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if eos_token_ids and token_id in eos_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if tokenizer_eos is not None and int(token_id) == int(tokenizer_eos):
                                tokens = tokens[: j + 1]
                                break
                            if additional_stop and token_id in additional_stop:
                                tokens = tokens[: j + 1]
                                break

                    req.output_ids.extend(int(tok) for tok in tokens)
                    appended = len(tokens)
                    if appended > 0:
                        req.check_finished(new_accepted_len=appended)
            else:
                for tok in proposed:
                    req.output_ids.append(int(tok))
                    appended += 1
                    req.check_finished()
                    if req.finished():
                        break
                    if req.grammar is not None:
                        req.grammar.accept_token(int(tok))

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]
            req.spec_draft_token_num += max(0, int(block_len) - 1)

            if compact_tree:
                chosen_path_idx = candidate_path_indices[chosen_idx]
                keep = out_cache_loc_compact[i, chosen_path_idx[:appended]]
                kept_segments.append(keep)
                if appended > 0:
                    target_hidden_segments.append(
                        hidden_compact[i, chosen_path_idx[:appended], :]
                    )

                free_mask = torch.ones(
                    (out_cache_loc_compact.shape[1],), dtype=torch.bool, device=device
                )
                if appended > 0:
                    free_mask[chosen_path_idx[:appended]] = False
                free_req = out_cache_loc_compact[i, free_mask]
                if free_req.numel() > 0:
                    free_segments.append(free_req)
            else:
                keep = out_cache_loc[i, chosen_idx, :appended]
                kept_segments.append(keep)
                if appended > 0:
                    target_hidden_segments.append(hidden[i, chosen_idx, :appended, :])

                free_req = []
                if chosen_idx > 0:
                    free_req.append(out_cache_loc[i, :chosen_idx, :].reshape(-1))
                if chosen_idx + 1 < num_candidates:
                    free_req.append(out_cache_loc[i, chosen_idx + 1 :, :].reshape(-1))
                if appended < block_len:
                    free_req.append(out_cache_loc[i, chosen_idx, appended:block_len])
                if free_req:
                    free_segments.append(torch.cat(free_req, dim=0))

        if free_segments:
            batch.token_to_kv_pool_allocator.free(torch.cat(free_segments, dim=0))
        batch.out_cache_loc = (
            torch.cat(kept_segments, dim=0)
            if kept_segments
            else batch.out_cache_loc[:0]
        )

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        next_target_hidden = (
            torch.cat(target_hidden_segments, dim=0)
            if target_hidden_segments
            else torch.empty((0, hidden_feature_dim), dtype=hidden.dtype, device=device)
        )
        logits_output.hidden_states = None
        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        )

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """DFlash verification for greedy and non-greedy sampling.

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        if int(self.num_candidates) > 1:
            return self._verify_multi_candidate(
                batch=batch,
                logits_output=logits_output,
                page_size=page_size,
            )

        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )

            # Keep speculative verify semantics consistent with normal sampling path.
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )

            if (
                sampling_info.penalizer_orchestrator.is_required
                or sampling_info.logit_bias is not None
            ):
                linear_penalty = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
                )

        candidates = self.draft_token.view(bs, self.draft_token_num)
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
            )
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, self.draft_token_num
            )
            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )

        # Single D2H transfer: candidates[1:] + accept_len + bonus
        packed = torch.cat(
            [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()

        max_acc = self.draft_token_num - 1
        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []

        for i, req in enumerate(batch.reqs):
            acc_len = int(packed[i, max_acc].item())
            proposed = packed[i, :acc_len].tolist() + [
                int(packed[i, max_acc + 1].item())
            ]

            appended = 0
            if (
                req.grammar is None
                and not req.sampling_params.stop_strs
                and not req.sampling_params.stop_regex_strs
            ):
                remaining = int(req.sampling_params.max_new_tokens) - len(
                    req.output_ids
                )
                if remaining > 0:
                    tokens = proposed[:remaining]
                    if not req.sampling_params.ignore_eos:
                        stop_token_ids = req.sampling_params.stop_token_ids
                        eos_token_ids = req.eos_token_ids
                        tokenizer = req.tokenizer
                        tokenizer_eos = (
                            tokenizer.eos_token_id if tokenizer is not None else None
                        )
                        additional_stop = (
                            tokenizer.additional_stop_token_ids
                            if tokenizer is not None
                            else None
                        )
                        vocab_size = getattr(req, "vocab_size", None)

                        for j, token_id in enumerate(tokens):
                            if vocab_size is not None and (
                                int(token_id) > int(vocab_size) or int(token_id) < 0
                            ):
                                tokens = tokens[: j + 1]
                                break
                            if stop_token_ids and token_id in stop_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if eos_token_ids and token_id in eos_token_ids:
                                tokens = tokens[: j + 1]
                                break
                            if tokenizer_eos is not None and int(token_id) == int(
                                tokenizer_eos
                            ):
                                tokens = tokens[: j + 1]
                                break
                            if additional_stop and token_id in additional_stop:
                                tokens = tokens[: j + 1]
                                break

                    req.output_ids.extend(int(tok) for tok in tokens)
                    appended = len(tokens)
                    if appended > 0:
                        req.check_finished(new_accepted_len=appended)
            else:
                for tok in proposed:
                    req.output_ids.append(int(tok))
                    appended += 1
                    req.check_finished()
                    if req.finished():
                        break
                    if req.grammar is not None:
                        req.grammar.accept_token(int(tok))

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                # If no token was appended in this verify step, keep the current token unchanged.
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]
            req.spec_draft_token_num += max(0, int(self.draft_token_num) - 1)

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        # Free uncommitted KV cache slots and compact out_cache_loc.
        if page_size == 1:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            keep_mask = (
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
            batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            row_offsets = torch.arange(self.draft_token_num, device=device)[None, :]
            keep_slots = _compute_paged_keep_slots(
                prefix_lens=batch.seq_lens,
                commit_lens=commit_lens,
                draft_token_num=self.draft_token_num,
                page_size=page_size,
            )
            free_mask = row_offsets >= keep_slots[:, None]
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])

            keep_mask = row_offsets < commit_lens[:, None]
            batch.out_cache_loc = out_cache_loc[keep_mask]

        # Update req-level KV cache accounting.
        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        # Update req_to_token pool mapping for newly committed tokens.
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # Update batch seq lens.
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        # Keep seq_lens_sum in sync; flashinfer indices updaters rely on this for buffer sizing.
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, self.draft_token_num, -1)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                segments.append(hidden[i, :ln, :])
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]

        # Avoid confusing downstream consumers (spec-v1 decode doesn't use this).
        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
