import atexit
import json
import os
from pathlib import Path
from typing import Optional

import torch


class PredictorDatasetShardWriter:
    def __init__(
        self,
        *,
        output_dir: str,
        worker_tag: str,
        shard_max_rows: int = 100000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker_tag = str(worker_tag)
        self.shard_max_rows = max(int(shard_max_rows), 1)
        self.pid = int(os.getpid())
        self._shard_idx = 0
        self._buffered_rows = 0
        self._request_id_by_rid: dict[str, int] = {}
        self._rid_by_request_id: list[str] = []
        self._written_shards: list[str] = []
        self._buffers = self._empty_buffers()
        atexit.register(self.close)

    def _empty_buffers(self) -> dict[str, object]:
        return {
            "request_id": [],
            "cycle_idx": [],
            "draft_pos": [],
            "runtime_block_size": [],
            "verify_token_num": [],
            "proposed_draft_tokens": [],
            "accepted_draft_tokens": [],
            "token_accepted": [],
            "first_reject_here": [],
            "draft_token_id": [],
            "draft_hidden": [],
        }

    def _get_request_id(self, rid: str) -> int:
        rid = str(rid)
        existing = self._request_id_by_rid.get(rid)
        if existing is not None:
            return int(existing)
        request_id = len(self._rid_by_request_id)
        self._request_id_by_rid[rid] = request_id
        self._rid_by_request_id.append(rid)
        return int(request_id)

    def add_cycle_batch(
        self,
        *,
        rids: list[str],
        cycle_indices: list[int],
        runtime_block_size: int,
        verify_token_nums: list[int],
        accepted_draft_tokens: list[int],
        draft_token_ids_2d: torch.Tensor,
        draft_hidden_3d: torch.Tensor,
    ) -> None:
        if draft_token_ids_2d.numel() == 0 or draft_hidden_3d.numel() == 0:
            return

        token_ids_cpu = draft_token_ids_2d.detach().to(device="cpu", dtype=torch.int32)
        hidden_cpu = draft_hidden_3d.detach().to(device="cpu", dtype=torch.float16)
        if token_ids_cpu.dim() != 2 or hidden_cpu.dim() != 3:
            raise ValueError("Predictor dataset expects token_ids [B, K] and hidden [B, K, H].")

        bs = int(token_ids_cpu.shape[0])
        proposed = int(token_ids_cpu.shape[1])
        if bs == 0 or proposed == 0:
            return
        if hidden_cpu.shape[0] != bs or hidden_cpu.shape[1] != proposed:
            raise ValueError("Draft token and hidden feature shapes do not align.")

        for i in range(bs):
            req_id = self._get_request_id(str(rids[i]))
            accepted = max(0, int(accepted_draft_tokens[i]))
            cycle_idx = int(cycle_indices[i])
            verify_token_num = int(verify_token_nums[i])
            positions = torch.arange(1, proposed + 1, dtype=torch.int16)

            self._buffers["request_id"].append(
                torch.full((proposed,), req_id, dtype=torch.int32)
            )
            self._buffers["cycle_idx"].append(
                torch.full((proposed,), cycle_idx, dtype=torch.int16)
            )
            self._buffers["draft_pos"].append(positions)
            self._buffers["runtime_block_size"].append(
                torch.full((proposed,), int(runtime_block_size), dtype=torch.int16)
            )
            self._buffers["verify_token_num"].append(
                torch.full((proposed,), verify_token_num, dtype=torch.int16)
            )
            self._buffers["proposed_draft_tokens"].append(
                torch.full((proposed,), proposed, dtype=torch.int16)
            )
            self._buffers["accepted_draft_tokens"].append(
                torch.full((proposed,), accepted, dtype=torch.int16)
            )
            self._buffers["token_accepted"].append(
                (positions <= accepted).to(torch.uint8)
            )
            first_reject = torch.zeros((proposed,), dtype=torch.uint8)
            if accepted < proposed:
                first_reject[accepted] = 1
            self._buffers["first_reject_here"].append(first_reject)
            self._buffers["draft_token_id"].append(token_ids_cpu[i])
            self._buffers["draft_hidden"].append(hidden_cpu[i].contiguous())

        self._buffered_rows += bs * proposed
        if self._buffered_rows >= self.shard_max_rows:
            self.flush()

    def flush(self) -> Optional[Path]:
        if self._buffered_rows <= 0:
            return None

        shard_name = f"predictor_features_{self.worker_tag}_shard{self._shard_idx:05d}.pt"
        shard_path = self.output_dir / shard_name
        payload = {
            "meta": {
                "worker_tag": self.worker_tag,
                "pid": self.pid,
                "num_rows": int(self._buffered_rows),
                "num_requests_seen": int(len(self._rid_by_request_id)),
            },
            "request_id": torch.cat(self._buffers["request_id"], dim=0),
            "cycle_idx": torch.cat(self._buffers["cycle_idx"], dim=0),
            "draft_pos": torch.cat(self._buffers["draft_pos"], dim=0),
            "runtime_block_size": torch.cat(self._buffers["runtime_block_size"], dim=0),
            "verify_token_num": torch.cat(self._buffers["verify_token_num"], dim=0),
            "proposed_draft_tokens": torch.cat(
                self._buffers["proposed_draft_tokens"], dim=0
            ),
            "accepted_draft_tokens": torch.cat(
                self._buffers["accepted_draft_tokens"], dim=0
            ),
            "token_accepted": torch.cat(self._buffers["token_accepted"], dim=0),
            "first_reject_here": torch.cat(self._buffers["first_reject_here"], dim=0),
            "draft_token_id": torch.cat(self._buffers["draft_token_id"], dim=0),
            "draft_hidden": torch.cat(self._buffers["draft_hidden"], dim=0),
        }
        torch.save(payload, shard_path)
        self._written_shards.append(shard_name)
        self._shard_idx += 1
        self._buffered_rows = 0
        self._buffers = self._empty_buffers()
        self._write_index()
        return shard_path

    def _write_index(self) -> None:
        index_path = self.output_dir / f"predictor_features_{self.worker_tag}_index.json"
        payload = {
            "worker_tag": self.worker_tag,
            "pid": self.pid,
            "shard_max_rows": int(self.shard_max_rows),
            "shards": list(self._written_shards),
            "request_id_to_rid": list(self._rid_by_request_id),
        }
        index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        self.flush()
        self._write_index()
