"""Engineering contracts plus short REAL-data CUDA trials, no final test access."""
from __future__ import annotations

import copy
import types

import numpy as np
import torch
from rtdl_revisiting_models import MultiheadAttention

from .data import load_data, prepare
from .search import defaults
from .train import build, cuda, loss_value, predict, run_trial, sdpa_forward
from .utils import HERE, now, write


def main():
    cuda(42)
    torch.manual_seed(42)
    attention = MultiheadAttention(d_embedding=32, n_heads=4, dropout=0.0).cuda().eval()
    accelerated = copy.deepcopy(attention)
    accelerated.forward = types.MethodType(sdpa_forward, accelerated)
    x = torch.randn(8, 12, 32, device="cuda")
    with torch.no_grad():
        a, b = attention(x, x), accelerated(x, x)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    contracts = {"sdpa_eval_equivalence_fp32": "PASS"}
    for loss in ["bce", "focal", "pairwise"]:
        z = torch.tensor([-.5, .8, 2., -1.], device="cuda", requires_grad=True)
        y = torch.tensor([0., 1., 0., 1.], device="cuda")
        value = loss_value(z, y, defaults() | {"loss": loss})
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(z.grad).all()
        contracts[f"{loss}_gradient"] = "PASS"
    prepare()
    records = []
    for i, (features, embedding, compression) in enumerate([
            ("original52", "linear", None), ("rich_pruned", "ple", None),
            ("rich_all", "periodic", .25)]):
        arrays, meta = load_data(features)
        arrays = {"train": {k: v[:8192] for k, v in arrays["train"].items()},
                  "validation": {k: v[:2048] for k, v in arrays["validation"].items()}}
        for name in ["train", "validation"]:
            assert set(np.unique(arrays[name]["y"])) == {0, 1}
        cfg = defaults() | {"features": features, "embedding": embedding, "compression": compression,
                            "width": 64, "depth": 2, "batch_size": 128 if features == "rich_all" else 256,
                            "max_epochs": 2, "min_epochs": 2, "patience": 2, "max_seconds": 600}
        record = run_trial(f"smoke_{i:03d}_{features}_{embedding}", cfg, arrays, meta)
        checkpoint = torch.load(HERE / "runs" / record["trial_id"] / "best.pt", map_location="cpu", weights_only=False)
        restored = build(meta, cfg, checkpoint["bins"]).cuda()
        restored.load_state_dict(checkpoint["model"])
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        p = predict(restored, torch.as_tensor(arrays["validation"]["x_num"], device="cuda"),
                    torch.as_tensor(arrays["validation"]["x_cat"], device="cuda"), cfg["batch_size"], dtype)
        with np.load(HERE / "runs" / record["trial_id"] / "predictions_validation.npz", allow_pickle=False) as f:
            np.testing.assert_allclose(p, f["p"], atol=1e-6, rtol=1e-6)
        contracts[f"{embedding}_checkpoint_prediction_replay"] = "PASS"
        records.append(record)
    write(HERE / "smoke_validation.json", {"status": "PASS_REAL_DATA_SMOKE", "finished_utc": now(),
                                           "contracts": contracts, "trials": records,
                                           "test_labels_accessed": False,
                                           "not_a_full_performance_benchmark": True})
    print("REAL DATA / CUDA / SERIALIZATION / GRADIENT / SDPA SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
