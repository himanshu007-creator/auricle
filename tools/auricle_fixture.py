"""Copies a trained Auricle Auricle checkpoint's export into the Rust crate and writes
the parity fixture the Rust test checks against: the PyTorch probabilities on
crates/auricle-vad/model/parity_input.f32 (the same 51,200-sample clip the Silero
parity test uses), one per 16 ms hop.

    VAD-train/.venv/bin/python tools/auricle_fixture.py ../VAD-train/runs/v1/best.pt

Run from the Auricle repository root with the VAD-train virtualenv so the
vadtrain package and torch are importable.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[2]))  # the VAD-train checkout
from vadtrain import export  # noqa: E402
from vadtrain.config import ModelConfig  # noqa: E402
from vadtrain.features import log_mel  # noqa: E402
from vadtrain.model import StreamingVad  # noqa: E402

ckpt_path = Path(sys.argv[1])
model_dir = Path(__file__).resolve().parents[1] / "crates" / "auricle-vad" / "model"
export_dir = ckpt_path.with_name("export")
export.main(ckpt_path, export_dir)
shutil.copy(export_dir / "vad.f32", model_dir / "auricle_v1_16k.f32")
shutil.copy(export_dir / "layout.json", model_dir / "auricle_v1_layout.json")

ckpt = torch.load(ckpt_path, map_location="cpu")
model = StreamingVad(ModelConfig(**ckpt["config"]["model"]))
model.load_state_dict(ckpt["ema"])
model.eval()
clip = np.fromfile(model_dir / "parity_input.f32", dtype="<f4")
with torch.no_grad():
    probs = torch.sigmoid(model(log_mel(torch.from_numpy(clip)[None]))[0])[0].numpy()
probs.astype("<f4").tofile(model_dir / "parity_auricle_expected.f32")
print(f"{len(probs)} hops, probabilities {probs.min():.3f} to {probs.max():.3f}; wrote {model_dir}")
