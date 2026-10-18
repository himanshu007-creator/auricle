"""Publishes Auricle v1 to Hugging Face as auricle-voice/auricle-v1: the
weights, the layout, the ONNX export, the charts and the model card with the
YAML front matter the Hub needs. Idempotent: re-running uploads the changed
files only.

    HF_TOKEN=hf_... ../VAD-train/.venv/bin/python tools/hf_release.py [--dry-run]

Needs `huggingface_hub` (pip install huggingface_hub) and a token with write
access to the auricle-voice organisation.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT.parent / "VAD-train"
REPO = "auricle-voice/auricle-v1"

FRONT_MATTER = """---
license: mit
language:
  - en
  - es
  - de
  - ar
  - hi
  - zh
  - ja
tags:
  - voice-activity-detection
  - vad
  - speech
  - audio
  - voice-agent
  - turn-taking
  - barge-in
  - rust
  - webassembly
  - onnx
library_name: auricle
pipeline_tag: voice-activity-detection
datasets:
  - openslr/librispeech_asr
  - google/fleurs
metrics:
  - f1
model-index:
  - name: Auricle v1
    results:
      - task:
          type: voice-activity-detection
          name: Voice activity detection
        dataset:
          name: TEN VAD test set (30 human-labelled recordings)
          type: ten-vad-testset
        metrics:
          - type: f1
            value: 0.927
            name: Frame F1 at threshold 0.5
      - task:
          type: voice-activity-detection
          name: Foreground speech with a competing talker
        dataset:
          name: Held-out LibriSpeech speakers, second talker 3 to 15 dB quieter, real rooms (400 clips)
          type: auricle-competing-talker
        metrics:
          - type: f1
            value: 0.942
            name: Foreground F1 at threshold 0.5
          - type: false_alarm_rate
            value: 0.351
            name: Background talker frames called speech (Silero VAD v6: 0.813)
---

"""


def main():
    dry = "--dry-run" in sys.argv
    stage = Path(tempfile.mkdtemp(prefix="auricle-hf-"))
    files = {
        "auricle_v1_16k.f32": ROOT / "crates/auricle-vad/model/auricle_v1_16k.f32",
        "auricle_v1_layout.json": ROOT / "crates/auricle-vad/model/auricle_v1_layout.json",
        "auricle_v1.onnx": TRAIN / "runs/v1/export/vad.onnx",
        "eval.json": TRAIN / "runs/v1/eval.json",
        "langs.json": TRAIN / "runs/v1/langs.json",
        "pr.json": ROOT / "bench/pr-v1.json",
        "charts/pr-competing.png": ROOT / "demo/public/charts/pr-competing.png",
        "charts/pr-ten.png": ROOT / "demo/public/charts/pr-ten.png",
        "charts/languages.png": ROOT / "demo/public/charts/languages.png",
        "LICENSE": ROOT / "LICENSE",
    }
    for dst, src in files.items():
        if not src.exists():
            sys.exit(f"missing {src}")
        (stage / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, stage / dst)
    card = (ROOT / "docs/MODEL_CARD.md").read_text()
    card = card.replace("https://raw.githubusercontent.com/himanshu007-creator/auricle/main/demo/public/brand/logo.svg", "https://himanshu007-creator.github.io/auricle/brand/logo.svg")
    card = card.replace("demo/public/charts/", "charts/")
    (stage / "README.md").write_text(FRONT_MATTER + card)
    print("staged", sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()))
    if dry:
        print("dry run, nothing uploaded;", stage)
        return
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=REPO, repo_type="model", commit_message="Auricle v1")
    print("uploaded to https://huggingface.co/" + REPO)
    shutil.rmtree(stage)


if __name__ == "__main__":
    main()
