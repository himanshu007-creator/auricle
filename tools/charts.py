"""Renders the benchmark charts for the site and README from the measured
numbers (bench/pr-v1.json, bench/langs-v1.json, docs/COMPARISON.md).

    ../VAD-train/.venv/bin/python tools/charts.py demo/public/charts
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/public/charts")
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent

LIGHT = os.environ.get("THEME") == "light"
if LIGHT:
    BG, CARD, TEXT, MUTED, LINE = "#fbfbfc", "#ffffff", "#0f1115", "#5d6472", "#e3e6ec"
    AURICLE, OTHER, THIRD = "#1f9a6b", "#2f7fd6", "#b8860b"
else:
    BG, CARD, TEXT, MUTED, LINE = "#0f1115", "#171a21", "#e6e8ee", "#8b91a1", "#2a2f3a"
    AURICLE, OTHER, THIRD = "#5ad1a1", "#5ab0ff", "#f4c95d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "grid.color": LINE, "axes.grid": True, "grid.alpha": 0.8,
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15, "legend.fontsize": 14,
    "legend.facecolor": CARD, "legend.edgecolor": LINE, "axes.titlepad": 14,
})


def save(fig, name, note=None):
    if note:
        import textwrap
        fig.text(0.01, 0.012, "\n".join(textwrap.wrap(note, 120)), color=MUTED, fontsize=12, va="bottom")
        fig.tight_layout(rect=(0, 0.06 + 0.03 * note.count(" ") // 20, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", OUT / name)


def bars(title, labels, values, colors, unit, name, note=None, fmt="{:,.0f}"):
    """Horizontal bars, biggest value at the bottom so the winner is on top."""
    fig, ax = plt.subplots(figsize=(11, 1.1 * len(labels) + 1.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlabel(unit)
    ax.set_title(title, loc="left")
    ax.set_xlim(0, max(values) * 1.22)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.015, yi, fmt.format(v), va="center", color=TEXT, fontsize=14, fontweight="bold")
    save(fig, name, note)


# 1 and 2: the precision-recall sweeps, one big panel each
pr = json.loads((ROOT / "bench/pr-v1.json").read_text())["curves"]
names = {"student": "Auricle v1", "teacher": "Silero v6"}
colors = {"student": AURICLE, "teacher": OTHER}

fig, ax = plt.subplots(figsize=(11, 7))
for k, c in pr["ten"].items():
    ax.plot(c["recall"], c["precision"], color=colors[k], lw=3.5, label=names[k])
ax.set_xlim(0.45, 1.0); ax.set_ylim(0.5, 1.0)
ax.set_xlabel("Recall (share of real speech frames caught)")
ax.set_ylabel("Precision (share of flagged frames that were speech)")
ax.set_title("One person talking: both models are close", loc="left")
ax.legend(loc="lower left")
save(fig, "pr-ten.png", "TEN VAD test set, 30 human-labelled recordings, 16,352 frames of 16 ms, every threshold. Higher and further right is better.")

fig, ax = plt.subplots(figsize=(11, 7))
for k, c in pr["competing"].items():
    ax.plot(c["recall"], c["false_alarm"], color=colors[k], lw=3.5, label=names[k])
ax.set_xlim(0.5, 1.0); ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Share of the user's speech caught")
ax.set_ylabel("Share of the other person's speech wrongly flagged")
ax.set_title("Two people talking: Auricle ignores the one in the background", loc="left")
ax.legend(loc="upper left")
ax.annotate("Silero at its default threshold:\n81% of the other person's frames\nare called speech", xy=(0.976, 0.81), xytext=(0.60, 0.86), color=OTHER, fontsize=13, arrowprops=dict(arrowstyle="->", color=OTHER))
ax.annotate("Auricle at its default threshold: 35%", xy=(0.975, 0.35), xytext=(0.64, 0.50), color=AURICLE, fontsize=13, arrowprops=dict(arrowstyle="->", color=AURICLE))
save(fig, "pr-competing.png", "The user at 0 dB, a second speaker 3 to 15 dB quieter, both through a real room response, 400 clips from held-out speakers, every threshold. Lower is better.")

# 3: per-frame cost, measured in real browsers (bench/browser-matrix.mjs)
fig, ax = plt.subplots(figsize=(11, 5))
engines = ["Chromium 145", "WebKit 26"]
v1, eng, ort = [0.038, 0.035], [0.052, 0.050], [0.182, 0.184]
x = np.arange(len(engines)); w = 0.26
ax.bar(x - w, v1, w, color=AURICLE, label="Auricle v1, as shipped")
ax.bar(x, eng, w, color=THIRD, label="Auricle engine with Silero v6 weights")
ax.bar(x + w, ort, w, color=OTHER, label="vad-web (onnxruntime-web, Silero v6)")
for i in range(2):
    ax.text(x[i] - w, v1[i] + 0.004, f"{v1[i]:.3f}", ha="center", color=TEXT, fontweight="bold")
    ax.text(x[i], eng[i] + 0.004, f"{eng[i]:.3f}", ha="center", color=TEXT)
    ax.text(x[i] + w, ort[i] + 0.004, f"{ort[i]:.3f}", ha="center", color=TEXT, fontweight="bold")
ax.set_xticks(x, engines); ax.set_ylabel("ms to score one 32 ms frame")
ax.set_ylim(0, 0.30); ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25]); ax.grid(axis="x", visible=False)
ax.set_title("Per-frame cost: 5x less CPU as shipped, 3.5x with identical weights", loc="left")
ax.legend(loc="upper center", ncol=3, fontsize=12)
save(fig, "speed.png")

# 4: start-up
bars("Start-up: time from first call to a model ready to score", ["Auricle v1, Chromium", "Auricle v1, WebKit", "Auricle engine + Silero weights, Chromium", "Auricle engine + Silero weights, WebKit", "vad-web, Chromium", "vad-web, WebKit"], [4, 10, 13, 15, 215, 175], [AURICLE, AURICLE, THIRD, THIRD, OTHER, OTHER], "milliseconds (network time excluded)", "startup.png")

# 5: download size
bars("Download before the first frame can be scored", ["Auricle with Auricle v1", "Auricle with Silero v6", "vad-web with Silero v6"], [346, 948, 5520], [AURICLE, AURICLE, OTHER], "KB gzipped", "size.png", note="vad-web ships the ONNX Runtime (3,583 KB) and the model file (1,900 KB); Auricle compiles the model into 346 or 948 KB of WASM with no runtime.")

# 6: turn end
bars("How long after you stop talking the agent knows you stopped", ["Auricle", "vad-web defaults"], [324, 1432], [AURICLE, OTHER], "milliseconds, median over 240 labelled utterances", "turn-end.png", note="Same speech probabilities, different end-of-turn policy. Both found every utterance with zero false alarms.")

# 7: languages
langs = json.loads((ROOT / "bench/langs-v1.json").read_text())
order = [("speech_test", "English"), ("lang_es_419", "Spanish"), ("lang_de_de", "German"), ("lang_ar_eg", "Arabic"), ("lang_hi_in", "Hindi"), ("lang_cmn_hans_cn", "Mandarin"), ("lang_ja_jp", "Japanese")]
fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(len(order)); w = 0.38
s = [100 * langs[k]["competing"]["student"]["background_false_alarm_rate"] for k, _ in order]
t = [100 * langs[k]["competing"]["teacher"]["background_false_alarm_rate"] for k, _ in order]
ax.bar(x - w / 2, s, w, color=AURICLE, label="Auricle v1")
ax.bar(x + w / 2, t, w, color=OTHER, label="Silero v6")
for i in range(len(order)):
    ax.text(x[i] - w / 2, s[i] + 1.5, f"{s[i]:.0f}%", ha="center", color=TEXT, fontsize=12, fontweight="bold")
    ax.text(x[i] + w / 2, t[i] + 1.5, f"{t[i]:.0f}%", ha="center", color=TEXT, fontsize=12)
ax.set_xticks(x, [n for _, n in order]); ax.set_ylim(0, 118); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.grid(axis="x", visible=False)
ax.set_ylabel("Other person's speech wrongly flagged, %")
ax.set_title("Seven languages, a second talker in the room: lower is better", loc="left")
ax.legend(loc="upper center", ncol=2)
save(fig, "languages.png")
