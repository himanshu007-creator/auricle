"""Builds a labelled speech set from macOS `say`: several voices, utterances
separated by known gaps, white noise at known SNRs. Labels are tight speech
intervals measured from the clean TTS audio, so both endpointers are scored
against the same ground truth.

    python3 bench/make_dataset.py bench/data
"""
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

RATE = 16000
VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa"]
SENTENCES = [
    "Can you book a table for two at seven tomorrow?",
    "I think the second option is better, but let me check the budget first.",
    "No.",
    "Yes, that works for me.",
    "What was the total for last month, including the shipping fees and taxes?",
    "Hmm, let me think about that for a second.",
    "Send it to the whole team, please.",
    "Actually, wait, change the meeting to Thursday instead of Wednesday afternoon.",
    "Okay.",
    "Tell me a joke about printers.",
]
GAPS_S = [0.3, 0.5, 0.8, 1.2, 2.0]
SNRS_DB = [None, 20, 10, 5]
LEAD_S = 1.0


def available_voices():
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    names = {line.split()[0] for line in out.splitlines() if line.strip()}
    return [v for v in VOICES if v in names] or ["Samantha"]


def tts(voice, text, tmp):
    aiff = tmp / "u.aiff"
    wav = tmp / "u.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1", str(aiff), str(wav)], check=True)
    with wave.open(str(wav)) as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
    return trim(pcm)


def trim(x, threshold_db=-40.0, win=160):
    """Cuts leading/trailing silence using a 10 ms RMS envelope."""
    frames = len(x) // win
    rms = np.sqrt(np.mean(x[: frames * win].reshape(frames, win) ** 2, axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms) > threshold_db)[0]
    if loud.size == 0:
        return x
    return x[loud[0] * win : (loud[-1] + 1) * win]


def write_wav(path, x):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    labels = {}
    tmp = Path(tempfile.mkdtemp())
    for vi, voice in enumerate(available_voices()):
        utterances = [tts(voice, s, tmp) for s in SENTENCES]
        order = rng.permutation(len(utterances))
        pieces = [np.zeros(int(LEAD_S * RATE), np.float32)]
        intervals = []
        gaps = []
        cursor = LEAD_S
        for i, idx in enumerate(order):
            u = utterances[idx]
            intervals.append([round(cursor, 3), round(cursor + len(u) / RATE, 3)])
            pieces.append(u)
            cursor += len(u) / RATE
            gap = GAPS_S[(i + vi) % len(GAPS_S)]
            gaps.append(gap)
            pieces.append(np.zeros(int(gap * RATE), np.float32))
            cursor += gap
        pieces.append(np.zeros(int(LEAD_S * RATE), np.float32))
        clean = np.concatenate(pieces)
        speech_mask = np.zeros(len(clean), bool)
        for s, e in intervals:
            speech_mask[int(s * RATE) : int(e * RATE)] = True
        speech_rms = np.sqrt(np.mean(clean[speech_mask] ** 2))
        for snr in SNRS_DB:
            name = f"{voice.lower()}_snr{'clean' if snr is None else snr}.wav"
            if snr is None:
                mixed = clean
            else:
                noise = rng.standard_normal(len(clean)).astype(np.float32)
                noise *= speech_rms / (10 ** (snr / 20)) / np.sqrt(np.mean(noise**2))
                mixed = clean + noise
            write_wav(out / name, mixed)
            labels[name] = {"voice": voice, "snr_db": snr, "speech": intervals, "gaps_s": gaps}
        print(f"{voice}: {len(intervals)} utterances, {cursor:.1f} s")
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main(sys.argv[1])
