//! Replays a WAV file through the exact crates the browser runs and prints the
//! graded events with their times, plus per-frame cost.
//!
//!   auricle-eval input.wav [--agent START_MS:END_MS ...] [--headphones] [--probs]
//!
//! `--agent` marks windows where the pretend agent is speaking, so barge-in
//! arbitration (duck / interrupt / misfire) is exercised offline.

use auricle_dsp::{FrameChunker, Resampler};
use auricle_turn::{Event, Profile, TurnMachine};
use auricle_vad::{Model, FRAME_LEN, SAMPLE_RATE};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: auricle-eval input.wav [--agent START_MS:END_MS ...] [--headphones] [--probs] [--silero]");
        std::process::exit(2);
    };
    let mut agent_windows: Vec<(u32, u32)> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--agent" {
            let (a, b) = args[i + 1].split_once(':').expect("--agent START_MS:END_MS");
            agent_windows.push((a.parse().unwrap(), b.parse().unwrap()));
            i += 1;
        }
        i += 1;
    }
    let headphones = args.iter().any(|a| a == "--headphones");
    let print_probs = args.iter().any(|a| a == "--probs");
    let model = if args.iter().any(|a| a == "--silero") { Model::Silero } else { Model::Auricle };

    let mut reader = hound::WavReader::open(path).expect("open wav");
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mono: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect::<Vec<f32>>(),
        hound::SampleFormat::Int => {
            let scale = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / scale).collect::<Vec<f32>>()
        }
    }
    .chunks(channels)
    .map(|c| c.iter().sum::<f32>() / channels as f32)
    .collect();

    let mut resampler = Resampler::new(spec.sample_rate, SAMPLE_RATE, 24);
    let mut resampled = vec![0.0; resampler.max_output_len(mono.len())];
    let n = resampler.process(&mono, &mut resampled);
    resampled.truncate(n);
    // Half a second of trailing silence so a segment that runs to the end of the file still closes.
    resampled.extend(std::iter::repeat(0.0).take(SAMPLE_RATE as usize / 2));

    let mut vad = model.build();
    let mut turn = TurnMachine::default();
    turn.profile = if headphones { Profile::Headphones } else { Profile::Speakers };
    let mut chunker = FrameChunker::new(FRAME_LEN);
    let mut frame_count = 0u32;
    let mut total = std::time::Duration::ZERO;
    let mut segments = 0;

    println!("{} Hz, {} channels, {:.2} s -> {} frames of 32 ms", spec.sample_rate, channels, mono.len() as f32 / spec.sample_rate as f32, n / FRAME_LEN);
    chunker.push(&resampled, |frame| {
        let now_ms = frame_count * 32;
        let agent_playing = agent_windows.iter().any(|&(a, b)| now_ms >= a && now_ms < b);
        turn.set_agent_playing(agent_playing);
        let start = Instant::now();
        let p = vad.process(frame.try_into().unwrap());
        turn.push(p);
        total += start.elapsed();
        if print_probs {
            println!("{now_ms:>7} ms  p={p:.3}{}", if agent_playing { "  agent" } else { "" });
        }
        for (event, at_ms) in turn.drain_events() {
            if event == Event::SpeechEnd {
                segments += 1;
            }
            println!("{at_ms:>7} ms  {event:?}");
        }
        frame_count += 1;
    });
    println!(
        "{segments} speech segments; VAD + turn cost {:.1} us per 32 ms frame (native, single thread)",
        total.as_secs_f64() * 1e6 / frame_count.max(1) as f64
    );
}
