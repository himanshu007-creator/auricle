//! wasm-bindgen entry point for the brain worker: VAD + turn machine + text cleanup.

use auricle_turn::{Profile, TurnMachine};
use auricle_vad::{Model, Vad, FRAME_LEN};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Brain {
    vad: Box<dyn Vad>,
    model: Model,
    turn: TurnMachine,
}

impl Default for Brain {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Brain {
    /// Auricle v1, the shipped detector.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Brain {
        Self::with_model(1)
    }

    /// `model`: 1 = Auricle v1 (16 ms hops, default); 0 = the optional Silero v6 build (32 ms frames), for the engine benchmark.
    pub fn with_model(model: u8) -> Brain {
        let model = Model::from_u8(model);
        Brain { vad: model.build(), model, turn: TurnMachine::default() }
    }

    pub fn model(&self) -> u8 {
        self.model as u8
    }

    /// Runs one 512-sample 16 kHz frame; returns the speech probability.
    /// Events are collected with `take_events`.
    pub fn process(&mut self, frame: &[f32]) -> f32 {
        let frame: &[f32; FRAME_LEN] = match frame.try_into() {
            Ok(f) => f,
            Err(_) => return 0.0,
        };
        let p = self.vad.process(frame);
        self.turn.push(p);
        p
    }

    /// Flattened pairs: [event_code, at_ms, event_code, at_ms, ...].
    pub fn take_events(&mut self) -> Vec<u32> {
        self.turn.drain_events().flat_map(|(e, t)| [e as u32, t]).collect()
    }

    pub fn set_agent_playing(&mut self, playing: bool) {
        self.turn.set_agent_playing(playing);
    }

    /// 0 = speakers, 1 = headphones.
    pub fn set_profile(&mut self, profile: u8) {
        self.turn.profile = if profile == 1 { Profile::Headphones } else { Profile::Speakers };
    }

    /// `which`: 0 = agent silent, 1 = agent speaking on speakers, 2 = agent speaking on headphones.
    pub fn set_thresholds(&mut self, which: u8, positive: f32, negative: f32, hangover_ms: u32, min_speech_ms: u32) {
        let th = match which {
            0 => &mut self.turn.config.agent_silent,
            1 => &mut self.turn.config.agent_speaking_speakers,
            _ => &mut self.turn.config.agent_speaking_headphones,
        };
        th.positive = positive;
        th.negative = negative.min(positive);
        th.hangover_ms = hangover_ms;
        th.min_speech_ms = min_speech_ms;
    }

    pub fn is_speaking(&self) -> bool {
        self.turn.is_speaking()
    }

    pub fn reset(&mut self) {
        self.vad.reset();
        self.turn.reset();
    }
}

/// Rules-stage cleanup with the subsequence verifier. Returns the clean text.
#[wasm_bindgen]
pub fn clean_text(verbatim: &str) -> String {
    auricle_text::clean(verbatim).0
}

/// One byte per whitespace token of `verbatim`: 1 kept, 0 removed.
#[wasm_bindgen]
pub fn keep_mask(verbatim: &str) -> Vec<u8> {
    auricle_text::clean(verbatim).1.into_iter().map(u8::from).collect()
}

#[wasm_bindgen]
pub fn verify_subsequence(verbatim: &str, clean: &str) -> bool {
    auricle_text::is_subsequence(verbatim, clean)
}

#[wasm_bindgen]
pub fn ends_open(verbatim: &str) -> bool {
    auricle_text::ends_open(verbatim)
}
