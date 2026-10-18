//! Voice activity detection behind one small trait, as a hand-written forward
//! pass with no inference runtime. `Auricle` is Auricle v1, our
//! foreground-selective model (16 kHz, 16 ms hop), on by default. The optional
//! `silero` feature adds `Silero` (Silero VAD v6) so the engine can be
//! benchmarked against other runtimes with identical weights.

#[cfg(feature = "auricle")]
mod auricle;
mod math;
#[cfg(feature = "silero")]
mod silero;

#[cfg(feature = "auricle")]
pub use auricle::Auricle;
#[cfg(feature = "silero")]
pub use silero::Silero;

#[cfg(not(any(feature = "silero", feature = "auricle")))]
compile_error!("auricle-vad needs at least one of the `silero` and `auricle` features");

/// Samples per VAD frame at 16 kHz (32 ms).
pub const FRAME_LEN: usize = 512;
pub const SAMPLE_RATE: u32 = 16_000;

pub trait Vad {
    /// Speech probability for one 512-sample 16 kHz frame.
    fn process(&mut self, frame: &[f32; FRAME_LEN]) -> f32;
    fn reset(&mut self);
}

/// Which bundled detector to construct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Model {
    Silero = 0,
    #[default]
    Auricle = 1,
}

impl Model {
    pub fn from_u8(v: u8) -> Self {
        if v == 0 { Model::Silero } else { Model::Auricle }
    }

    /// Builds the detector; a model compiled out of this build falls back to the one compiled in.
    pub fn build(self) -> Box<dyn Vad> {
        match self {
            #[cfg(feature = "silero")]
            Model::Silero => Box::new(Silero::new()),
            #[cfg(feature = "auricle")]
            Model::Auricle => Box::new(Auricle::new()),
            #[allow(unreachable_patterns)]
            _ => Self::compiled_in(),
        }
    }

    /// The detector this build actually contains.
    pub fn compiled_in() -> Box<dyn Vad> {
        #[cfg(feature = "auricle")]
        {
            Box::new(Auricle::new())
        }
        #[cfg(not(feature = "auricle"))]
        {
            Box::new(Silero::new())
        }
    }
}
