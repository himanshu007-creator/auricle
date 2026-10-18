//! Per-frame speech evidence -> graded turn events, plus the barge-in arbiter.
//!
//! Candidate events trigger reversible actions (duck, buffer); confirmed events
//! trigger irreversible ones (stop, cancel, truncate). Thresholds depend on
//! whether the agent is playing and on the speaker profile.

/// Frame duration used for every time stamp: 512 samples at 16 kHz.
pub const FRAME_MS: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Profile {
    Speakers = 0,
    Headphones = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Event {
    /// Enough voiced frames to start reversible actions.
    SpeechCandidate = 1,
    /// Confirmed speech; irreversible actions may start.
    SpeechConfirmed = 2,
    /// Probability dropped; end of turn may follow.
    SilenceCandidate = 3,
    /// Hangover elapsed; the segment is over.
    SpeechEnd = 4,
    /// A candidate that never confirmed (noise, or a backchannel while the agent spoke).
    Misfire = 5,
    /// Arbiter: agent output should duck now.
    Duck = 6,
    /// Arbiter: restore agent output.
    Unduck = 7,
    /// Arbiter: hard stop, the user has taken over.
    Interrupt = 8,
}

impl Event {
    pub fn from_code(code: u8) -> Option<Event> {
        use Event::*;
        Some(match code {
            1 => SpeechCandidate,
            2 => SpeechConfirmed,
            3 => SilenceCandidate,
            4 => SpeechEnd,
            5 => Misfire,
            6 => Duck,
            7 => Unduck,
            8 => Interrupt,
            _ => return None,
        })
    }
}

/// Thresholds for one situation (agent silent, or agent speaking on a profile).
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    pub positive: f32,
    pub negative: f32,
    /// Consecutive frames above `positive` before SpeechCandidate.
    pub candidate_frames: u32,
    /// SpeechConfirmed when at least `confirm_hits` of the last `confirm_window` frames are positive.
    pub confirm_hits: u32,
    pub confirm_window: u32,
    /// Silence below `negative` for this long before SpeechEnd.
    pub hangover_ms: u32,
    /// A confirmed segment with fewer voiced milliseconds than this ends as a
    /// Misfire instead of SpeechEnd (rejects short noise bursts).
    pub min_speech_ms: u32,
}

impl Thresholds {
    pub const AGENT_SILENT: Thresholds =
        Thresholds { positive: 0.50, negative: 0.35, candidate_frames: 3, confirm_hits: 6, confirm_window: 10, hangover_ms: 240, min_speech_ms: 100 };
    pub const AGENT_SPEAKING_SPEAKERS: Thresholds =
        Thresholds { positive: 0.70, negative: 0.50, candidate_frames: 3, confirm_hits: 8, confirm_window: 12, hangover_ms: 320, min_speech_ms: 100 };
    pub const AGENT_SPEAKING_HEADPHONES: Thresholds =
        Thresholds { positive: 0.55, negative: 0.40, candidate_frames: 3, confirm_hits: 6, confirm_window: 10, hangover_ms: 240, min_speech_ms: 100 };
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub agent_silent: Thresholds,
    pub agent_speaking_speakers: Thresholds,
    pub agent_speaking_headphones: Thresholds,
    /// Longest time to stay ducked on an unconfirmed candidate before giving up.
    pub unclear_max_ms: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            agent_silent: Thresholds::AGENT_SILENT,
            agent_speaking_speakers: Thresholds::AGENT_SPEAKING_SPEAKERS,
            agent_speaking_headphones: Thresholds::AGENT_SPEAKING_HEADPHONES,
            unclear_max_ms: 800,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Silent,
    /// Positive frames seen but fewer than `candidate_frames`.
    Rising,
    /// SpeechCandidate emitted, not yet confirmed.
    Candidate,
    Speaking,
    /// Below negative after Speaking or Candidate; waiting out the hangover.
    Hangover,
}

pub struct TurnMachine {
    pub config: Config,
    pub profile: Profile,
    agent_playing: bool,
    phase: Phase,
    confirmed: bool,
    ducked: bool,
    frame_index: u64,
    rising_count: u32,
    hangover_frames: u32,
    candidate_started_ms: u32,
    /// Voiced frames since the current candidate started.
    voiced_frames: u32,
    /// Bit history of positive frames, newest in bit 0.
    history: u32,
    events: Vec<(Event, u32)>,
}

impl Default for TurnMachine {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl TurnMachine {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            profile: Profile::Speakers,
            agent_playing: false,
            phase: Phase::Silent,
            confirmed: false,
            ducked: false,
            frame_index: 0,
            rising_count: 0,
            hangover_frames: 0,
            candidate_started_ms: 0,
            voiced_frames: 0,
            history: 0,
            events: Vec::with_capacity(8),
        }
    }

    pub fn set_agent_playing(&mut self, playing: bool) {
        self.agent_playing = playing;
        if !playing {
            self.ducked = false;
        }
    }

    pub fn is_speaking(&self) -> bool {
        matches!(self.phase, Phase::Candidate | Phase::Speaking | Phase::Hangover)
    }

    pub fn now_ms(&self) -> u32 {
        (self.frame_index * FRAME_MS as u64) as u32
    }

    fn thresholds(&self) -> Thresholds {
        match (self.agent_playing, self.profile) {
            (false, _) => self.config.agent_silent,
            (true, Profile::Speakers) => self.config.agent_speaking_speakers,
            (true, Profile::Headphones) => self.config.agent_speaking_headphones,
        }
    }

    /// Feed one frame's speech probability. Events are read back with `drain_events`.
    pub fn push(&mut self, p: f32) {
        let th = self.thresholds();
        let now = self.now_ms();
        let positive = p >= th.positive;
        let negative = p < th.negative;
        self.history = (self.history << 1) | positive as u32;
        if positive && self.phase != Phase::Silent {
            self.voiced_frames += 1;
        }

        match self.phase {
            Phase::Silent | Phase::Rising => {
                if positive {
                    self.rising_count += 1;
                    self.phase = Phase::Rising;
                    if self.rising_count >= th.candidate_frames {
                        self.phase = Phase::Candidate;
                        self.confirmed = false;
                        self.candidate_started_ms = now;
                        self.voiced_frames = self.rising_count;
                        self.emit(Event::SpeechCandidate, now);
                        if self.agent_playing && !self.ducked {
                            self.ducked = true;
                            self.emit(Event::Duck, now);
                        }
                    }
                } else {
                    self.rising_count = 0;
                    self.phase = Phase::Silent;
                }
            }
            Phase::Candidate => {
                if self.confirm_reached(th) {
                    self.phase = Phase::Speaking;
                    self.confirmed = true;
                    self.emit(Event::SpeechConfirmed, now);
                    if self.agent_playing {
                        self.emit(Event::Interrupt, now);
                        self.ducked = false;
                    }
                } else if negative {
                    self.phase = Phase::Hangover;
                    self.hangover_frames = 0;
                } else if self.agent_playing && now - self.candidate_started_ms >= self.config.unclear_max_ms {
                    // Stayed unclear too long while ducked: give the agent its voice back.
                    self.end_segment(now);
                }
            }
            Phase::Speaking => {
                if negative {
                    self.phase = Phase::Hangover;
                    self.hangover_frames = 0;
                    self.emit(Event::SilenceCandidate, now);
                }
            }
            Phase::Hangover => {
                if positive {
                    self.phase = if self.confirmed { Phase::Speaking } else { Phase::Candidate };
                    if !self.confirmed && self.confirm_reached(th) {
                        self.phase = Phase::Speaking;
                        self.confirmed = true;
                        self.emit(Event::SpeechConfirmed, now);
                        if self.agent_playing {
                            self.emit(Event::Interrupt, now);
                            self.ducked = false;
                        }
                    }
                } else {
                    self.hangover_frames += 1;
                    if self.hangover_frames * FRAME_MS >= th.hangover_ms {
                        self.end_segment(now);
                    }
                }
            }
        }
        self.frame_index += 1;
    }

    fn confirm_reached(&self, th: Thresholds) -> bool {
        let window_mask = if th.confirm_window >= 32 { u32::MAX } else { (1u32 << th.confirm_window) - 1 };
        (self.history & window_mask).count_ones() >= th.confirm_hits
    }

    fn end_segment(&mut self, now: u32) {
        let long_enough = self.voiced_frames * FRAME_MS >= self.thresholds().min_speech_ms;
        self.emit(if self.confirmed && long_enough { Event::SpeechEnd } else { Event::Misfire }, now);
        if self.ducked {
            self.ducked = false;
            self.emit(Event::Unduck, now);
        }
        self.phase = Phase::Silent;
        self.rising_count = 0;
        self.voiced_frames = 0;
        self.confirmed = false;
    }

    fn emit(&mut self, event: Event, at_ms: u32) {
        self.events.push((event, at_ms));
    }

    /// Returns and clears the events produced since the last call.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, (Event, u32)> {
        self.events.drain(..)
    }

    pub fn reset(&mut self) {
        let config = self.config;
        let profile = self.profile;
        *self = Self::new(config);
        self.profile = profile;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(machine: &mut TurnMachine, probs: &[f32]) -> Vec<Event> {
        let mut out = Vec::new();
        for &p in probs {
            machine.push(p);
            out.extend(machine.drain_events().map(|(e, _)| e));
        }
        out
    }

    #[test]
    fn clean_speech_segment() {
        let mut m = TurnMachine::default();
        let mut probs = vec![0.1; 5];
        probs.extend(vec![0.9; 12]);
        probs.extend(vec![0.1; 10]); // 320 ms of silence > 240 ms hangover
        let events = run(&mut m, &probs);
        assert_eq!(events, vec![Event::SpeechCandidate, Event::SpeechConfirmed, Event::SilenceCandidate, Event::SpeechEnd]);
    }

    #[test]
    fn short_burst_is_a_misfire() {
        let mut m = TurnMachine::default();
        let mut probs = vec![0.9; 3];
        probs.extend(vec![0.1; 10]);
        assert_eq!(run(&mut m, &probs), vec![Event::SpeechCandidate, Event::Misfire]);
    }

    #[test]
    fn confirmed_but_too_short_is_a_misfire() {
        let mut m = TurnMachine::default();
        m.config.agent_silent.min_speech_ms = 400;
        let mut probs = vec![0.9; 7]; // 224 ms voiced: confirms, but under 400 ms
        probs.extend(vec![0.1; 10]);
        assert_eq!(run(&mut m, &probs), vec![Event::SpeechCandidate, Event::SpeechConfirmed, Event::SilenceCandidate, Event::Misfire]);
    }

    #[test]
    fn barge_in_ducks_then_interrupts() {
        let mut m = TurnMachine::default();
        m.set_agent_playing(true);
        let events = run(&mut m, &[0.9; 12]);
        assert_eq!(events, vec![Event::SpeechCandidate, Event::Duck, Event::SpeechConfirmed, Event::Interrupt]);
    }

    #[test]
    fn backchannel_unducks() {
        let mut m = TurnMachine::default();
        m.set_agent_playing(true);
        let mut probs = vec![0.9; 4];
        probs.extend(vec![0.1; 12]);
        assert_eq!(run(&mut m, &probs), vec![Event::SpeechCandidate, Event::Duck, Event::Misfire, Event::Unduck]);
    }

    #[test]
    fn unclear_overlap_gives_up_after_cap() {
        let mut m = TurnMachine::default();
        m.set_agent_playing(true);
        // Hovering between negative and positive: never confirms, never drops.
        let mut probs = vec![0.9; 3];
        probs.extend(vec![0.6; 40]);
        let events = run(&mut m, &probs);
        assert_eq!(events, vec![Event::SpeechCandidate, Event::Duck, Event::Misfire, Event::Unduck]);
    }
}
