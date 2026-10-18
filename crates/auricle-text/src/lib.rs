//! Deletion-only transcript cleanup with a subsequence guarantee, and the text
//! cues the turn manager uses to decide whether a turn is still open.
//!
//! Tokens are whitespace-separated words. Cleanup never rewrites a token: it
//! only decides which verbatim tokens to keep, and `verify` checks that the
//! result is a subsequence of the input after normalization.

/// Standalone fillers removed by the rules stage.
const FILLERS: &[&str] = &["um", "uh", "er", "ah", "hmm", "hm", "mm", "erm", "uhm", "umm", "uhh", "mhm"];

/// Function words whose immediate repeats are collapsed ("the the", "I I").
const FUNCTION_WORDS: &[&str] = &[
    "a", "an", "the", "i", "it", "is", "to", "of", "in", "on", "at", "and", "but", "or", "so", "we", "you", "he",
    "she", "they", "that", "this", "my", "your", "if", "do", "no", "not", "like", "just", "was", "were", "are",
];

/// A turn that ends with one of these is probably not finished.
const TRAILING_OPEN: &[&str] = &["and", "so", "but", "because", "or", "then", "with", "to", "the", "a", "an", "of", "if", "that"];

/// Lowercase, punctuation stripped. Empty for tokens that were only punctuation.
pub fn normalize(token: &str) -> String {
    token.chars().filter(|c| c.is_alphanumeric() || *c == '\'').flat_map(|c| c.to_lowercase()).collect()
}

/// Rules stage: returns one keep flag per whitespace token of `verbatim`.
pub fn rules_keep_mask(verbatim: &str) -> Vec<bool> {
    let tokens: Vec<&str> = verbatim.split_whitespace().collect();
    let norm: Vec<String> = tokens.iter().map(|t| normalize(t)).collect();
    let mut keep = vec![true; tokens.len()];
    let mut last_kept: Option<usize> = None;
    for i in 0..tokens.len() {
        let n = &norm[i];
        let drop = n.is_empty()
            || FILLERS.contains(&n.as_str())
            // Stutter fragment: "th-" or "b-" written with a trailing hyphen.
            || tokens[i].ends_with('-')
            // Immediate repeat of a function word.
            || last_kept.is_some_and(|k| FUNCTION_WORDS.contains(&n.as_str()) && norm[k] == *n)
            // Short fragment that the next word begins with: "wha what".
            || (n.len() <= 3 && i + 1 < tokens.len() && norm[i + 1].len() > n.len() && norm[i + 1].starts_with(n.as_str()));
        if drop {
            keep[i] = false;
        } else {
            last_kept = Some(i);
        }
    }
    keep
}

/// Verifier: true when `clean` is a subsequence of `verbatim` after normalization.
pub fn is_subsequence(verbatim: &str, clean: &str) -> bool {
    let mut source = verbatim.split_whitespace().map(normalize).filter(|t| !t.is_empty());
    clean.split_whitespace().map(normalize).filter(|t| !t.is_empty()).all(|c| source.any(|s| s == c))
}

/// Applies the rules stage and the verifier. Falls back to the verbatim text
/// (with every token kept) if the verifier ever fails, so the output can never
/// contain words the user did not say.
pub fn clean(verbatim: &str) -> (String, Vec<bool>) {
    let mask = rules_keep_mask(verbatim);
    let cleaned = join_kept(verbatim, &mask);
    if is_subsequence(verbatim, &cleaned) {
        (cleaned, mask)
    } else {
        (verbatim.split_whitespace().collect::<Vec<_>>().join(" "), vec![true; mask.len()])
    }
}

pub fn join_kept(verbatim: &str, keep: &[bool]) -> String {
    verbatim.split_whitespace().zip(keep).filter(|(_, k)| **k).map(|(t, _)| t).collect::<Vec<_>>().join(" ")
}

/// Floor-holding cue: the verbatim text ends on a filler, a conjunction or an
/// unfinished clause, so the user is probably still mid-turn.
pub fn ends_open(verbatim: &str) -> bool {
    let Some(last) = verbatim.split_whitespace().last() else { return false };
    let n = normalize(last);
    let ends_with_terminal_punct = last.ends_with(['.', '?', '!']);
    !ends_with_terminal_punct && (FILLERS.contains(&n.as_str()) || TRAILING_OPEN.contains(&n.as_str()) || last.ends_with(','))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_fillers_and_repeats() {
        let (clean, mask) = clean("Um, I I want to to go to, uh, Boston th- this week.");
        assert_eq!(clean, "I want to go to, Boston this week.");
        assert_eq!(mask.iter().filter(|k| !**k).count(), 5);
    }

    #[test]
    fn keeps_meaningful_repeats() {
        let (clean, _) = clean("it was very very good");
        assert_eq!(clean, "it was very very good");
    }

    #[test]
    fn verifier_rejects_invented_words() {
        assert!(is_subsequence("I want to go", "I go"));
        assert!(!is_subsequence("I want to go", "I go now"));
        assert!(!is_subsequence("I want to go", "go I"));
    }

    #[test]
    fn open_turn_cues() {
        assert!(ends_open("I was thinking that we could, um"));
        assert!(ends_open("so I went to the store and"));
        assert!(!ends_open("so I went to the store."));
        assert!(!ends_open("book it for Tuesday"));
    }
}
