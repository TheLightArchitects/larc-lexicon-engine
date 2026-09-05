//! Pure-Rust computation of `LinguisticProfile`. No POS tagger, no trained
//! model, no network — every field here is tokenization + word-list lookup,
//! so results are deterministic and reproducible from the text alone.

use crate::schema::LinguisticProfile;

const FIRST_PERSON_SINGULAR: &[&str] = &["i", "me", "my", "mine", "myself"];
const FIRST_PERSON_PLURAL: &[&str] = &["we", "us", "our", "ours", "ourselves"];
const SECOND_PERSON: &[&str] = &["you", "your", "yours", "yourself", "yourselves"];

// Hyland (1998) epistemic-stance categories: boosters (certainty) vs. hedges.
const CERTAINTY_WORDS: &[&str] = &[
    "definitely",
    "always",
    "never",
    "clearly",
    "obviously",
    "certainly",
    "undoubtedly",
    "must",
    "absolutely",
    "surely",
    "guaranteed",
    "fact",
];
const HEDGE_WORDS: &[&str] = &[
    "maybe",
    "perhaps",
    "might",
    "possibly",
    "probably",
    "seems",
    "appears",
    "somewhat",
    "likely",
    "roughly",
    "approximately",
];
const HEDGE_PHRASES: &[&str] = &[
    "i think",
    "i guess",
    "i believe",
    "kind of",
    "sort of",
    "not sure",
    "could be",
];

const CAUSAL_CONNECTIVES: &[&str] = &[
    "because",
    "therefore",
    "so",
    "since",
    "thus",
    "hence",
    "consequently",
];
const CONTRAST_CONNECTIVES: &[&str] = &[
    "but",
    "however",
    "although",
    "yet",
    "though",
    "nevertheless",
    "whereas",
];

// Heuristic verb-first list for imperative-mood detection without a POS
// tagger — common command verbs. Approximate by construction: a sentence
// starting with one of these is *likely* imperative, not certainly so.
const IMPERATIVE_VERBS: &[&str] = &[
    "check", "run", "use", "add", "remove", "try", "make", "build", "write", "read", "fix",
    "update", "ensure", "verify", "confirm", "stop", "go", "look", "find", "show", "give", "take",
    "keep", "let", "start", "open", "close", "move", "send", "call", "set", "get", "put", "do",
    "don't", "never", "always", "avoid", "consider", "note", "remember", "please",
];

fn tokenize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Approximate sentence splitter on `.`/`!`/`?`. Does not special-case
/// abbreviations (e.g. "Dr.", "e.g.") — acceptable for corpus-level
/// aggregate statistics, not intended for exact sentence boundary detection.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            let end = i + c.len_utf8();
            let candidate = text[start..end].trim();
            if !candidate.is_empty() {
                sentences.push(candidate);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim());
    }
    sentences
}

/// Vowel-group heuristic syllable counter (the standard approximation used
/// by most practical readability-formula implementations).
fn count_syllables(word: &str) -> u32 {
    let word: String = word.chars().filter(|c| c.is_alphabetic()).collect();
    let word = word.to_lowercase();
    if word.is_empty() {
        return 0;
    }
    let vowels = "aeiouy";
    let mut count = 0u32;
    let mut prev_was_vowel = false;
    for c in word.chars() {
        let is_vowel = vowels.contains(c);
        if is_vowel && !prev_was_vowel {
            count += 1;
        }
        prev_was_vowel = is_vowel;
    }
    if word.ends_with('e') && count > 1 {
        count -= 1;
    }
    count.max(1)
}

fn count_word_hits(words: &[String], list: &[&str]) -> u32 {
    words.iter().filter(|w| list.contains(&w.as_str())).count() as u32
}

fn count_phrase_hits(text_lower: &str, phrases: &[&str]) -> u32 {
    phrases
        .iter()
        .map(|p| text_lower.matches(p).count() as u32)
        .sum()
}

fn per_100_words(count: u32, total_words: u32) -> f32 {
    if total_words == 0 {
        0.0
    } else {
        count as f32 / total_words as f32 * 100.0
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn stddev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

/// Compute every tokenization-derivable field of `LinguisticProfile` from
/// raw text. The `Option` fields (formality, sentiment) are left `None` —
/// they need a POS tagger / affect lexicon this crate doesn't bundle.
pub fn compute_linguistic_profile(text: &str) -> LinguisticProfile {
    let words = tokenize_words(text);
    let text_lower = text.to_lowercase();
    let sentences = split_sentences(text);

    let total_words = words.len() as u32;
    let total_chars: usize = words.iter().map(|w| w.chars().count()).sum();
    let avg_word_length = if total_words > 0 {
        total_chars as f32 / total_words as f32
    } else {
        0.0
    };

    let unique_words: std::collections::HashSet<&str> = words.iter().map(|w| w.as_str()).collect();
    let type_token_ratio = if total_words > 0 {
        unique_words.len() as f32 / total_words as f32
    } else {
        0.0
    };

    let mut word_counts: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for w in &words {
        *word_counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let hapax_count = word_counts.values().filter(|&&c| c == 1).count() as u32;
    let hapax_legomena_ratio = if total_words > 0 {
        hapax_count as f32 / total_words as f32
    } else {
        0.0
    };

    let sentence_lengths: Vec<f32> = sentences
        .iter()
        .map(|s| tokenize_words(s).len() as f32)
        .collect();
    let avg_sentence_length = mean(&sentence_lengths);
    let sentence_length_stddev = stddev(&sentence_lengths);

    let comma_count = text.matches(',').count() as u32;
    let semicolon_count = text.matches(';').count() as u32;
    let dash_count = (text.matches('—').count() + text.matches("--").count()) as u32;
    let question_count = text.matches('?').count() as u32;
    let exclamation_count = text.matches('!').count() as u32;

    let syllable_counts: Vec<u32> = words.iter().map(|w| count_syllables(w)).collect();
    let total_syllables: u32 = syllable_counts.iter().sum();
    let complex_word_count = syllable_counts.iter().filter(|&&s| s >= 3).count() as u32;
    let num_sentences = sentences.len().max(1) as f32;
    let words_per_sentence = if total_words > 0 {
        total_words as f32 / num_sentences
    } else {
        0.0
    };
    let syllables_per_word = if total_words > 0 {
        total_syllables as f32 / total_words as f32
    } else {
        0.0
    };

    // Flesch (1948); Kincaid et al. (1975); Gunning (1952) — standard closed-form formulas.
    let flesch_reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word;
    let flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59;
    let gunning_fog_index = if total_words > 0 {
        0.4 * (words_per_sentence + 100.0 * (complex_word_count as f32 / total_words as f32))
    } else {
        0.0
    };

    let first_person_singular_hits = count_word_hits(&words, FIRST_PERSON_SINGULAR);
    let first_person_plural_hits = count_word_hits(&words, FIRST_PERSON_PLURAL);
    let second_person_hits = count_word_hits(&words, SECOND_PERSON);

    let certainty_hits = count_word_hits(&words, CERTAINTY_WORDS);
    let hedge_hits =
        count_word_hits(&words, HEDGE_WORDS) + count_phrase_hits(&text_lower, HEDGE_PHRASES);

    let causal_hits = count_word_hits(&words, CAUSAL_CONNECTIVES);
    let contrast_hits = count_word_hits(&words, CONTRAST_CONNECTIVES);

    let imperative_sentences = sentences
        .iter()
        .filter(|s| {
            tokenize_words(s)
                .first()
                .map(|w| IMPERATIVE_VERBS.contains(&w.as_str()))
                .unwrap_or(false)
        })
        .count() as f32;
    let imperative_rate = imperative_sentences / num_sentences;

    LinguisticProfile {
        type_token_ratio,
        avg_word_length,
        hapax_legomena_ratio,
        avg_sentence_length,
        sentence_length_stddev,
        comma_rate_per_100_words: per_100_words(comma_count, total_words),
        semicolon_rate_per_100_words: per_100_words(semicolon_count, total_words),
        dash_rate_per_100_words: per_100_words(dash_count, total_words),
        question_rate_per_100_words: per_100_words(question_count, total_words),
        exclamation_rate_per_100_words: per_100_words(exclamation_count, total_words),
        flesch_reading_ease,
        flesch_kincaid_grade,
        gunning_fog_index,
        first_person_singular_rate_per_100_words: per_100_words(
            first_person_singular_hits,
            total_words,
        ),
        first_person_plural_rate_per_100_words: per_100_words(
            first_person_plural_hits,
            total_words,
        ),
        second_person_rate_per_100_words: per_100_words(second_person_hits, total_words),
        imperative_rate,
        certainty_rate_per_100_words: per_100_words(certainty_hits, total_words),
        hedge_rate_per_100_words: per_100_words(hedge_hits, total_words),
        causal_connective_rate_per_100_words: per_100_words(causal_hits, total_words),
        contrast_connective_rate_per_100_words: per_100_words(contrast_hits, total_words),
        formality_score: None,
        sentiment_polarity: None,
        sentiment_subjectivity: None,
    }
}
