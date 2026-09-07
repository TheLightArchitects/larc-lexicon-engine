//! Pure-Rust computation of `LinguisticProfile`. No POS tagger, no trained
//! model, no network — every field here is tokenization + word-list lookup,
//! so results are deterministic and reproducible from the text alone.

use std::collections::HashMap;

use crate::schema::{CorpusProfile, LinguisticProfile};

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

/// Canonical word count — the *same* tokenization [`compute_linguistic_profile`]
/// uses, so a `VoiceSample::word_count` and the `*_per_100_words` rates in its
/// `LinguisticProfile` are always denominated identically.
///
/// Deliberately not `split_whitespace().count()`. That convention counts
/// `wait—no` as one word where this tokenizer counts two — desynchronizing
/// `word_count` from precisely the dash-joined constructions that
/// `dash_rate_per_100_words` exists to measure. Any code building a
/// `VoiceSample` should call this rather than rolling its own count.
pub fn word_count(text: &str) -> u32 {
    tokenize_words(text).len() as u32
}

/// Whether `text`'s first word is a bare imperative verb — the same
/// heuristic [`compute_linguistic_profile`]'s `imperative_rate` uses per
/// sentence, exposed standalone so callers checking one short text (a whole
/// chat turn, typically one sentence) don't need to duplicate
/// [`IMPERATIVE_VERBS`] to ask the same question.
pub fn starts_with_imperative(text: &str) -> bool {
    tokenize_words(text)
        .first()
        .is_some_and(|w| IMPERATIVE_VERBS.contains(&w.as_str()))
}

/// Whether `text` contains any hedge word or phrase — the same word lists
/// [`compute_linguistic_profile`]'s `hedge_rate_per_100_words` counts,
/// exposed as a presence check for callers that want "did this turn hedge
/// at all" rather than a corpus-wide rate.
pub fn contains_hedge(text: &str) -> bool {
    let words = tokenize_words(text);
    let text_lower = text.to_lowercase();
    count_word_hits(&words, HEDGE_WORDS) > 0 || count_phrase_hits(&text_lower, HEDGE_PHRASES) > 0
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

/// Raw per-document counts that both [`compute_linguistic_profile`] (one
/// document) and [`aggregate_corpus_profile`] (many, pooled) are built from.
///
/// Kept as one source of truth so the two never define a metric differently
/// — the entire reason `aggregate_corpus_profile` exists is that its numbers
/// must be reachable the same way a single profile's are, just pooled
/// correctly instead of averaged or computed over concatenated text.
struct TextStats {
    words: Vec<String>,
    total_words: u32,
    total_chars: usize,
    /// Word count of each sentence, using this text's own sentence
    /// boundaries. A corpus aggregate pools these lists across documents —
    /// it never concatenates raw text first, which is what silently fuses
    /// adjacent turns lacking terminal punctuation into one inflated
    /// "sentence".
    sentence_word_counts: Vec<u32>,
    num_sentences: u32,
    comma_count: u32,
    semicolon_count: u32,
    dash_count: u32,
    question_count: u32,
    exclamation_count: u32,
    total_syllables: u32,
    complex_word_count: u32,
    first_person_singular_hits: u32,
    first_person_plural_hits: u32,
    second_person_hits: u32,
    certainty_hits: u32,
    hedge_hits: u32,
    causal_hits: u32,
    contrast_hits: u32,
    imperative_sentences: u32,
}

fn text_stats(text: &str) -> TextStats {
    let words = tokenize_words(text);
    let text_lower = text.to_lowercase();
    let sentences = split_sentences(text);

    let total_words = words.len() as u32;
    let total_chars: usize = words.iter().map(|w| w.chars().count()).sum();

    let sentence_word_counts: Vec<u32> = sentences
        .iter()
        .map(|s| tokenize_words(s).len() as u32)
        .collect();
    let num_sentences = sentences.len() as u32;

    let comma_count = text.matches(',').count() as u32;
    let semicolon_count = text.matches(';').count() as u32;
    let dash_count = (text.matches('—').count() + text.matches("--").count()) as u32;
    let question_count = text.matches('?').count() as u32;
    let exclamation_count = text.matches('!').count() as u32;

    let syllable_counts: Vec<u32> = words.iter().map(|w| count_syllables(w)).collect();
    let total_syllables: u32 = syllable_counts.iter().sum();
    let complex_word_count = syllable_counts.iter().filter(|&&s| s >= 3).count() as u32;

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
        .count() as u32;

    TextStats {
        words,
        total_words,
        total_chars,
        sentence_word_counts,
        num_sentences,
        comma_count,
        semicolon_count,
        dash_count,
        question_count,
        exclamation_count,
        total_syllables,
        complex_word_count,
        first_person_singular_hits,
        first_person_plural_hits,
        second_person_hits,
        certainty_hits,
        hedge_hits,
        causal_hits,
        contrast_hits,
        imperative_sentences,
    }
}

/// Compute every tokenization-derivable field of `LinguisticProfile` from
/// raw text. The `Option` fields (formality, sentiment) are left `None` —
/// they need a POS tagger / affect lexicon this crate doesn't bundle.
///
/// Single-document only. For many short texts from one author, use
/// [`aggregate_corpus_profile`] instead of concatenating them and calling
/// this once, or of averaging many calls to this — both are invalid; see
/// [`CorpusProfile`] for why.
pub fn compute_linguistic_profile(text: &str) -> LinguisticProfile {
    let stats = text_stats(text);
    let total_words = stats.total_words;

    let avg_word_length = if total_words > 0 {
        stats.total_chars as f32 / total_words as f32
    } else {
        0.0
    };

    let unique_words: std::collections::HashSet<&str> =
        stats.words.iter().map(|w| w.as_str()).collect();
    let type_token_ratio = if total_words > 0 {
        unique_words.len() as f32 / total_words as f32
    } else {
        0.0
    };

    let mut word_counts: HashMap<&str, u32> = HashMap::new();
    for w in &stats.words {
        *word_counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let hapax_count = word_counts.values().filter(|&&c| c == 1).count() as u32;
    let hapax_legomena_ratio = if total_words > 0 {
        hapax_count as f32 / total_words as f32
    } else {
        0.0
    };

    let sentence_lengths: Vec<f32> = stats
        .sentence_word_counts
        .iter()
        .map(|&n| n as f32)
        .collect();
    let avg_sentence_length = mean(&sentence_lengths);
    let sentence_length_stddev = stddev(&sentence_lengths);

    // `.max(1)`: only ever 0 when `text` is itself empty (`split_sentences`
    // guarantees at least one "sentence" for any non-empty input), so this
    // avoids a 0/0 on empty text without disturbing any real corpus.
    let num_sentences = stats.num_sentences.max(1) as f32;
    let words_per_sentence = if total_words > 0 {
        total_words as f32 / num_sentences
    } else {
        0.0
    };
    let syllables_per_word = if total_words > 0 {
        stats.total_syllables as f32 / total_words as f32
    } else {
        0.0
    };

    // Flesch (1948); Kincaid et al. (1975); Gunning (1952) — standard closed-form formulas.
    // Gated by `total_words > 0` like every other field here (see
    // `gunning_fog_index` just below): both formulas have a non-zero
    // y-intercept, so on empty input they'd otherwise report specific
    // fabricated scores instead of "no signal."
    let flesch_reading_ease = if total_words > 0 {
        206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    } else {
        0.0
    };
    let flesch_kincaid_grade = if total_words > 0 {
        0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    } else {
        0.0
    };
    let gunning_fog_index = if total_words > 0 {
        0.4 * (words_per_sentence + 100.0 * (stats.complex_word_count as f32 / total_words as f32))
    } else {
        0.0
    };

    let imperative_rate = stats.imperative_sentences as f32 / num_sentences;

    LinguisticProfile {
        type_token_ratio,
        avg_word_length,
        hapax_legomena_ratio,
        avg_sentence_length,
        sentence_length_stddev,
        comma_rate_per_100_words: per_100_words(stats.comma_count, total_words),
        semicolon_rate_per_100_words: per_100_words(stats.semicolon_count, total_words),
        dash_rate_per_100_words: per_100_words(stats.dash_count, total_words),
        question_rate_per_100_words: per_100_words(stats.question_count, total_words),
        exclamation_rate_per_100_words: per_100_words(stats.exclamation_count, total_words),
        flesch_reading_ease,
        flesch_kincaid_grade,
        gunning_fog_index,
        first_person_singular_rate_per_100_words: per_100_words(
            stats.first_person_singular_hits,
            total_words,
        ),
        first_person_plural_rate_per_100_words: per_100_words(
            stats.first_person_plural_hits,
            total_words,
        ),
        second_person_rate_per_100_words: per_100_words(stats.second_person_hits, total_words),
        imperative_rate,
        certainty_rate_per_100_words: per_100_words(stats.certainty_hits, total_words),
        hedge_rate_per_100_words: per_100_words(stats.hedge_hits, total_words),
        causal_connective_rate_per_100_words: per_100_words(stats.causal_hits, total_words),
        contrast_connective_rate_per_100_words: per_100_words(stats.contrast_hits, total_words),
        formality_score: None,
        sentiment_polarity: None,
        sentiment_subjectivity: None,
    }
}

/// Correctly pooled [`CorpusProfile`] over many texts (e.g. every
/// `VoiceSample::text` from one author). See [`CorpusProfile`]'s docs for why
/// this cannot be approximated by concatenating texts and calling
/// [`compute_linguistic_profile`] once, or by averaging many single-document
/// profiles.
///
/// Empty and whitespace-only texts are skipped and do not count toward
/// `document_count`.
pub fn aggregate_corpus_profile<'a>(texts: impl IntoIterator<Item = &'a str>) -> CorpusProfile {
    let mut document_count = 0u32;
    let mut total_words = 0u32;
    let mut total_chars: usize = 0;
    let mut word_freq: HashMap<String, u32> = HashMap::new();
    let mut sentence_lengths: Vec<f32> = Vec::new();
    let mut total_sentences = 0u32;
    let mut comma = 0u32;
    let mut semicolon = 0u32;
    let mut dash = 0u32;
    let mut question = 0u32;
    let mut exclamation = 0u32;
    let mut total_syllables = 0u32;
    let mut complex_words = 0u32;
    let mut fp_singular = 0u32;
    let mut fp_plural = 0u32;
    let mut second_person = 0u32;
    let mut certainty = 0u32;
    let mut hedge = 0u32;
    let mut causal = 0u32;
    let mut contrast = 0u32;
    let mut imperative_sentences = 0u32;

    for text in texts {
        if text.trim().is_empty() {
            continue;
        }
        document_count += 1;
        let stats = text_stats(text);

        total_words += stats.total_words;
        total_chars += stats.total_chars;
        for w in stats.words {
            *word_freq.entry(w).or_insert(0) += 1;
        }
        sentence_lengths.extend(stats.sentence_word_counts.iter().map(|&n| n as f32));
        total_sentences += stats.num_sentences;
        comma += stats.comma_count;
        semicolon += stats.semicolon_count;
        dash += stats.dash_count;
        question += stats.question_count;
        exclamation += stats.exclamation_count;
        total_syllables += stats.total_syllables;
        complex_words += stats.complex_word_count;
        fp_singular += stats.first_person_singular_hits;
        fp_plural += stats.first_person_plural_hits;
        second_person += stats.second_person_hits;
        certainty += stats.certainty_hits;
        hedge += stats.hedge_hits;
        causal += stats.causal_hits;
        contrast += stats.contrast_hits;
        imperative_sentences += stats.imperative_sentences;
    }

    // Corpus-scoped, not summed per-document: a word appearing once in each
    // of two documents is not a corpus-level hapax, and the two documents'
    // separate "100% unique" ratios don't compose into a corpus TTR.
    let type_token_ratio = if total_words > 0 {
        word_freq.len() as f32 / total_words as f32
    } else {
        0.0
    };
    let hapax_count = word_freq.values().filter(|&&c| c == 1).count() as u32;
    let hapax_legomena_ratio = if total_words > 0 {
        hapax_count as f32 / total_words as f32
    } else {
        0.0
    };
    let avg_word_length = if total_words > 0 {
        total_chars as f32 / total_words as f32
    } else {
        0.0
    };

    let avg_sentence_length = mean(&sentence_lengths);
    let sentence_length_stddev = stddev(&sentence_lengths);

    let sentence_denom = total_sentences.max(1) as f32;
    let words_per_sentence = if total_words > 0 {
        total_words as f32 / sentence_denom
    } else {
        0.0
    };
    let syllables_per_word = if total_words > 0 {
        total_syllables as f32 / total_words as f32
    } else {
        0.0
    };

    // Gated by `total_words > 0` for the same reason as in
    // `compute_linguistic_profile`: both formulas have a non-zero
    // y-intercept, so an empty corpus would otherwise report fabricated
    // scores instead of "no signal."
    let flesch_reading_ease = if total_words > 0 {
        206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    } else {
        0.0
    };
    let flesch_kincaid_grade = if total_words > 0 {
        0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    } else {
        0.0
    };
    let gunning_fog_index = if total_words > 0 {
        0.4 * (words_per_sentence + 100.0 * (complex_words as f32 / total_words as f32))
    } else {
        0.0
    };

    let imperative_rate = imperative_sentences as f32 / sentence_denom;

    CorpusProfile {
        document_count,
        total_words,
        type_token_ratio,
        avg_word_length,
        hapax_legomena_ratio,
        avg_sentence_length,
        sentence_length_stddev,
        comma_rate_per_100_words: per_100_words(comma, total_words),
        semicolon_rate_per_100_words: per_100_words(semicolon, total_words),
        dash_rate_per_100_words: per_100_words(dash, total_words),
        question_rate_per_100_words: per_100_words(question, total_words),
        exclamation_rate_per_100_words: per_100_words(exclamation, total_words),
        flesch_reading_ease,
        flesch_kincaid_grade,
        gunning_fog_index,
        first_person_singular_rate_per_100_words: per_100_words(fp_singular, total_words),
        first_person_plural_rate_per_100_words: per_100_words(fp_plural, total_words),
        second_person_rate_per_100_words: per_100_words(second_person, total_words),
        imperative_rate,
        certainty_rate_per_100_words: per_100_words(certainty, total_words),
        hedge_rate_per_100_words: per_100_words(hedge, total_words),
        causal_connective_rate_per_100_words: per_100_words(causal, total_words),
        contrast_connective_rate_per_100_words: per_100_words(contrast, total_words),
        formality_score: None,
        sentiment_polarity: None,
        sentiment_subjectivity: None,
    }
}
