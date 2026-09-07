//! Regression tests for `aggregate_corpus_profile`, each one a direct proof
//! against a specific way corpus-level aggregation was measured wrong before
//! this type existed: concatenating texts before profiling, averaging
//! per-document rates, and summing per-document lexical-diversity counts.

use larc_lexicon_engine::aggregate_corpus_profile;

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn sentence_length_pools_per_document_boundaries_not_concatenated_text() {
    // Neither turn ends in terminal punctuation — the common case for short,
    // informal samples. Concatenating them first ("run the tests check the
    // logs") gives the splitter one 6-word "sentence" (avg length 6.0).
    // Pooling each document's own split gives two 3-word sentences (avg 3.0).
    let profile = aggregate_corpus_profile(["run the tests", "check the logs"]);
    assert!(
        approx(profile.avg_sentence_length, 3.0, 0.001),
        "got {} — turns were fused across document boundaries",
        profile.avg_sentence_length
    );
}

#[test]
fn rates_pool_as_sum_of_hits_over_sum_of_words_not_mean_of_rates() {
    // doc A: 4 words, 3 commas -> per-doc rate 75/100w.
    // doc B: 20 words, 0 commas -> per-doc rate 0/100w.
    // Mean of per-document rates: (75 + 0) / 2 = 37.5 — wrong, ignores that
    // doc B is five times longer than doc A.
    // Correct pooled rate: (3 + 0) / (4 + 20) * 100 = 12.5.
    let doc_a = "a, b, c, d";
    let doc_b = "a b c d e f g h i j k l m n o p q r s t";
    let profile = aggregate_corpus_profile([doc_a, doc_b]);
    assert!(
        approx(profile.comma_rate_per_100_words, 12.5, 0.01),
        "got {} — rate was averaged across documents instead of pooled by word count",
        profile.comma_rate_per_100_words
    );
}

#[test]
fn lexical_diversity_is_corpus_scoped_not_summed_per_document() {
    // "apple" appears once in each document (2 total) -- not a corpus-level
    // hapax. "banana" and "cherry" each appear once, total -- those are.
    // A per-document-summed approach would wrongly count "apple" as a hapax
    // twice (each document sees it exactly once, locally) and report
    // TTR = 4 unique-per-doc / 4 words = 1.0 and hapax = 4/4 = 1.0.
    // Corpus-scoped: 3 distinct words / 4 tokens = 0.75 TTR;
    // 2 true hapaxes (banana, cherry) / 4 tokens = 0.5.
    let profile = aggregate_corpus_profile(["apple banana", "apple cherry"]);
    assert!(
        approx(profile.type_token_ratio, 0.75, 0.001),
        "got {} — apple's second occurrence in a different document wasn't counted",
        profile.type_token_ratio
    );
    assert!(
        approx(profile.hapax_legomena_ratio, 0.5, 0.001),
        "got {} — apple was wrongly treated as a corpus-level hapax",
        profile.hapax_legomena_ratio
    );
}

#[test]
fn empty_and_whitespace_only_texts_are_skipped_and_do_not_panic() {
    let profile = aggregate_corpus_profile(["", "   ", "real text here"]);
    assert_eq!(profile.document_count, 1);
    assert!(profile.total_words > 0);
}

#[test]
fn fully_empty_corpus_does_not_panic_or_nan() {
    let empty: [&str; 0] = [];
    let profile = aggregate_corpus_profile(empty);
    assert_eq!(profile.document_count, 0);
    assert_eq!(profile.total_words, 0);
    assert_eq!(profile.type_token_ratio, 0.0);
    assert!(!profile.imperative_rate.is_nan());
    assert!(!profile.avg_sentence_length.is_nan());
}

#[test]
fn single_document_corpus_matches_compute_linguistic_profile() {
    use larc_lexicon_engine::compute_linguistic_profile;
    let text = "I think you should check this carefully, but the fix is simple.";
    let single = compute_linguistic_profile(text);
    let corpus = aggregate_corpus_profile([text]);
    assert_eq!(corpus.document_count, 1);
    assert!(approx(
        corpus.hedge_rate_per_100_words,
        single.hedge_rate_per_100_words,
        0.01
    ));
    assert!(approx(
        corpus.contrast_connective_rate_per_100_words,
        single.contrast_connective_rate_per_100_words,
        0.01
    ));
    assert!(approx(
        corpus.avg_sentence_length,
        single.avg_sentence_length,
        0.01
    ));
}
