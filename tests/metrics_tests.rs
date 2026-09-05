use larc_lexicon_engine::compute_linguistic_profile;

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn flesch_scores_match_hand_computed_values_for_trivial_sentence() {
    // "The cat sat on the mat." -- 6 words, 1 sentence, 1 syllable each.
    let profile = compute_linguistic_profile("The cat sat on the mat.");
    // Flesch Reading Ease = 206.835 - 1.015*6 - 84.6*1.0 = 116.145
    assert!(
        approx(profile.flesch_reading_ease, 116.145, 0.01),
        "got {}",
        profile.flesch_reading_ease
    );
    // Flesch-Kincaid Grade = 0.39*6 + 11.8*1.0 - 15.59 = -1.45
    assert!(
        approx(profile.flesch_kincaid_grade, -1.45, 0.01),
        "got {}",
        profile.flesch_kincaid_grade
    );
}

#[test]
fn lexical_richness_counts_are_exact_on_a_known_text() {
    // "the" repeats (hapax excludes it); every other word is unique.
    let profile = compute_linguistic_profile("the cat sat on the mat");
    // 6 tokens, 5 distinct ("the" collapses) -> TTR = 5/6
    assert!(approx(profile.type_token_ratio, 5.0 / 6.0, 0.001));
    // hapax legomena: cat, sat, on, mat = 4 singletons / 6 total tokens
    assert!(approx(profile.hapax_legomena_ratio, 4.0 / 6.0, 0.001));
}

#[test]
fn pronoun_and_hedge_detection_on_a_hedged_directive() {
    let profile = compute_linguistic_profile("I think you should check this carefully.");
    // "i" -> first person singular hit (1 of 7 words)
    assert!(profile.first_person_singular_rate_per_100_words > 0.0);
    // "you" -> second person hit
    assert!(profile.second_person_rate_per_100_words > 0.0);
    // "i think" -> hedge phrase hit
    assert!(profile.hedge_rate_per_100_words > 0.0);
}

#[test]
fn imperative_detection_on_a_bare_command_sentence() {
    let profile = compute_linguistic_profile("Check the deploy logs before pushing.");
    assert!(approx(profile.imperative_rate, 1.0, 0.001));
}

#[test]
fn certainty_and_contrast_connectives_are_counted() {
    let profile = compute_linguistic_profile("This is definitely broken, but the fix is simple.");
    assert!(profile.certainty_rate_per_100_words > 0.0);
    assert!(profile.contrast_connective_rate_per_100_words > 0.0);
}

#[test]
fn empty_text_does_not_panic_and_yields_zeroed_rates() {
    let profile = compute_linguistic_profile("");
    assert_eq!(profile.type_token_ratio, 0.0);
    assert_eq!(profile.comma_rate_per_100_words, 0.0);
}
