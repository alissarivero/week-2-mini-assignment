"""Feature engineering: theme hits and rates per 100 words."""

import pandas as pd

from analysis import (
    FAMILY_EXACT,
    FAMILY_PREFIXES,
    GRATITUDE_LOVE_EXACT,
    GRATITUDE_LOVE_PREFIXES,
    RELIGION_EXACT,
    RELIGION_PREFIXES,
    REMORSE_EXACT,
    REMORSE_PREFIXES,
    add_text_scores,
    build_model_frame,
    label_prior_groups,
    spoken_long_enough,
    theme_hits,
    tokenize,
)


def test_tokenize_splits_words_and_keeps_apostrophes():
    assert tokenize("I love y'all. I'm sorry!") == [
        "i",
        "love",
        "y'all",
        "i'm",
        "sorry",
    ]


def test_theme_hits_counts_remorse_not_ask_or_tell():
    words = tokenize("I am sorry. Please forgive me. I ask and I tell the truth.")
    hits = theme_hits(words, REMORSE_EXACT, REMORSE_PREFIXES)
    assert hits == 2
    assert "ask" not in REMORSE_EXACT
    assert "tell" not in REMORSE_EXACT
    assert "ask" not in GRATITUDE_LOVE_EXACT
    assert "tell" not in FAMILY_EXACT


def test_theme_hits_does_not_count_person_or_goodbye():
    words = tokenize("This person said goodbye to the landlord.")
    assert theme_hits(words, FAMILY_EXACT, FAMILY_PREFIXES) == 0
    assert theme_hits(words, RELIGION_EXACT, RELIGION_PREFIXES) == 0


def test_family_and_gratitude_are_separate():
    words = tokenize("I love my mom and my kids. Thank you all.")
    family = theme_hits(words, FAMILY_EXACT, FAMILY_PREFIXES)
    gratitude = theme_hits(words, GRATITUDE_LOVE_EXACT, GRATITUDE_LOVE_PREFIXES)
    assert family == 2
    assert gratitude == 2


def test_add_text_scores_rates_per_100_words():
    df = pd.DataFrame(
        {
            "LastStatement": [
                "sorry sorry sorry sorry sorry sorry sorry sorry sorry sorry",
            ]
        }
    )
    scored = add_text_scores(df)
    assert scored.loc[0, "word_count"] == 10
    assert scored.loc[0, "remorse_hits"] == 10
    assert scored.loc[0, "apology_rate"] == 100.0
    assert scored.loc[0, "remorse_rate"] == 100.0


def test_add_text_scores_declined_has_zero_words_and_rates():
    df = pd.DataFrame(
        {"LastStatement": [None, "This offender declined to make a last statement."]}
    )
    scored = add_text_scores(df)
    assert scored["declined"].all()
    assert list(scored["word_count"]) == [0, 0]
    assert list(scored["apology_rate"]) == [0.0, 0.0]
    assert list(scored["religion_rate"]) == [0.0, 0.0]


def test_add_text_scores_mixed_themes():
    text = "I am sorry. I love you mom. Thank you. God bless."
    scored = add_text_scores(pd.DataFrame({"LastStatement": [text]}))
    assert scored.loc[0, "remorse_hits"] >= 1
    assert scored.loc[0, "gratitude_love_hits"] >= 2
    assert scored.loc[0, "family_hits"] >= 1
    assert scored.loc[0, "religion_hits"] >= 2
    assert scored.loc[0, "apology_rate"] == scored.loc[0, "remorse_rate"]


def test_one_word_statement_scores_100_and_is_filtered():
    df = pd.DataFrame(
        {
            "LastStatement": ["sorry"],
            "PreviousCrime": [0.0],
            "Age": [40],
            "EducationLevel": [12],
            "Race": ["White"],
        }
    )
    scored = add_text_scores(df)
    assert scored.loc[0, "word_count"] == 1
    assert scored.loc[0, "apology_rate"] == 100.0
    kept = spoken_long_enough(build_model_frame(label_prior_groups(scored)))
    assert len(kept) == 0


def test_length_filter_drops_declined_and_keeps_long_statement():
    long_text = " ".join(["family"] * 25)
    df = pd.DataFrame(
        {
            "LastStatement": [
                "This offender declined to make a last statement.",
                long_text,
            ],
            "PreviousCrime": [0.0, 1.0],
            "Age": [40, 41],
            "EducationLevel": [12, 10],
            "Race": ["White", "Black"],
        }
    )
    scored = add_text_scores(df)
    kept = spoken_long_enough(build_model_frame(label_prior_groups(scored)))
    assert len(kept) == 1
    assert kept.iloc[0]["word_count"] == 25
    assert bool(kept.iloc[0]["declined"]) is False
    edge = spoken_long_enough(
        pd.DataFrame(
            {
                "word_count": [19, 20],
                "declined": [False, False],
            }
        ),
        min_words=20,
    )
    assert list(edge["word_count"]) == [20]
