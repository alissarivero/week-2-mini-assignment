"""Data preprocessing and transformation."""

import pandas as pd

from analysis import coerce_numeric, is_declined, label_prior_groups


def test_is_declined_blank_none_and_missing():
    assert is_declined(None) is True
    assert is_declined(float("nan")) is True
    assert is_declined("") is True
    assert is_declined("None") is True
    assert is_declined("  declined to make a last statement ") is True
    assert is_declined("This offender declined to make a last statement.") is True
    assert is_declined("No last statement") is True


def test_is_declined_real_statement_is_false():
    assert is_declined("I am sorry. I love my family. God bless.") is False
    assert is_declined("I just want to tell my family thank you.") is False


def test_coerce_numeric_turns_na_strings_to_nan():
    raw = pd.DataFrame(
        {
            "Age": ["39", "NA", ""],
            "PreviousCrime": ["0", "1", "NA"],
            "EducationLevel": ["12", "bad", "10"],
        }
    )
    out = coerce_numeric(raw)
    assert out["Age"].tolist()[0] == 39.0
    assert pd.isna(out["Age"].iloc[1])
    assert pd.isna(out["Age"].iloc[2])
    assert out["PreviousCrime"].tolist()[:2] == [0.0, 1.0]
    assert pd.isna(out["PreviousCrime"].iloc[2])
    assert pd.isna(out["EducationLevel"].iloc[1])


def test_label_prior_groups_drops_missing_and_unknown():
    df = pd.DataFrame(
        {
            "PreviousCrime": [0.0, 1.0, None, 3.0],
            "apology_rate": [1.0, 2.0, 3.0, 4.0],
        }
    )
    labeled = label_prior_groups(df)
    assert len(labeled) == 2
    assert set(labeled["group"]) == {"No prior", "Prior"}
    assert labeled.loc[labeled["PreviousCrime"] == 0, "group"].iloc[0] == "No prior"
    assert labeled.loc[labeled["PreviousCrime"] == 1, "group"].iloc[0] == "Prior"
