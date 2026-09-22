"""Data loading."""

from pathlib import Path

import pandas as pd
import pytest

from analysis import DATA_PATH, load_statements

EXPECTED_COLUMNS = {
    "Execution",
    "LastName",
    "FirstName",
    "Age",
    "Race",
    "EducationLevel",
    "PreviousCrime",
    "LastStatement",
}


def test_load_statements_shape_and_columns():
    df = load_statements()
    assert len(df) == 545
    assert EXPECTED_COLUMNS.issubset(set(df.columns))
    assert "NativeCounty" in df.columns
    assert "NativeCounty " not in df.columns


def test_load_statements_latin1_does_not_raise():
    df = load_statements(DATA_PATH)
    assert df["LastStatement"].notna().sum() >= 400


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_statements(Path("/tmp/does-not-exist-texas-statements.csv"))


def test_load_statements_from_temp_copy(tmp_path):
    copy = tmp_path / "copy.csv"
    pd.read_csv(DATA_PATH, encoding="latin-1").to_csv(copy, index=False, encoding="latin-1")
    df = load_statements(copy)
    assert len(df) == 545
