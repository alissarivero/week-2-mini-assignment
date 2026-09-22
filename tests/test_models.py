"""Machine learning model training, prediction, and evaluation."""

import pandas as pd
import pytest

from analysis import (
    fit_prior_models,
    fit_theme_models,
    prepare_demographic_frame,
)


def _prior_frame(apology, religion, prior):
    return pd.DataFrame(
        {
            "apology_rate": apology,
            "religion_rate": religion,
            "prior_crime": prior,
            "group": ["No prior" if p == 0 else "Prior" for p in prior],
        }
    )


def test_fit_prior_models_recovers_known_shift():
    # No-prior apology = 1; prior apology = 3. Religion is flat at 2.
    model_df = _prior_frame(
        apology=[1, 1, 1, 1, 3, 3, 3, 3],
        religion=[2, 2, 2, 2, 2, 2, 2, 2],
        prior=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    models = fit_prior_models(model_df)
    assert set(models) == {"apology", "religion"}
    assert models["apology"].params["const"] == pytest.approx(1.0)
    assert models["apology"].params["prior_crime"] == pytest.approx(2.0)
    assert models["apology"].rsquared == pytest.approx(1.0)
    assert models["religion"].params["prior_crime"] == pytest.approx(0.0)
    pred = models["apology"].predict([1.0, 1.0])[0]
    assert pred == pytest.approx(3.0)


def test_fit_prior_models_reports_pvalue_and_nobs():
    model_df = _prior_frame(
        apology=[1.0, 1.2, 0.8, 2.9, 3.1, 2.7],
        religion=[4.0, 3.5, 4.2, 1.0, 0.8, 1.1],
        prior=[0, 0, 0, 1, 1, 1],
    )
    models = fit_prior_models(model_df)
    assert models["apology"].nobs == 6
    assert 0.0 <= models["apology"].pvalues["prior_crime"] <= 1.0
    assert models["religion"].params["prior_crime"] < 0


def test_fit_theme_models_includes_race_dummies():
    demo = prepare_demographic_frame(
        pd.DataFrame(
            {
                "prior_crime": [0, 1, 0, 1, 0, 1],
                "Age": [30, 40, 35, 45, 32, 50],
                "EducationLevel": [10, 12, 11, 9, 8, 12],
                "Race": ["White", "Black", "Hispanic", "White", "Black", "Hispanic"],
                "remorse_rate": [2, 1, 1.5, 2.2, 0.8, 1.4],
                "gratitude_love_rate": [3, 4, 5, 3, 4, 6],
                "family_rate": [2, 2, 2, 2, 2, 2],
                "religion_rate": [1, 1, 3, 1, 1, 3],
            }
        )
    )
    models = fit_theme_models(demo)
    assert set(models) == {
        "remorse_rate",
        "gratitude_love_rate",
        "family_rate",
        "religion_rate",
    }
    names = set(models["remorse_rate"].params.index)
    assert "prior_crime" in names
    assert "Age" in names
    assert "EducationLevel" in names
    assert "Race_Black" in names
    assert "Race_Hispanic" in names
    family_pred = models["family_rate"].predict(models["family_rate"].model.exog)
    assert family_pred == pytest.approx([2.0] * len(demo), abs=1e-8)
    pred = models["religion_rate"].predict(models["religion_rate"].model.exog)
    assert len(pred) == len(demo)


def test_prepare_demographic_frame_drops_other_race_and_missing():
    raw = pd.DataFrame(
        {
            "prior_crime": [0, 1, 0, 1],
            "Age": [30, None, 35, 40],
            "EducationLevel": [10, 12, 11, 9],
            "Race": ["White", "Black", "Other", "Hispanic"],
            "remorse_rate": [1, 1, 1, 1],
            "gratitude_love_rate": [1, 1, 1, 1],
            "family_rate": [1, 1, 1, 1],
            "religion_rate": [1, 1, 1, 1],
        }
    )
    demo = prepare_demographic_frame(raw)
    assert len(demo) == 2
    assert set(demo["Race"]) == {"White", "Hispanic"}
