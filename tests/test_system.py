"""End-to-end system test of the full analysis pipeline."""

from analysis import DATA_PATH, run_pipeline


def test_full_pipeline_from_csv_to_models_and_plots(tmp_path):
    result = run_pipeline(data_path=DATA_PATH, plot_dir=tmp_path)

    assert len(result["df"]) == 545
    assert "apology_rate" in result["df"].columns
    assert result["df"]["declined"].sum() == 114

    labeled = result["labeled"]
    assert len(labeled) == 509
    assert set(labeled["group"]) == {"No prior", "Prior"}
    assert (labeled["PreviousCrime"].isin([0.0, 1.0])).all()

    apology = result["prior_models"]["apology"]
    religion = result["prior_models"]["religion"]
    assert apology.nobs == 509
    assert religion.nobs == 509
    assert 0.0 <= apology.pvalues["prior_crime"] <= 1.0
    assert "prior_crime" in religion.params.index

    theme_models = result["theme_models"]
    assert len(result["demo"]) == 479
    assert set(theme_models) == {
        "remorse_rate",
        "gratitude_love_rate",
        "family_rate",
        "religion_rate",
    }
    hispanic = theme_models["gratitude_love_rate"].params["Race_Hispanic"]
    assert hispanic > 0

    assert result["prior_plot"].exists()
    assert result["theme_plot"].exists()
    assert result["prior_plot"].stat().st_size > 1000
    assert result["theme_plot"].stat().st_size > 1000
