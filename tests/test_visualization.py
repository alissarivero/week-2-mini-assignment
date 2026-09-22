"""Data visualization."""

from pathlib import Path

import pandas as pd

from analysis import plot_prior_crime, plot_themes
from visuals import (
    plot_theme_heatmap,
    plot_theme_profile,
    plot_wordcloud_all,
    plot_wordclouds_by_prior,
    plot_wordclouds_by_theme,
)


def test_plot_prior_crime_writes_png(tmp_path):
    model_df = pd.DataFrame(
        {
            "group": ["No prior", "No prior", "Prior", "Prior"],
            "apology_rate": [1.0, 1.5, 0.8, 2.0],
            "religion_rate": [2.0, 0.5, 3.0, 1.0],
        }
    )
    out = tmp_path / "prior.png"
    written = plot_prior_crime(model_df, out)
    assert Path(written).exists()
    assert written.stat().st_size > 1000
    assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_themes_writes_png(tmp_path):
    demo = pd.DataFrame(
        {
            "Race": ["White", "Black", "Hispanic", "White", "Black", "Hispanic"],
            "remorse_rate": [1, 0.5, 1.2, 0.8, 0.4, 1.1],
            "gratitude_love_rate": [2, 3, 4, 2.5, 3.2, 4.1],
            "family_rate": [2, 2, 2, 1, 3, 2],
            "religion_rate": [1, 1, 2, 0.5, 0.8, 2.2],
        }
    )
    out = tmp_path / "themes.png"
    written = plot_themes(demo, out)
    assert Path(written).exists()
    assert written.stat().st_size > 1000
    assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _tiny_statements():
    return pd.DataFrame(
        {
            "LastStatement": [
                "I am sorry. I love my mom. Thank you God.",
                "Please forgive me. I love my kids and my wife.",
                "God bless you all. I pray. Amen.",
            ],
            "declined": [False, False, False],
            "group": ["No prior", "Prior", "Prior"],
            "Race": ["White", "Black", "Hispanic"],
            "remorse_rate": [2.0, 1.0, 0.5],
            "gratitude_love_rate": [1.5, 3.0, 2.0],
            "family_rate": [1.0, 3.0, 0.5],
            "religion_rate": [1.0, 0.5, 4.0],
        }
    )


def test_wordclouds_and_gallery_write_pngs(tmp_path):
    df = _tiny_statements()
    all_path = plot_wordcloud_all(df, tmp_path / "all.png")
    prior_path = plot_wordclouds_by_prior(df, tmp_path / "prior.png")
    theme_path = plot_wordclouds_by_theme(df, tmp_path / "themes.png")
    heat_path = plot_theme_heatmap(df, tmp_path / "heat.png")
    profile_path = plot_theme_profile(df, tmp_path / "profile.png")
    for path in (all_path, prior_path, theme_path, heat_path, profile_path):
        assert Path(path).exists()
        assert path.stat().st_size > 1000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
