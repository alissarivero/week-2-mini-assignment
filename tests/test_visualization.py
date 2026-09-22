"""Data visualization."""

from pathlib import Path

import pandas as pd

from analysis import plot_prior_crime, plot_themes


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
