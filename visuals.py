"""Gallery figures: word clouds, heatmap, and theme profiles."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

from analysis import (
    FAMILY_EXACT,
    FAMILY_PREFIXES,
    GRATITUDE_LOVE_EXACT,
    GRATITUDE_LOVE_PREFIXES,
    RELIGION_EXACT,
    RELIGION_PREFIXES,
    REMORSE_EXACT,
    REMORSE_PREFIXES,
    is_declined,
    theme_hits,
    tokenize,
)

BG = "#FBF9F6"
INK = "#2F2B26"
MUTED = "#5A554E"
RULE = "#C9C3B8"

THEME_COLORS = {
    "Remorse": "#5E7A32",
    "Gratitude and love": "#B56B86",
    "Family": "#3D6B8A",
    "Religion": "#8A6A2F",
}

EXTRA_STOP = {
    "ask",
    "asks",
    "asked",
    "asking",
    "tell",
    "tells",
    "told",
    "telling",
    "say",
    "says",
    "said",
    "saying",
    "just",
    "want",
    "know",
    "like",
    "yes",
    "sir",
    "warden",
    "okay",
    "yeah",
    "uh",
    "um",
    "gonna",
    "wanna",
    "ain't",
    "y'all",
    "that's",
    "don't",
    "i'm",
    "i've",
    "i'll",
    "it's",
    "none",
    "nan",
}

STOP = set(STOPWORDS) | EXTRA_STOP


def _spoken(df):
    out = df.copy()
    if "declined" in out.columns:
        return out.loc[~out["declined"]]
    mask = ~out["LastStatement"].map(is_declined)
    return out.loc[mask]


def _join_statements(frame):
    texts = _spoken(frame)["LastStatement"].dropna().astype(str)
    return " ".join(texts.tolist())


def _color_cycle(colors):
    palette = list(colors)

    def _func(word, font_size, position, orientation, random_state=None, **kwargs):
        return palette[abs(hash((word, font_size))) % len(palette)]

    return _func


def _wordcloud(text, color_func, width=1400, height=900):
    if not text.strip():
        text = "none"
    return WordCloud(
        width=width,
        height=height,
        background_color=BG,
        color_func=color_func,
        max_words=140,
        prefer_horizontal=0.82,
        min_font_size=9,
        collocations=False,
        stopwords=STOP,
        random_state=7,
        relative_scaling=0.45,
        margin=12,
    ).generate(text)


def _style_ax(ax, title, color=INK):
    ax.set_facecolor(BG)
    ax.set_title(title, color=color, pad=12, fontsize=14, fontweight="semibold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_wordcloud_all(df, out_path):
    """One large cloud of every spoken last statement."""
    text = _join_statements(df)
    cloud = _wordcloud(
        text,
        _color_cycle(["#5E7A32", "#3D6B8A", "#B56B86", "#8A6A2F", "#2F2B26", "#7A6A4A"]),
        width=1800,
        height=1000,
    )
    fig, ax = plt.subplots(figsize=(14, 7.6), facecolor=BG)
    ax.imshow(cloud, interpolation="bilinear")
    _style_ax(ax, "What people actually said")
    fig.suptitle(
        "Last words",
        y=1.02,
        fontsize=20,
        fontweight="semibold",
        color=INK,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def plot_wordclouds_by_prior(labeled, out_path):
    """Two clouds: no prior record vs prior record."""
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4), facecolor=BG)
    specs = [
        ("No prior", "#5E7A32", ["#D7E2B4", "#8AA05A", "#5E7A32", "#3F5320"]),
        ("Prior", "#7A3F56", ["#F3D6E0", "#C98AA3", "#B56B86", "#7A3F56"]),
    ]
    for ax, (group, edge, colors) in zip(axes, specs):
        text = _join_statements(labeled[labeled["group"] == group])
        cloud = _wordcloud(text, _color_cycle(colors), width=1100, height=900)
        ax.imshow(cloud, interpolation="bilinear")
        _style_ax(ax, group, color=edge)
    fig.suptitle(
        "Last words by prior criminal record",
        y=1.03,
        fontsize=18,
        fontweight="semibold",
        color=INK,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def _theme_frequencies(frame, exact, prefixes):
    counts = Counter()
    for text in _spoken(frame)["LastStatement"].dropna().astype(str):
        words = tokenize(text)
        for word in words:
            if theme_hits([word], exact, prefixes):
                counts[word] += 1
    return counts


def plot_wordclouds_by_theme(df, out_path):
    """Four clouds built only from remorse, gratitude/love, family, and religion words."""
    panels = [
        ("Remorse", REMORSE_EXACT, REMORSE_PREFIXES, ["#D7E2B4", "#8AA05A", "#5E7A32", "#3F5320"]),
        (
            "Gratitude and love",
            GRATITUDE_LOVE_EXACT,
            GRATITUDE_LOVE_PREFIXES,
            ["#F3D6E0", "#C98AA3", "#B56B86", "#7A3F56"],
        ),
        ("Family", FAMILY_EXACT, FAMILY_PREFIXES, ["#C5D7E4", "#6F93AE", "#3D6B8A", "#24485C"]),
        ("Religion", RELIGION_EXACT, RELIGION_PREFIXES, ["#E6D3A8", "#C4A45C", "#8A6A2F", "#5A4318"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4), facecolor=BG)
    for ax, (title, exact, prefixes, colors) in zip(axes.ravel(), panels):
        freqs = _theme_frequencies(df, exact, prefixes)
        if not freqs:
            freqs = Counter({"none": 1})
        cloud = WordCloud(
            width=1000,
            height=800,
            background_color=BG,
            color_func=_color_cycle(colors),
            max_words=80,
            prefer_horizontal=0.8,
            min_font_size=10,
            collocations=False,
            random_state=7,
            relative_scaling=0.5,
            margin=10,
        ).generate_from_frequencies(freqs)
        ax.imshow(cloud, interpolation="bilinear")
        _style_ax(ax, title, color=THEME_COLORS[title])
    fig.suptitle(
        "The four last-word themes",
        y=1.02,
        fontsize=18,
        fontweight="semibold",
        color=INK,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def plot_theme_heatmap(demo, out_path):
    """Mean theme rate (hits per 100 words) by race."""
    cols = {
        "remorse_rate": "Remorse",
        "gratitude_love_rate": "Gratitude & love",
        "family_rate": "Family",
        "religion_rate": "Religion",
    }
    means = (
        demo.groupby("Race", observed=True)[list(cols)]
        .mean()
        .rename(columns=cols)
        .reindex(["White", "Black", "Hispanic"])
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor=BG)
    ax.set_facecolor(BG)
    sns.heatmap(
        means,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        linewidths=3,
        linecolor=BG,
        cbar_kws={"label": "Hits per 100 words", "shrink": 0.8},
        annot_kws={"size": 11, "color": INK},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=MUTED)
    ax.set_title("Average theme rate by race", pad=14, color=INK, fontsize=15, fontweight="semibold")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def plot_theme_profile(demo, out_path):
    """Lollipop chart of theme means by race."""
    long = demo.melt(
        id_vars="Race",
        value_vars=["remorse_rate", "gratitude_love_rate", "family_rate", "religion_rate"],
        var_name="theme",
        value_name="rate",
    )
    long["theme"] = long["theme"].map(
        {
            "remorse_rate": "Remorse",
            "gratitude_love_rate": "Gratitude & love",
            "family_rate": "Family",
            "religion_rate": "Religion",
        }
    )
    summary = (
        long.groupby(["theme", "Race"], observed=True)["rate"]
        .mean()
        .reset_index()
    )
    theme_order = ["Remorse", "Gratitude & love", "Family", "Religion"]
    race_order = ["White", "Black", "Hispanic"]
    race_colors = {"White": "#C4B49A", "Black": "#5A554E", "Hispanic": "#3D6B8A"}

    fig, ax = plt.subplots(figsize=(10.8, 5.6), facecolor=BG)
    ax.set_facecolor(BG)
    y_base = np.arange(len(theme_order))
    offsets = {"White": -0.22, "Black": 0.0, "Hispanic": 0.22}
    for race in race_order:
        rows = summary[summary["Race"] == race].set_index("theme").reindex(theme_order)
        ys = y_base + offsets[race]
        ax.hlines(ys, 0, rows["rate"], color=race_colors[race], linewidth=2.2, alpha=0.85)
        ax.scatter(
            rows["rate"],
            ys,
            s=90,
            color=race_colors[race],
            edgecolors="white",
            linewidths=1.1,
            zorder=3,
            label=race,
        )
    ax.set_yticks(y_base)
    ax.set_yticklabels(theme_order)
    ax.set_xlabel("Mean hits per 100 words")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, title="")
    sns.despine(ax=ax)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.xaxis.grid(True, color="#E6E1D8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Theme profile by race",
        pad=14,
        color=INK,
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


def save_visuals(df, labeled, demo, plot_dir):
    """Write the extra gallery figures. Returns a dict of paths."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "wordcloud_all": plot_wordcloud_all(df, plot_dir / "wordcloud_all.png"),
        "wordclouds_prior": plot_wordclouds_by_prior(labeled, plot_dir / "wordclouds_prior.png"),
        "wordclouds_themes": plot_wordclouds_by_theme(df, plot_dir / "wordclouds_themes.png"),
        "theme_heatmap": plot_theme_heatmap(demo, plot_dir / "theme_heatmap.png"),
        "theme_profile": plot_theme_profile(demo, plot_dir / "theme_profile.png"),
    }
    return paths
