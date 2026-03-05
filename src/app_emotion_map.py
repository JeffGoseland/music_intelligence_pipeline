"""
Emotion Map — Shiny for Python
Interactive 2D scatter plot of songs on the valence/arousal plane.

Requirements:
    pip install shiny shinywidgets plotly pandas

Run from your project root:
    shiny run app_emotion_map.py

Expects:
    data/processed/emotion_predictions.csv  — song_id, predicted_arousal, predicted_valence
    data/processed/song_features.csv        — optional, adds key / tempo_bpm / genre metadata
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

# ── Default paths (relative to project root) ─────────────────────────────────

DEFAULT_PREDICTIONS = "data/processed/emotion_predictions.csv"
DEFAULT_FEATURES = "data/processed/song_features.csv"

# ── Colour palette ────────────────────────────────────────────────────────────

COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load(pred_path: str, feat_path: str) -> pd.DataFrame:
    """Load predictions and optionally merge song feature metadata."""
    df = pd.read_csv(pred_path)
    feat = Path(feat_path)
    if feat.exists():
        sf = pd.read_csv(feat)
        extra = [c for c in ("key", "tempo_bpm", "genre") if c in sf.columns]
        if extra:
            df = df.merge(sf[["song_id"] + extra], on="song_id", how="left")
    return df


def _tempo_bucket(bpm: float) -> str:
    if pd.isna(bpm):
        return "Unknown"
    if bpm < 80:
        return "Slow  (<80 BPM)"
    if bpm < 120:
        return "Mid   (80–120 BPM)"
    if bpm < 160:
        return "Fast  (120–160 BPM)"
    return "Very Fast  (160+ BPM)"


# ── UI ────────────────────────────────────────────────────────────────────────

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h5("Data Paths"),
        ui.input_text("pred_path", "Predictions CSV", value=DEFAULT_PREDICTIONS),
        ui.input_text("feat_path", "Features CSV (optional)", value=DEFAULT_FEATURES),
        ui.input_action_button(
            "load_btn", "Load / Reload", class_="btn-primary w-100 mb-3"
        ),
        ui.hr(),
        ui.h5("Filters"),
        ui.output_ui("arousal_slider_ui"),
        ui.output_ui("valence_slider_ui"),
        ui.hr(),
        ui.h5("Colour By"),
        ui.input_select(
            "color_by",
            None,
            choices={
                "none": "None",
                "key": "Musical Key",
                "tempo_bucket": "Tempo Bucket",
            },
        ),
        ui.hr(),
        ui.output_text("song_count"),
        width=300,
    ),
    ui.card(
        ui.card_header("🎵 Emotion Map — Valence vs Arousal"),
        output_widget("emotion_map"),
        full_screen=True,
    ),
    ui.card(
        ui.card_header("Song Table"),
        ui.output_data_frame("song_table"),
    ),
    title="Music Emotion Map",
    fillable=True,
)

# ── Server ────────────────────────────────────────────────────────────────────


def server(input, output, session):

    # ── Load data ──────────────────────────────────────────────────────────────

    @reactive.calc
    @reactive.event(input.load_btn, ignore_none=False)
    def raw_df() -> pd.DataFrame:
        try:
            df = _load(input.pred_path(), input.feat_path())
            if "tempo_bpm" in df.columns:
                df["tempo_bucket"] = df["tempo_bpm"].apply(_tempo_bucket)
            return df
        except Exception as e:
            print(f"[load error] {e}")
            return pd.DataFrame()

    # ── Dynamic sliders built from data ranges ─────────────────────────────────

    @output
    @render.ui
    def arousal_slider_ui():
        df = raw_df()
        if df.empty or "predicted_arousal" not in df.columns:
            return ui.p("Load data to enable filters.", class_="text-muted small")
        lo, hi = float(df["predicted_arousal"].min()), float(
            df["predicted_arousal"].max()
        )
        step = round((hi - lo) / 100, 3)
        return ui.input_slider(
            "arousal_range",
            "Arousal",
            min=round(lo, 2),
            max=round(hi, 2),
            value=[round(lo, 2), round(hi, 2)],
            step=step,
        )

    @output
    @render.ui
    def valence_slider_ui():
        df = raw_df()
        if df.empty or "predicted_valence" not in df.columns:
            return ui.p("")
        lo, hi = float(df["predicted_valence"].min()), float(
            df["predicted_valence"].max()
        )
        step = round((hi - lo) / 100, 3)
        return ui.input_slider(
            "valence_range",
            "Valence",
            min=round(lo, 2),
            max=round(hi, 2),
            value=[round(lo, 2), round(hi, 2)],
            step=step,
        )

    # ── Filtered data ──────────────────────────────────────────────────────────

    @reactive.calc
    def filtered_df() -> pd.DataFrame:
        df = raw_df()
        if df.empty:
            return df
        try:
            ar = input.arousal_range()
            vr = input.valence_range()
            mask = df["predicted_arousal"].between(ar[0], ar[1]) & df[
                "predicted_valence"
            ].between(vr[0], vr[1])
            return df[mask].reset_index(drop=True)
        except Exception:
            return df

    # ── Song count ─────────────────────────────────────────────────────────────

    @output
    @render.text
    def song_count():
        df_all = raw_df()
        df_filt = filtered_df()
        if df_all.empty:
            return "No data loaded."
        return f"Showing {len(df_filt):,} of {len(df_all):,} songs"

    # ── Emotion map ────────────────────────────────────────────────────────────

    @output
    @render_plotly
    def emotion_map():
        df = filtered_df()

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data — enter paths above and click Load / Reload",
                showarrow=False,
                font=dict(size=15, color="gray"),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            return fig

        x_col, y_col = "predicted_valence", "predicted_arousal"

        x_lo, x_hi = df[x_col].min(), df[x_col].max()
        y_lo, y_hi = df[y_col].min(), df[y_col].max()
        x_pad = (x_hi - x_lo) * 0.05 or 0.3
        y_pad = (y_hi - y_lo) * 0.05 or 0.3
        x_range = [x_lo - x_pad, x_hi + x_pad]
        y_range = [y_lo - y_pad, y_hi + y_pad]
        x_mid = (x_lo + x_hi) / 2
        y_mid = (y_lo + y_hi) / 2

        fig = go.Figure()

        # Quadrant shading + labels
        quadrants = [
            (
                x_range[0],
                x_mid,
                y_mid,
                y_range[1],
                "Tense / Angry",
                "rgba(220,60,60,0.07)",
                "left",
                "top",
            ),
            (
                x_mid,
                x_range[1],
                y_mid,
                y_range[1],
                "Happy / Euphoric",
                "rgba(60,180,60,0.07)",
                "right",
                "top",
            ),
            (
                x_range[0],
                x_mid,
                y_range[0],
                y_mid,
                "Sad / Somber",
                "rgba(60,60,220,0.07)",
                "left",
                "bottom",
            ),
            (
                x_mid,
                x_range[1],
                y_range[0],
                y_mid,
                "Calm / Peaceful",
                "rgba(180,160,30,0.07)",
                "right",
                "bottom",
            ),
        ]
        label_x = {"left": x_range[0] + x_pad * 0.4, "right": x_range[1] - x_pad * 0.4}
        label_y = {"top": y_range[1] - y_pad * 0.4, "bottom": y_range[0] + y_pad * 0.4}

        for x0, x1, y0, y1, label, color, xanchor, yanchor in quadrants:
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=color,
                line_width=0,
                layer="below",
            )
            fig.add_annotation(
                x=label_x[xanchor],
                y=label_y[yanchor],
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=11, color="rgba(80,80,80,0.8)"),
                xanchor=xanchor,
                yanchor=yanchor,
            )

        # Centre dividers
        fig.add_hline(
            y=y_mid, line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1
        )
        fig.add_vline(
            x=x_mid, line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1
        )

        # Scatter — grouped by colour_by if selected
        color_by = input.color_by()
        use_color = color_by in ("key", "tempo_bucket") and color_by in df.columns

        def _hover(row, extra_col=None, extra_val=None):
            txt = f"<b>{row['song_id']}</b><br>Valence: {row[x_col]:.2f}<br>Arousal: {row[y_col]:.2f}"
            if extra_col:
                txt += f"<br>{extra_col.replace('_', ' ').title()}: {extra_val}"
            return txt

        if use_color:
            groups = sorted(df[color_by].fillna("Unknown").unique())
            for i, grp in enumerate(groups):
                sub = df[df[color_by].fillna("Unknown") == grp]
                hover = sub.apply(lambda r: _hover(r, color_by, grp), axis=1)
                fig.add_trace(
                    go.Scatter(
                        x=sub[x_col],
                        y=sub[y_col],
                        mode="markers",
                        name=str(grp),
                        marker=dict(
                            size=7, opacity=0.75, color=COLORS[i % len(COLORS)]
                        ),
                        text=hover,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
        else:
            hover = df.apply(_hover, axis=1)
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    name="Songs",
                    marker=dict(
                        size=7,
                        opacity=0.65,
                        color="#636EFA",
                        line=dict(width=0.5, color="white"),
                    ),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )

        fig.update_layout(
            xaxis=dict(
                title="Valence  (negative ◄──────► positive)",
                range=x_range,
                zeroline=False,
                showgrid=False,
            ),
            yaxis=dict(
                title="Arousal  (calm ◄──────► energetic)",
                range=y_range,
                zeroline=False,
                showgrid=False,
            ),
            legend=dict(
                title=dict(
                    text=color_by.replace("_", " ").title() if use_color else ""
                ),
                itemsizing="constant",
            ),
            margin=dict(l=70, r=20, t=20, b=60),
            hovermode="closest",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    # ── Song table ─────────────────────────────────────────────────────────────

    @output
    @render.data_frame
    def song_table():
        df = filtered_df()
        if df.empty:
            return render.DataGrid(pd.DataFrame())
        display = ["song_id", "predicted_arousal", "predicted_valence"]
        for col in ("key", "tempo_bpm", "genre", "tempo_bucket"):
            if col in df.columns:
                display.append(col)
        out = df[display].copy()
        out["predicted_arousal"] = out["predicted_arousal"].round(3)
        out["predicted_valence"] = out["predicted_valence"].round(3)
        return render.DataGrid(out, filters=True, height="280px")


# ── Entry point ───────────────────────────────────────────────────────────────

app = App(app_ui, server)
