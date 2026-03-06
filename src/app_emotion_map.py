"""
Emotion Map — Shiny for Python
Interactive 2D scatter plot of songs on the valence/arousal plane.

Requirements:
    pip install shiny shinywidgets plotly pandas

Run from your project root:
    shiny run src/app_emotion_map.py

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

# ── Colour palette (distinct, accessible) ────────────────────────────────────

COLORS = [
    "#5B8DEF",  # blue
    "#E85D75",  # rose
    "#34C77B",  # green
    "#9B59B6",  # violet
    "#F39C12",  # amber
    "#1ABC9C",  # teal
    "#E74C3C",  # red
    "#3498DB",  # sky
    "#E91E63",  # pink
    "#00BCD4",  # cyan
]

# Plot quadrant fills (soft, emotion-aligned)
QUAD_COLORS = {
    "tense": "rgba(231, 76, 60, 0.08)",  # red tint
    "happy": "rgba(46, 204, 113, 0.08)",  # green tint
    "sad": "rgba(52, 152, 219, 0.08)",  # blue tint
    "calm": "rgba(241, 196, 15, 0.06)",  # gold tint
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load(pred_path: str, feat_path: str) -> pd.DataFrame:
    """Load predictions and optionally merge song feature metadata."""
    df = pd.read_csv(pred_path)
    feat = Path(feat_path)
    if feat.exists():
        sf = pd.read_csv(feat)
        extra = [
            c
            for c in (
                "key",
                "tempo_bpm",
                "genre",
                "key_note",
                "key_mode",
                "key_signature",
                "is_major",
            )
            if c in sf.columns
        ]
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


# ── Custom styles ─────────────────────────────────────────────────────────────

APP_FONT = "Plus Jakarta Sans"
HEADER_BG = "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"
SIDEBAR_BG = "#f8fafc"
CARD_SHADOW = "0 4px 20px rgba(0,0,0,0.06)"
PLOT_BG = "#f1f5f9"
PAPER_BG = "#ffffff"

custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
* {{ font-family: '{APP_FONT}', system-ui, sans-serif; }}
.app-header {{ background: {HEADER_BG}; color: white; padding: 1rem 1.5rem; margin: -1rem -1rem 1rem -1rem; border-radius: 0 0 12px 12px; }}
.app-header h1 {{ margin: 0; font-weight: 700; font-size: 1.5rem; letter-spacing: -0.02em; }}
.app-header p {{ margin: 0.25rem 0 0 0; opacity: 0.9; font-size: 0.9rem; font-weight: 400; }}
.bslib-sidebar {{ background: {SIDEBAR_BG}; border-right: 1px solid #e2e8f0; }}
.bslib-sidebar .bslib-sidebar-layout > div {{ padding: 1rem; }}
.section-label {{ font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.5rem; }}
.card {{ border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: {CARD_SHADOW}; overflow: hidden; }}
.card-header {{ font-weight: 600; font-size: 1.05rem; padding: 1rem 1.25rem; border-bottom: 1px solid #e2e8f0; background: #fafbfc; }}
.btn-primary {{ background: linear-gradient(135deg, #5B8DEF 0%, #4a7bd8 100%); border: none; font-weight: 500; }}
.song-count {{ font-size: 0.85rem; color: #64748b; padding: 0.5rem 0; }}
"""

# ── UI ────────────────────────────────────────────────────────────────────────

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.div({"class": "section-label"}, "Data"),
        ui.input_text("pred_path", "Predictions CSV", value=DEFAULT_PREDICTIONS),
        ui.input_text("feat_path", "Features CSV (optional)", value=DEFAULT_FEATURES),
        ui.input_action_button(
            "load_btn", "Load / Reload", class_="btn-primary w-100 mb-3"
        ),
        ui.hr(),
        ui.div({"class": "section-label"}, "Filters"),
        ui.output_ui("arousal_slider_ui"),
        ui.output_ui("valence_slider_ui"),
        ui.hr(),
        ui.div({"class": "section-label"}, "Colour by"),
        ui.input_select(
            "color_by",
            None,
            choices={
                "none": "None",
                "key": "Key (full)",
                "key_note": "Key note (A–G)",
                "key_mode": "Key mode (major/minor)",
                "key_signature": "Key signature (♯/♭/natural)",
                "tempo_bucket": "Tempo Bucket",
            },
        ),
        ui.hr(),
        ui.output_text("song_count"),
        width=300,
    ),
    ui.tags.head(ui.tags.style(custom_css)),
    ui.div(
        {"class": "app-header"},
        ui.tags.h1("Music Emotion Map"),
        ui.tags.p("Valence × Arousal — explore your library by mood"),
    ),
    ui.card(
        ui.card_header("Emotion space"),
        output_widget("emotion_map"),
        full_screen=True,
    ),
    ui.card(
        ui.card_header("Songs in view"),
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
        lo, hi = (
            float(df["predicted_arousal"].min()),
            float(df["predicted_arousal"].max()),
        )
        step = round((hi - lo) / 100, 3)
        return ui.TagList(
            ui.input_slider(
                "arousal_range",
                "Arousal",
                min=round(lo, 2),
                max=round(hi, 2),
                value=[round(lo, 2), round(hi, 2)],
                step=step,
            ),
            ui.p(
                "Low = calm, sleepy, relaxed · High = energetic, tense, excited",
                class_="text-muted small mt-1 mb-2",
                style="font-size: 0.75rem; line-height: 1.3;",
            ),
        )

    @output
    @render.ui
    def valence_slider_ui():
        df = raw_df()
        if df.empty or "predicted_valence" not in df.columns:
            return ui.p("")
        lo, hi = (
            float(df["predicted_valence"].min()),
            float(df["predicted_valence"].max()),
        )
        step = round((hi - lo) / 100, 3)
        return ui.TagList(
            ui.input_slider(
                "valence_range",
                "Valence",
                min=round(lo, 2),
                max=round(hi, 2),
                value=[round(lo, 2), round(hi, 2)],
                step=step,
            ),
            ui.p(
                "Low = negative, sad, unpleasant · High = positive, happy, pleasant",
                class_="text-muted small mt-1 mb-2",
                style="font-size: 0.75rem; line-height: 1.3;",
            ),
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
                font=dict(size=15, color="#64748b", family=APP_FONT),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            fig.update_layout(
                plot_bgcolor=PLOT_BG,
                paper_bgcolor=PAPER_BG,
                font=dict(family=APP_FONT, size=12),
            )
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
                QUAD_COLORS["tense"],
                "left",
                "top",
            ),
            (
                x_mid,
                x_range[1],
                y_mid,
                y_range[1],
                "Happy / Euphoric",
                QUAD_COLORS["happy"],
                "right",
                "top",
            ),
            (
                x_range[0],
                x_mid,
                y_range[0],
                y_mid,
                "Sad / Somber",
                QUAD_COLORS["sad"],
                "left",
                "bottom",
            ),
            (
                x_mid,
                x_range[1],
                y_range[0],
                y_mid,
                "Calm / Peaceful",
                QUAD_COLORS["calm"],
                "right",
                "bottom",
            ),
        ]
        label_x = {"left": x_range[0] + x_pad * 0.4, "right": x_range[1] - x_pad * 0.4}
        label_y = {"top": y_range[1] - y_pad * 0.4, "bottom": y_range[0] + y_pad * 0.4}

        for x0, x1, y0, y1, label, color, xanchor, yanchor in quadrants:
            font_size = 15
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
                font=dict(size=font_size, color="#475569", family=APP_FONT),
                xanchor=xanchor,
                yanchor=yanchor,
            )

        # Centre dividers
        fig.add_hline(
            y=y_mid,
            line_dash="dot",
            line_color="rgba(148,163,184,0.6)",
            line_width=1,
        )
        fig.add_vline(
            x=x_mid,
            line_dash="dot",
            line_color="rgba(148,163,184,0.6)",
            line_width=1,
        )

        # Scatter — grouped by colour_by if selected
        color_by = input.color_by()
        color_options = (
            "key",
            "tempo_bucket",
            "key_note",
            "key_mode",
            "key_signature",
        )
        use_color = color_by in color_options and color_by in df.columns

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
                            size=8,
                            opacity=0.8,
                            color=COLORS[i % len(COLORS)],
                            line=dict(width=0.5, color="rgba(255,255,255,0.8)"),
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
                        size=8,
                        opacity=0.7,
                        color="#5B8DEF",
                        line=dict(width=0.5, color="rgba(255,255,255,0.9)"),
                    ),
                    text=hover,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )

        fig.update_layout(
            xaxis=dict(
                title=dict(
                    text="Valence  (negative ◄──────► positive)",
                    font=dict(size=13, family=APP_FONT),
                ),
                range=x_range,
                zeroline=False,
                showgrid=True,
                gridcolor="rgba(226,232,240,0.8)",
                gridwidth=1,
                tickfont=dict(size=11, family=APP_FONT),
            ),
            yaxis=dict(
                title=dict(
                    text="Arousal  (calm ◄──────► energetic)",
                    font=dict(size=13, family=APP_FONT),
                ),
                range=y_range,
                zeroline=False,
                showgrid=True,
                gridcolor="rgba(226,232,240,0.8)",
                gridwidth=1,
                tickfont=dict(size=11, family=APP_FONT),
            ),
            legend=dict(
                title=dict(
                    text=color_by.replace("_", " ").title() if use_color else "",
                    font=dict(family=APP_FONT),
                ),
                itemsizing="constant",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0",
            ),
            margin=dict(l=70, r=24, t=24, b=65),
            hovermode="closest",
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PAPER_BG,
            font=dict(family=APP_FONT, size=12),
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
        for col in (
            "key",
            "key_note",
            "key_mode",
            "key_signature",
            "is_major",
            "tempo_bpm",
            "genre",
            "tempo_bucket",
        ):
            if col in df.columns:
                display.append(col)
        out = df[display].copy()
        out["predicted_arousal"] = out["predicted_arousal"].round(3)
        out["predicted_valence"] = out["predicted_valence"].round(3)
        return render.DataGrid(out, filters=True, height="280px")


# ── Entry point ───────────────────────────────────────────────────────────────

app = App(app_ui, server)
