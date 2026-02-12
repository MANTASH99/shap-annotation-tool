"""
SHAP Annotation Tool — Streamlit Cloud Edition
================================================
- Highlighted sentence text colored by SHAP values
- Per-feature annotation (mark individual words/phrases as Correct / Wrong)
- Per-annotator sample assignment (no overlap)
- Free navigation: go back to any sample, edit previous annotations
- Persistence: Google Sheets on cloud, local CSV when running locally
"""

import streamlit as st
import json
import csv
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NORMAL_JSON = DATA_DIR / "normal_shap_values.json"
RELATIONAL_JSON = DATA_DIR / "relational_shap_values.json"
RELATIONS_JSON = DATA_DIR / "relation_extraction_all.json"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# --- Annotator assignment: map name -> (start_id, end_id) ---
ANNOTATOR_ASSIGNMENTS = {
    "Benni": (1, 716),
    "Emilia": (717, 1432),
    "Vanessa": (1433, 2148),
    "Anna": (2149, 2861),
}

FEATURE_LABELS = ["—", "Correct", "Wrong"]


# ---------------------------------------------------------------------------
# Google Sheets helpers
# ---------------------------------------------------------------------------
def get_gsheet_client():
    if not GSPREAD_AVAILABLE:
        return None
    creds_dict = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


def get_or_create_worksheet(client, sheet_name, worksheet_title):
    sh = client.open(sheet_name)
    try:
        ws = sh.worksheet(worksheet_title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_title, rows=3000, cols=20)
        ws.append_row([
            "sample_id", "annotator", "sentence", "true_label",
            "predicted_label", "confidence",
            "normal_shap_label", "relational_shap_label",
            "normal_feature_annotations", "relational_feature_annotations",
            "normal_shap_values", "relational_shap_values",
            "comment", "timestamp",
        ])
    return ws


def load_annotations_from_sheet(ws, annotator_name):
    records = ws.get_all_records()
    out = {}
    for r in records:
        if str(r["annotator"]) == annotator_name:
            sid = int(r["sample_id"])
            try:
                nfa = json.loads(r.get("normal_feature_annotations", "{}"))
            except (json.JSONDecodeError, TypeError):
                nfa = {}
            try:
                rfa = json.loads(r.get("relational_feature_annotations", "{}"))
            except (json.JSONDecodeError, TypeError):
                rfa = {}
            out[sid] = {
                **r,
                "normal_feature_annotations": nfa,
                "relational_feature_annotations": rfa,
            }
    return out


def save_annotation_to_sheet(ws, row_data):
    sample_id = str(row_data["sample_id"])
    annotator = row_data["annotator"]
    cell_list = ws.findall(sample_id, in_column=1)
    row_values = [
        row_data["sample_id"],
        row_data["annotator"],
        row_data["sentence"],
        row_data["true_label"],
        row_data["predicted_label"],
        row_data["confidence"],
        row_data["normal_shap_label"],
        row_data["relational_shap_label"],
        json.dumps(row_data["normal_feature_annotations"]),
        json.dumps(row_data["relational_feature_annotations"]),
        json.dumps(row_data["normal_shap_values"]),
        json.dumps(row_data["relational_shap_values"]),
        row_data["comment"],
        row_data["timestamp"],
    ]
    for cell in cell_list:
        existing = ws.row_values(cell.row)
        if len(existing) >= 2 and existing[1] == annotator:
            ws.update(f"A{cell.row}:N{cell.row}", [row_values])
            return
    ws.append_row(row_values)


# ---------------------------------------------------------------------------
# Local CSV saving (works when running locally)
# ---------------------------------------------------------------------------
def save_annotation_to_csv(annotator_name, row_data):
    csv_path = OUTPUT_DIR / f"{annotator_name}_annotations.csv"
    file_exists = csv_path.exists()

    fieldnames = [
        "sample_id", "annotator", "sentence", "true_label",
        "predicted_label", "confidence",
        "normal_shap_label", "relational_shap_label",
        "normal_feature_annotations", "relational_feature_annotations",
        "normal_shap_values", "relational_shap_values",
        "comment", "timestamp",
    ]

    rows = {}
    if file_exists:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows[int(r["sample_id"])] = r

    rows[row_data["sample_id"]] = {
        **row_data,
        "normal_feature_annotations": json.dumps(row_data["normal_feature_annotations"]),
        "relational_feature_annotations": json.dumps(row_data["relational_feature_annotations"]),
        "normal_shap_values": json.dumps(row_data["normal_shap_values"]),
        "relational_shap_values": json.dumps(row_data["relational_shap_values"]),
    }

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sid in sorted(rows.keys()):
            writer.writerow(rows[sid])


def load_annotations_from_csv(annotator_name):
    csv_path = OUTPUT_DIR / f"{annotator_name}_annotations.csv"
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = int(r["sample_id"])
            try:
                nfa = json.loads(r.get("normal_feature_annotations", "{}"))
            except (json.JSONDecodeError, TypeError):
                nfa = {}
            try:
                rfa = json.loads(r.get("relational_feature_annotations", "{}"))
            except (json.JSONDecodeError, TypeError):
                rfa = {}
            out[sid] = {**r, "normal_feature_annotations": nfa, "relational_feature_annotations": rfa}
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_normal_shap():
    with open(NORMAL_JSON) as f:
        return {item["id"]: item for item in json.load(f)}


@st.cache_data
def load_relational_shap():
    with open(RELATIONAL_JSON) as f:
        return {item["id"]: item for item in json.load(f)}


@st.cache_data
def load_relation_data():
    with open(RELATIONS_JSON) as f:
        return {item["id"]: item for item in json.load(f)}


# ---------------------------------------------------------------------------
# SHAP visualization
# ---------------------------------------------------------------------------
def shap_to_color(value, max_abs):
    if max_abs == 0:
        return "rgba(200, 200, 200, 0.15)"
    intensity = min(abs(value) / max_abs, 1.0)
    alpha = 0.1 + intensity * 0.7
    if value > 0:
        return f"rgba(231, 76, 60, {alpha:.2f})"
    elif value < 0:
        return f"rgba(52, 152, 219, {alpha:.2f})"
    return "rgba(200, 200, 200, 0.15)"


def render_highlighted_text(features, title):
    if not features:
        return "<p>No SHAP data</p>"
    max_abs = max(abs(v) for v in features.values()) if features else 1.0
    html = [
        f'<div style="margin-bottom:8px;font-size:14px;color:#666;"><b>{title}</b></div>',
        '<div style="line-height:2.4;font-size:16px;">',
    ]
    for token, value in features.items():
        bg = shap_to_color(value, max_abs)
        sign = "+" if value >= 0 else ""
        tooltip = f"{token}: {sign}{value:.4f}"
        html.append(
            f'<span title="{tooltip}" style="'
            f"background:{bg};padding:4px 6px;margin:2px 3px;"
            f"border-radius:5px;display:inline-block;"
            f"border:1px solid rgba(0,0,0,0.08);"
            f"cursor:default;font-family:'Source Sans Pro',sans-serif;"
            f'">{token}</span> '
        )
    html.append("</div>")
    return "".join(html)


def render_shap_bar(features, title):
    sorted_feats = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    sorted_feats = sorted_feats[::-1]
    labels = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="SHAP value",
        yaxis=dict(tickfont=dict(size=12)),
        height=300,
        margin=dict(l=10, r=60, t=35, b=35),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee", zeroline=True, zerolinecolor="#ccc")
    return fig


def render_feature_annotation(features, shap_type, sample_id, existing_fa):
    """Per-feature annotation controls. Returns {feature: "Correct"/"Wrong"}."""
    sorted_feats = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    nonzero_feats = [(f, v) for f, v in sorted_feats if abs(v) > 0.001]

    if not nonzero_feats:
        st.write("No significant features.")
        return {}

    annotations = {}
    max_abs = max(abs(v) for _, v in nonzero_feats)

    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:4px;">'
        'Mark each feature (leave &quot;—&quot; to skip):</div>',
        unsafe_allow_html=True,
    )

    for feat, val in nonzero_feats:
        bg = shap_to_color(val, max_abs)
        sign = "+" if val >= 0 else ""

        col_chip, col_toggle = st.columns([2, 1])
        with col_chip:
            st.markdown(
                f'<div style="padding:6px 0;">'
                f'<span style="background:{bg};padding:4px 8px;border-radius:5px;'
                f'font-size:15px;border:1px solid rgba(0,0,0,0.08);">'
                f'{feat}</span>'
                f'<span style="color:#888;font-size:13px;margin-left:8px;">'
                f'{sign}{val:.4f}</span></div>',
                unsafe_allow_html=True,
            )
        with col_toggle:
            existing_val = existing_fa.get(feat, "—")
            idx = FEATURE_LABELS.index(existing_val) if existing_val in FEATURE_LABELS else 0
            choice = st.selectbox(
                feat, FEATURE_LABELS, index=idx,
                key=f"{shap_type}_{sample_id}_{feat}",
                label_visibility="collapsed",
            )
            if choice != "—":
                annotations[feat] = choice

    return annotations


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------
def go_to(idx):
    """Set the current sample index in session state."""
    st.session_state.current_idx = idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SHAP Annotation Tool", layout="wide")

    st.title("SHAP Annotation Tool")
    st.caption("Annotate whether emotion model explanations highlight the correct reason.")

    # --- Login ---
    st.sidebar.header("Annotator Login")
    annotator_name = st.sidebar.selectbox(
        "Select your ID",
        options=[""] + list(ANNOTATOR_ASSIGNMENTS.keys()),
        format_func=lambda x: "Choose..." if x == "" else x,
    )

    if not annotator_name:
        st.info("Select your annotator ID in the sidebar to begin.")
        st.markdown("### Annotator Assignments")
        for name, (start, end) in ANNOTATOR_ASSIGNMENTS.items():
            st.write(f"**{name}**: samples {start} - {end} ({end - start + 1} samples)")
        return

    start_id, end_id = ANNOTATOR_ASSIGNMENTS[annotator_name]
    st.sidebar.success(f"Your samples: **{start_id}** to **{end_id}**")

    # --- Storage connection ---
    sheets_connected = False
    ws = None
    try:
        gc = get_gsheet_client()
        if gc:
            sheet_name = st.secrets.get("sheet_name", "SHAP_Annotations")
            ws = get_or_create_worksheet(gc, sheet_name, "annotations")
            sheets_connected = True
    except Exception as e:
        st.sidebar.error(f"Google Sheets error: {e}")

    if sheets_connected:
        st.sidebar.success("Storage: Google Sheets")
        save_info = (
            "Your annotations are **saved to Google Sheets** each time you click "
            "**Save**. You can close the browser anytime — when you come back, "
            "all your work will be loaded automatically. No export needed."
        )
    else:
        st.sidebar.info("Storage: Local CSV")
        save_info = (
            "Your annotations are **saved to a local CSV file** each time you click "
            "**Save**. You can close and reopen — your progress is preserved on this machine."
        )

    # --- Auto-save info ---
    st.info(save_info, icon="💾")

    # --- Legend ---
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:10px;">'
        '<span style="background:rgba(231,76,60,0.5);padding:2px 8px;border-radius:3px;">Red</span> '
        '= pushes <b>toward</b> predicted emotion &nbsp;&nbsp; '
        '<span style="background:rgba(52,152,219,0.5);padding:2px 8px;border-radius:3px;">Blue</span> '
        '= pushes <b>against</b> &nbsp;&nbsp; '
        'Hover to see SHAP values. Mark individual features below each chart.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Load data ---
    normal_data = load_normal_shap()
    relational_data = load_relational_shap()
    relation_data = load_relation_data()

    my_ids = sorted([sid for sid in normal_data if start_id <= sid <= end_id])
    total = len(my_ids)

    # --- Load existing annotations ---
    if "annotations" not in st.session_state or st.session_state.get("_annotator") != annotator_name:
        if sheets_connected:
            with st.spinner("Loading your previous annotations..."):
                st.session_state.annotations = load_annotations_from_sheet(ws, annotator_name)
        else:
            st.session_state.annotations = load_annotations_from_csv(annotator_name)
        st.session_state._annotator = annotator_name
        # Start at first unannotated sample
        first_unannotated = 0
        for i, sid in enumerate(my_ids):
            if sid not in st.session_state.annotations:
                first_unannotated = i
                break
        st.session_state.current_idx = first_unannotated

    annotations = st.session_state.annotations
    annotated_count = sum(1 for sid in my_ids if sid in annotations)

    # --- Sidebar: progress ---
    st.sidebar.header("Progress")
    st.sidebar.progress(annotated_count / total if total else 0)
    st.sidebar.write(f"**{annotated_count} / {total}** annotated")

    # --- Sidebar: navigation mode ---
    st.sidebar.header("Navigation")
    view_mode = st.sidebar.radio(
        "View mode",
        ["All samples", "Unannotated only", "Annotated only"],
    )

    if view_mode == "Unannotated only":
        display_ids = [sid for sid in my_ids if sid not in annotations]
    elif view_mode == "Annotated only":
        display_ids = [sid for sid in my_ids if sid in annotations]
    else:
        display_ids = my_ids

    if not display_ids:
        if view_mode == "Unannotated only":
            st.success("All your samples are annotated! You're done.")
        else:
            st.info("No annotated samples yet.")
        return

    # --- Sidebar: sample index control ---
    # Clamp current_idx to display range
    current_idx = st.session_state.get("current_idx", 0)
    if current_idx >= len(display_ids):
        current_idx = len(display_ids) - 1
    if current_idx < 0:
        current_idx = 0

    idx_input = st.sidebar.number_input(
        f"Sample position (1-{len(display_ids)})",
        min_value=1, max_value=len(display_ids),
        value=current_idx + 1, step=1,
    )
    current_idx = idx_input - 1
    sample_id = display_ids[current_idx]

    # --- Sidebar: jump to sample ID ---
    jump_id = st.sidebar.number_input(
        "Jump to sample ID",
        min_value=start_id, max_value=end_id,
        value=sample_id, step=1,
    )
    if jump_id != sample_id and jump_id in normal_data:
        sample_id = jump_id
        if sample_id in display_ids:
            current_idx = display_ids.index(sample_id)

    st.sidebar.write(f"Viewing: **Sample {sample_id}**")
    if sample_id in annotations:
        st.sidebar.write("Status: **Annotated**")
    else:
        st.sidebar.write("Status: **Not annotated**")

    # --- Sample metadata ---
    n_meta = normal_data[sample_id]
    r_meta = relational_data[sample_id]
    rel_info = relation_data.get(sample_id, {})

    st.markdown("---")

    # Header with status
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.subheader(f"Sample {sample_id}")
    with col_status:
        if sample_id in annotations:
            st.success("Saved")
        else:
            st.warning("Not saved yet")

    st.markdown(f'> *"{n_meta["sentence"]}"*')

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("True Label", n_meta["label"])
    col_m2.metric("Predicted", n_meta["predicted_class"])
    col_m3.metric("Confidence", f"{n_meta['confidence']:.2%}")

    relations = rel_info.get("manual_relations", [])
    if relations:
        phrases = [" ".join(r) if isinstance(r, list) else r for r in relations]
        st.markdown(f"**Relation groups:** {' | '.join(phrases)}")

    # --- Existing per-feature annotations (for pre-filling) ---
    existing = annotations.get(sample_id, {})
    existing_nfa = existing.get("normal_feature_annotations", {})
    existing_rfa = existing.get("relational_feature_annotations", {})

    # === SHAP VISUALIZATIONS ===
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        html_n = render_highlighted_text(n_meta["shap_values"], "Normal SHAP (token-level)")
        st.markdown(html_n, unsafe_allow_html=True)
        st.markdown("")
        fig_n = render_shap_bar(n_meta["shap_values"], "Top-8 Normal SHAP")
        st.plotly_chart(fig_n, use_container_width=True, key=f"bar_n_{sample_id}")
        st.markdown("**Mark each feature:**")
        normal_fa = render_feature_annotation(
            n_meta["shap_values"], "normal", sample_id, existing_nfa,
        )

    with col2:
        html_r = render_highlighted_text(r_meta["shap_values"], "Relational SHAP (phrase-level)")
        st.markdown(html_r, unsafe_allow_html=True)
        st.markdown("")
        fig_r = render_shap_bar(r_meta["shap_values"], "Top-8 Relational SHAP")
        st.plotly_chart(fig_r, use_container_width=True, key=f"bar_r_{sample_id}")
        st.markdown("**Mark each feature:**")
        relational_fa = render_feature_annotation(
            r_meta["shap_values"], "relational", sample_id, existing_rfa,
        )

    # === OVERALL ANNOTATION ===
    st.markdown("---")
    st.subheader("Overall Judgment")
    st.markdown("**Overall, does the model predict the emotion for the correct reason?**")

    col_a1, col_a2 = st.columns(2)
    overall_options = ["Correct Reason", "Wrong Reason", "Unclear / Cannot Decide"]

    with col_a1:
        st.markdown("**Normal SHAP overall:**")
        ex_n = existing.get("normal_shap_label", "")
        n_idx = overall_options.index(ex_n) if ex_n in overall_options else 0
        normal_label = st.radio(
            "Normal", overall_options, index=n_idx,
            key=f"overall_n_{sample_id}", label_visibility="collapsed",
        )

    with col_a2:
        st.markdown("**Relational SHAP overall:**")
        ex_r = existing.get("relational_shap_label", "")
        r_idx = overall_options.index(ex_r) if ex_r in overall_options else 0
        relational_label = st.radio(
            "Relational", overall_options, index=r_idx,
            key=f"overall_r_{sample_id}", label_visibility="collapsed",
        )

    comment = st.text_input(
        "Optional comment",
        value=existing.get("comment", ""),
        key=f"c_{sample_id}",
    )

    # === NAVIGATION + SAVE BUTTONS ===
    st.markdown("---")
    col_prev, col_save, col_save_next, col_next, _ = st.columns([1, 1, 1, 1, 2])

    def build_row_data():
        return {
            "sample_id": sample_id,
            "annotator": annotator_name,
            "sentence": n_meta["sentence"],
            "true_label": n_meta["label"],
            "predicted_label": n_meta["predicted_class"],
            "confidence": round(n_meta["confidence"], 4),
            "normal_shap_label": normal_label,
            "relational_shap_label": relational_label,
            "normal_feature_annotations": normal_fa,
            "relational_feature_annotations": relational_fa,
            "normal_shap_values": n_meta["shap_values"],
            "relational_shap_values": r_meta["shap_values"],
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }

    def do_save(row_data):
        save_annotation_to_csv(annotator_name, row_data)
        if sheets_connected and ws:
            save_annotation_to_sheet(ws, row_data)
        annotations[sample_id] = row_data
        st.session_state.annotations = annotations

    with col_prev:
        if st.button("Prev", key=f"prev_{sample_id}", disabled=(current_idx == 0)):
            st.session_state.current_idx = current_idx - 1
            st.rerun()

    with col_save:
        if st.button("Save", key=f"save_{sample_id}"):
            row_data = build_row_data()
            do_save(row_data)
            st.success("Saved!")
            st.rerun()

    with col_save_next:
        if st.button("Save & Next", type="primary", key=f"savenext_{sample_id}",
                      disabled=(current_idx >= len(display_ids) - 1)):
            row_data = build_row_data()
            do_save(row_data)
            st.session_state.current_idx = current_idx + 1
            st.rerun()

    with col_next:
        if st.button("Next", key=f"next_{sample_id}",
                      disabled=(current_idx >= len(display_ids) - 1)):
            st.session_state.current_idx = current_idx + 1
            st.rerun()

    # === SIDEBAR STATS ===
    st.sidebar.markdown("---")
    st.sidebar.header("Stats")
    if annotations:
        my_anns = {k: v for k, v in annotations.items() if k in set(my_ids)}
        vals = list(my_anns.values())
        if vals:
            n_corr = sum(1 for a in vals if a.get("normal_shap_label") == "Correct Reason")
            n_wrong = sum(1 for a in vals if a.get("normal_shap_label") == "Wrong Reason")
            r_corr = sum(1 for a in vals if a.get("relational_shap_label") == "Correct Reason")
            r_wrong = sum(1 for a in vals if a.get("relational_shap_label") == "Wrong Reason")
            st.sidebar.markdown(f"""
| | Correct | Wrong |
|---|---|---|
| **Normal** | {n_corr} | {n_wrong} |
| **Relational** | {r_corr} | {r_wrong} |
            """)

    # === SIDEBAR DOWNLOAD ===
    st.sidebar.markdown("---")
    csv_path = OUTPUT_DIR / f"{annotator_name}_annotations.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            st.sidebar.download_button(
                "Download my annotations (CSV)",
                f.read(),
                file_name=f"{annotator_name}_annotations.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
