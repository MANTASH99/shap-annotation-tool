"""
SHAP Annotation Tool — Streamlit Cloud Edition
================================================
- Highlighted sentence text colored by SHAP values
- Per-feature annotation (mark individual words/phrases as Correct / Wrong)
- Per-annotator sample assignment (no overlap)
- Free navigation: go back to any sample, edit previous annotations
- Persistence: Google Sheets on cloud, local CSV when running locally
- IAA mode: Inter-Annotator Agreement measurement on shared samples
"""

import streamlit as st
import json
import csv
import time
import numpy as np
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

# --- Annotator assignment: map name -> (start_id, end_id) or list of IDs ---
ANNOTATOR_ASSIGNMENTS = {
    "Benni": (1, 716),
    "Emilia": (717, 1432),
    "Vanessa": (1433, 2148),
    "Anna": (2149, 2861),
    "Roman": [42, 156, 389, 512, 678, 803, 971, 1105, 1290, 1401,
              1522, 1689, 1834, 2001, 2100, 2250, 2415, 2600, 2750, 2830],
}

ANNOTATOR_NAMES = list(ANNOTATOR_ASSIGNMENTS.keys())
# Roman is a supervisor / reviewer — exclude from IAA calculations
IAA_ANNOTATOR_NAMES = [n for n in ANNOTATOR_NAMES if n != "Roman"]

FEATURE_LABELS = ["—", "Correct", "Wrong"]

# --- IAA Round 1: last 25 from each annotator's range (100 total, 4 emotions) ---
IAA_ROUND1_IDS = sorted(
    list(range(692, 717)) +      # Benni's last 25 (disgust)
    list(range(1408, 1433)) +    # Emilia's last 25 (joy)
    list(range(2124, 2149)) +    # Vanessa's last 25 (sadness)
    list(range(2837, 2862))      # Anna's last 25 (trust)
)

# --- IAA Round 2: 120 samples for 6 missing emotions (20 each, seed=2026) ---
# Each annotator skips the emotion they already annotated (owner).
IAA_ROUND2 = {
    "anger":    {"owner": "Benni",   "ids": [46, 57, 93, 109, 162, 224, 281, 313, 323, 353, 370, 395, 400, 412, 420, 426, 439, 442, 451, 477]},
    "boredom":  {"owner": "Benni",   "ids": [503, 520, 523, 526, 529, 537, 554, 565, 574, 575, 581, 588, 590, 592, 597, 602, 616, 618, 626, 632]},
    "fear":     {"owner": "Emilia",  "ids": [889, 923, 936, 1035, 1063, 1065, 1080, 1081, 1093, 1094, 1113, 1128, 1132, 1136, 1145, 1160, 1172, 1176, 1184, 1207]},
    "pride":    {"owner": "Vanessa", "ids": [1504, 1517, 1544, 1554, 1558, 1564, 1606, 1609, 1620, 1624, 1649, 1697, 1707, 1711, 1725, 1732, 1737, 1738, 1748, 1754]},
    "relief":   {"owner": "Vanessa", "ids": [1759, 1765, 1795, 1808, 1823, 1825, 1832, 1849, 1859, 1867, 1874, 1876, 1879, 1881, 1882, 1890, 1909, 1924, 1934, 1935]},
    "surprise": {"owner": "Anna",    "ids": [2296, 2311, 2328, 2329, 2372, 2411, 2415, 2427, 2441, 2468, 2495, 2500, 2502, 2509, 2512, 2523, 2532, 2533, 2539, 2541]},
}
IAA_ROUND2_ALL_IDS = sorted(sid for info in IAA_ROUND2.values() for sid in info["ids"])


def get_iaa_ids_for_annotator(annotator_name, iaa_round):
    """Get IAA sample IDs for a given annotator and round."""
    if iaa_round == 1:
        return IAA_ROUND1_IDS
    else:
        # Round 2: skip emotions the annotator owns
        ids = []
        for emo, info in IAA_ROUND2.items():
            if info["owner"] != annotator_name:
                ids.extend(info["ids"])
        return sorted(ids)


# Combined for IAA dashboard (all samples from both rounds)
IAA_SAMPLE_IDS = sorted(set(IAA_ROUND1_IDS + IAA_ROUND2_ALL_IDS))

IAA_CATEGORIES = ["Correct Reason", "Wrong Reason", "Unclear / Cannot Decide"]


# ---------------------------------------------------------------------------
# Google Sheets helpers
# ---------------------------------------------------------------------------
def get_gsheet_client():
    if not GSPREAD_AVAILABLE:
        return None
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
    except (FileNotFoundError, KeyError):
        return None
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


def load_annotations_from_sheet(ws, annotator_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            records = ws.get_all_records()
            break
        except gspread.exceptions.APIError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
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


def save_annotation_to_sheet(ws, row_data, max_retries=3):
    sample_id = str(row_data["sample_id"])
    annotator = row_data["annotator"]
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
    for attempt in range(max_retries):
        try:
            cell_list = ws.findall(sample_id, in_column=1)
            for cell in cell_list:
                existing = ws.row_values(cell.row)
                if len(existing) >= 2 and existing[1] == annotator:
                    ws.update(f"A{cell.row}:N{cell.row}", [row_values])
                    return
            ws.append_row(row_values)
            return
        except gspread.exceptions.APIError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # wait 1s, 2s, 4s
            else:
                raise


# ---------------------------------------------------------------------------
# Local CSV saving (works when running locally)
# ---------------------------------------------------------------------------
def save_annotation_to_csv(annotator_name, row_data, suffix=""):
    csv_path = OUTPUT_DIR / f"{annotator_name}{suffix}_annotations.csv"
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


def load_annotations_from_csv(annotator_name, suffix=""):
    csv_path = OUTPUT_DIR / f"{annotator_name}{suffix}_annotations.csv"
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


def render_feature_annotation(features, shap_type, sample_id, existing_fa, key_prefix=""):
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
                key=f"{key_prefix}{shap_type}_{sample_id}_{feat}",
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
# IAA: Agreement computation
# ---------------------------------------------------------------------------
def compute_cohens_kappa(labels_a, labels_b, categories):
    """Compute Cohen's kappa for two raters."""
    n = len(labels_a)
    if n == 0:
        return float("nan")

    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = np.zeros((k, k), dtype=float)
    for a, b in zip(labels_a, labels_b):
        if a in cat_idx and b in cat_idx:
            matrix[cat_idx[a]][cat_idx[b]] += 1

    total = matrix.sum()
    if total == 0:
        return float("nan")

    po = np.trace(matrix) / total  # observed agreement

    pe = 0.0
    for i in range(k):
        pe += (matrix[i, :].sum() / total) * (matrix[:, i].sum() / total)

    if pe == 1.0:
        return 1.0

    return (po - pe) / (1 - pe)


def compute_fleiss_kappa(ratings_matrix):
    """
    Compute Fleiss' kappa for multiple raters.

    ratings_matrix: numpy array of shape (n_subjects, n_categories)
        Each entry is the count of raters who assigned that category.
    """
    n_subjects, n_cats = ratings_matrix.shape
    n_raters = int(ratings_matrix[0].sum())

    if n_subjects == 0 or n_raters <= 1:
        return float("nan")

    # P_i for each subject
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # P_j for each category
    p_j = np.sum(ratings_matrix, axis=0) / (n_subjects * n_raters)
    P_e_bar = np.sum(p_j ** 2)

    if P_e_bar == 1.0:
        return 1.0

    return (P_bar - P_e_bar) / (1 - P_e_bar)


def compute_percent_agreement(labels_per_sample):
    """Fraction of samples where all raters agree."""
    if not labels_per_sample:
        return 0.0
    unanimous = sum(1 for labels in labels_per_sample.values() if len(set(labels)) == 1)
    return unanimous / len(labels_per_sample)


def build_iaa_rating_data(records, label_field, categories):
    """
    Organize raw IAA annotation records into structures for kappa computation.

    Returns:
        labels_by_annotator: {annotator: {sample_id: label}}
        labels_per_sample: {sample_id: [label1, ...]}  (only samples with 2+ raters)
    """
    labels_by_annotator = {}
    for r in records:
        ann = str(r.get("annotator", ""))
        sid = int(r.get("sample_id", 0))
        label = str(r.get(label_field, ""))
        if not ann or not label:
            continue
        if ann not in labels_by_annotator:
            labels_by_annotator[ann] = {}
        labels_by_annotator[ann][sid] = label

    # Build per-sample lists (only samples with 2+ raters)
    all_sids = set()
    for ann_labels in labels_by_annotator.values():
        all_sids.update(ann_labels.keys())

    labels_per_sample = {}
    for sid in sorted(all_sids):
        labels = []
        for ann in IAA_ANNOTATOR_NAMES:
            if ann in labels_by_annotator and sid in labels_by_annotator[ann]:
                labels.append(labels_by_annotator[ann][sid])
        if len(labels) >= 2:
            labels_per_sample[sid] = labels

    return labels_by_annotator, labels_per_sample


# ---------------------------------------------------------------------------
# IAA: Data loading
# ---------------------------------------------------------------------------
def load_iaa_dashboard_data(ws_iaa):
    """Load all IAA annotation records (from all annotators). Cached for 30s."""
    cache_key = "iaa_dashboard_cache"
    cache_time_key = "iaa_dashboard_cache_time"

    now = time.time()
    if (cache_key in st.session_state
            and cache_time_key in st.session_state
            and now - st.session_state[cache_time_key] < 30):
        return st.session_state[cache_key]

    if ws_iaa is not None:
        try:
            records = ws_iaa.get_all_records()
        except Exception:
            records = []
    else:
        # Load from all CSV files
        records = []
        for name in ANNOTATOR_NAMES:
            csv_path = OUTPUT_DIR / f"{name}_iaa_annotations.csv"
            if csv_path.exists():
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        records.append(r)

    st.session_state[cache_key] = records
    st.session_state[cache_time_key] = now
    return records


# ---------------------------------------------------------------------------
# IAA: Annotation UI
# ---------------------------------------------------------------------------
def render_iaa_annotation(annotator_name, normal_data, relational_data,
                          relation_data, ws_iaa, sheets_connected):
    """IAA annotation UI — shared samples for inter-annotator agreement."""

    # Round selector
    iaa_round = st.radio(
        "IAA Round", [1, 2], horizontal=True,
        format_func=lambda x: f"Round {x}" + (" (4 emotions, 100 samples)" if x == 1
                                               else " (6 emotions, 80-100 samples)"),
        key="iaa_round_select",
    )

    # Get samples for this annotator and round
    round_ids = get_iaa_ids_for_annotator(annotator_name, iaa_round)
    iaa_ids = [sid for sid in round_ids if sid in normal_data]
    total = len(iaa_ids)

    if total == 0:
        st.warning("No IAA samples found in loaded data.")
        return

    # Save info
    if sheets_connected and ws_iaa:
        st.info(
            "IAA annotations are **saved to Google Sheets** (worksheet: iaa_annotations) "
            "each time you click **Save**.",
            icon="💾",
        )
    else:
        st.info(
            "IAA annotations are **saved to a local CSV file** each time you click **Save**.",
            icon="💾",
        )

    # Legend
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

    # Load existing IAA annotations
    if ("iaa_annotations" not in st.session_state
            or st.session_state.get("_iaa_annotator") != annotator_name):
        if sheets_connected and ws_iaa:
            with st.spinner("Loading your IAA annotations..."):
                st.session_state.iaa_annotations = load_annotations_from_sheet(
                    ws_iaa, annotator_name)
        else:
            st.session_state.iaa_annotations = load_annotations_from_csv(
                annotator_name, suffix="_iaa")
        st.session_state._iaa_annotator = annotator_name
        # Start at first unannotated sample
        first_unannotated = 0
        for i, sid in enumerate(iaa_ids):
            if sid not in st.session_state.iaa_annotations:
                first_unannotated = i
                break
        st.session_state.iaa_current_idx = first_unannotated

    annotations = st.session_state.iaa_annotations
    annotated_count = sum(1 for sid in iaa_ids if sid in annotations)

    # Sidebar progress
    st.sidebar.header("IAA Progress")
    st.sidebar.progress(annotated_count / total if total else 0)
    st.sidebar.write(f"**{annotated_count} / {total}** annotated")

    # Sidebar navigation
    st.sidebar.header("Navigation")
    view_mode = st.sidebar.radio(
        "View mode",
        ["All samples", "Unannotated only", "Annotated only"],
        key="iaa_view_mode",
    )

    if view_mode == "Unannotated only":
        display_ids = [sid for sid in iaa_ids if sid not in annotations]
    elif view_mode == "Annotated only":
        display_ids = [sid for sid in iaa_ids if sid in annotations]
    else:
        display_ids = iaa_ids

    if not display_ids:
        if view_mode == "Unannotated only":
            st.success("All IAA samples are annotated!")
        else:
            st.info("No annotated IAA samples yet.")
        return

    # Navigation control
    current_idx = st.session_state.get("iaa_current_idx", 0)
    if current_idx >= len(display_ids):
        current_idx = len(display_ids) - 1
    if current_idx < 0:
        current_idx = 0

    idx_input = st.sidebar.number_input(
        f"IAA sample position (1-{len(display_ids)})",
        min_value=1, max_value=len(display_ids),
        value=current_idx + 1, step=1,
    )
    current_idx = idx_input - 1
    sample_id = display_ids[current_idx]

    # Jump to sample ID
    jump_id = st.sidebar.number_input(
        "Jump to IAA sample ID",
        min_value=min(iaa_ids), max_value=max(iaa_ids),
        value=sample_id, step=1,
    )
    if jump_id != sample_id and jump_id in set(iaa_ids):
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

    existing = annotations.get(sample_id, {})

    # === SHAP VISUALIZATIONS (view only, no per-feature marking for IAA) ===
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        html_n = render_highlighted_text(n_meta["shap_values"], "Normal SHAP (token-level)")
        st.markdown(html_n, unsafe_allow_html=True)
        st.markdown("")
        fig_n = render_shap_bar(n_meta["shap_values"], "Top-8 Normal SHAP")
        st.plotly_chart(fig_n, use_container_width=True, key=f"iaa_bar_n_{sample_id}")

    with col2:
        html_r = render_highlighted_text(r_meta["shap_values"], "Relational SHAP (phrase-level)")
        st.markdown(html_r, unsafe_allow_html=True)
        st.markdown("")
        fig_r = render_shap_bar(r_meta["shap_values"], "Top-8 Relational SHAP")
        st.plotly_chart(fig_r, use_container_width=True, key=f"iaa_bar_r_{sample_id}")

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
            key=f"iaa_overall_n_{sample_id}", label_visibility="collapsed",
        )

    with col_a2:
        st.markdown("**Relational SHAP overall:**")
        ex_r = existing.get("relational_shap_label", "")
        r_idx = overall_options.index(ex_r) if ex_r in overall_options else 0
        relational_label = st.radio(
            "Relational", overall_options, index=r_idx,
            key=f"iaa_overall_r_{sample_id}", label_visibility="collapsed",
        )

    comment = st.text_input(
        "Optional comment",
        value=existing.get("comment", ""),
        key=f"iaa_c_{sample_id}",
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
            "normal_feature_annotations": {},
            "relational_feature_annotations": {},
            "normal_shap_values": n_meta["shap_values"],
            "relational_shap_values": r_meta["shap_values"],
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }

    def do_save(row_data):
        save_annotation_to_csv(annotator_name, row_data, suffix="_iaa")
        if sheets_connected and ws_iaa:
            save_annotation_to_sheet(ws_iaa, row_data)
        annotations[sample_id] = row_data
        st.session_state.iaa_annotations = annotations

    with col_prev:
        if st.button("Prev", key=f"iaa_prev_{sample_id}", disabled=(current_idx == 0)):
            st.session_state.iaa_current_idx = current_idx - 1
            st.rerun()

    with col_save:
        if st.button("Save", key=f"iaa_save_{sample_id}"):
            row_data = build_row_data()
            do_save(row_data)
            st.success("Saved!")
            st.rerun()

    with col_save_next:
        if st.button("Save & Next", type="primary", key=f"iaa_savenext_{sample_id}",
                      disabled=(current_idx >= len(display_ids) - 1)):
            row_data = build_row_data()
            do_save(row_data)
            st.session_state.iaa_current_idx = current_idx + 1
            st.rerun()

    with col_next:
        if st.button("Next", key=f"iaa_next_{sample_id}",
                      disabled=(current_idx >= len(display_ids) - 1)):
            st.session_state.iaa_current_idx = current_idx + 1
            st.rerun()


# ---------------------------------------------------------------------------
# IAA: Dashboard
# ---------------------------------------------------------------------------
def display_agreement_metrics(records, label_field, label_display_name):
    """Compute and display agreement metrics for a single label type."""
    categories = IAA_CATEGORIES
    labels_by_annotator, labels_per_sample = build_iaa_rating_data(
        records, label_field, categories)

    if not labels_by_annotator:
        st.info("No annotations available yet.")
        return

    # --- Pairwise Cohen's kappa ---
    st.markdown(f"#### Pairwise Cohen's Kappa")
    pairs = []
    for i in range(len(IAA_ANNOTATOR_NAMES)):
        for j in range(i + 1, len(IAA_ANNOTATOR_NAMES)):
            a, b = IAA_ANNOTATOR_NAMES[i], IAA_ANNOTATOR_NAMES[j]
            a_labels = labels_by_annotator.get(a, {})
            b_labels = labels_by_annotator.get(b, {})
            shared_sids = sorted(set(a_labels.keys()) & set(b_labels.keys()))
            if shared_sids:
                la = [a_labels[sid] for sid in shared_sids]
                lb = [b_labels[sid] for sid in shared_sids]
                kappa = compute_cohens_kappa(la, lb, categories)
                pairs.append({
                    "Pair": f"{a} vs {b}",
                    "Cohen's kappa": f"{kappa:.3f}",
                    "Shared samples": len(shared_sids),
                })
            else:
                pairs.append({
                    "Pair": f"{a} vs {b}",
                    "Cohen's kappa": "N/A",
                    "Shared samples": 0,
                })

    st.table(pairs)

    # --- Fleiss' kappa (samples where ALL raters annotated) ---
    full_samples = {
        sid: labels for sid, labels in labels_per_sample.items()
        if len(labels) == len(IAA_ANNOTATOR_NAMES)
    }

    if full_samples:
        cat_idx = {c: i for i, c in enumerate(categories)}
        n_subjects = len(full_samples)
        ratings_matrix = np.zeros((n_subjects, len(categories)), dtype=float)
        for row_i, (sid, labels) in enumerate(sorted(full_samples.items())):
            for label in labels:
                if label in cat_idx:
                    ratings_matrix[row_i][cat_idx[label]] += 1

        fleiss_k = compute_fleiss_kappa(ratings_matrix)
        st.markdown(f"#### Fleiss' Kappa (all {len(IAA_ANNOTATOR_NAMES)} raters)")
        st.metric(
            f"n={n_subjects} samples, {len(IAA_ANNOTATOR_NAMES)} raters",
            f"{fleiss_k:.3f}",
        )
    else:
        st.info(
            f"Fleiss' kappa requires all {len(IAA_ANNOTATOR_NAMES)} annotators "
            "to rate the same samples. No fully-rated samples yet."
        )

    # --- Percent agreement ---
    if full_samples:
        pct = compute_percent_agreement(full_samples)
        st.metric("Unanimous agreement", f"{pct:.1%}")

    # --- Disagreement browser ---
    disagreed = {
        sid: labels for sid, labels in labels_per_sample.items()
        if len(set(labels)) > 1
    }
    if disagreed:
        st.markdown("#### Disagreements")
        with st.expander(f"View {len(disagreed)} disagreed samples"):
            for sid in sorted(disagreed.keys()):
                annotators_for_sid = []
                for ann in IAA_ANNOTATOR_NAMES:
                    if ann in labels_by_annotator and sid in labels_by_annotator[ann]:
                        annotators_for_sid.append(
                            f"**{ann}**: {labels_by_annotator[ann][sid]}")
                st.markdown(
                    f"**Sample {sid}**: " + " | ".join(annotators_for_sid))


def render_iaa_dashboard(ws_iaa):
    """IAA Dashboard: progress, agreement metrics, disagreement browser."""
    st.subheader("Inter-Annotator Agreement Dashboard")

    # Round selector
    dash_round = st.radio(
        "Show results for", ["Round 1 (4 emotions)", "Round 2 (6 emotions)", "Both rounds combined"],
        horizontal=True, key="iaa_dash_round",
    )

    records = load_iaa_dashboard_data(ws_iaa)

    if not records:
        st.info("No IAA annotations found yet. Start annotating in the Annotate tab!")
        return

    # Filter records by round
    if dash_round.startswith("Round 1"):
        valid_ids = set(IAA_ROUND1_IDS)
        records = [r for r in records if int(r.get("sample_id", 0)) in valid_ids]
    elif dash_round.startswith("Round 2"):
        valid_ids = set(IAA_ROUND2_ALL_IDS)
        records = [r for r in records if int(r.get("sample_id", 0)) in valid_ids]
    # else: combined, keep all records

    # Refresh button
    if st.button("Refresh data", key="iaa_refresh"):
        for key in ["iaa_dashboard_cache", "iaa_dashboard_cache_time"]:
            st.session_state.pop(key, None)
        st.rerun()

    # --- Per-annotator progress ---
    st.markdown("### Annotator Progress")

    if dash_round.startswith("Round 1"):
        # Round 1: all annotators see all 100 samples
        total_iaa = len(IAA_ROUND1_IDS)
    elif dash_round.startswith("Round 2"):
        # Round 2: varies per annotator (80 or 100)
        total_iaa = len(IAA_ROUND2_ALL_IDS)  # max, adjusted per annotator below
    else:
        total_iaa = len(IAA_SAMPLE_IDS)

    progress_data = {name: set() for name in IAA_ANNOTATOR_NAMES}
    for r in records:
        ann = str(r.get("annotator", ""))
        if ann in progress_data:
            progress_data[ann].add(int(r.get("sample_id", 0)))

    cols = st.columns(len(IAA_ANNOTATOR_NAMES))
    for col, name in zip(cols, IAA_ANNOTATOR_NAMES):
        count = len(progress_data[name])
        # For Round 2, each annotator has a different total (skips owned emotions)
        if dash_round.startswith("Round 2"):
            ann_total = len(get_iaa_ids_for_annotator(name, 2))
        else:
            ann_total = total_iaa
        with col:
            st.markdown(f"**{name}**")
            st.progress(count / ann_total if ann_total else 0)
            st.write(f"{count} / {ann_total}")

    st.markdown("---")

    # --- Agreement metrics ---
    st.markdown("### Agreement Metrics")
    tab_normal, tab_relational = st.tabs(["Normal SHAP", "Relational SHAP"])

    with tab_normal:
        display_agreement_metrics(records, "normal_shap_label", "Normal SHAP")

    with tab_relational:
        display_agreement_metrics(records, "relational_shap_label", "Relational SHAP")


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
        for name, assignment in ANNOTATOR_ASSIGNMENTS.items():
            if isinstance(assignment, list):
                st.write(f"**{name}**: {len(assignment)} selected samples")
            else:
                start, end = assignment
                st.write(f"**{name}**: samples {start} - {end} ({end - start + 1} samples)")
        return

    assignment = ANNOTATOR_ASSIGNMENTS[annotator_name]
    if isinstance(assignment, list):
        explicit_ids = assignment
        start_id, end_id = None, None
        st.sidebar.success(f"Your samples: **{len(explicit_ids)} selected samples**")
    else:
        explicit_ids = None
        start_id, end_id = assignment
        st.sidebar.success(f"Your samples: **{start_id}** to **{end_id}**")

    # --- Storage connection (cached in session_state to avoid API calls on every rerun) ---
    sheets_connected = False
    ws = None
    gc = None
    if "ws" in st.session_state and st.session_state.ws is not None:
        ws = st.session_state.ws
        gc = st.session_state.get("gc")
        sheets_connected = True
    else:
        try:
            gc = get_gsheet_client()
            if gc:
                sheet_name = st.secrets.get("sheet_name", "SHAP_Annotations")
                ws = get_or_create_worksheet(gc, sheet_name, "annotations")
                st.session_state.ws = ws
                st.session_state.gc = gc
                sheets_connected = True
        except Exception as e:
            st.sidebar.error(f"Google Sheets error: {e}")

    if sheets_connected:
        st.sidebar.success("Storage: Google Sheets")
    else:
        st.sidebar.info("Storage: Local CSV")

    # --- Mode toggle ---
    mode = st.sidebar.radio("Mode", ["Regular", "IAA"], key="mode_toggle")

    # === IAA MODE ===
    if mode == "IAA":
        # Connect IAA worksheet
        ws_iaa = None
        if sheets_connected and gc:
            if "ws_iaa" in st.session_state:
                ws_iaa = st.session_state.ws_iaa
            else:
                try:
                    sheet_name = st.secrets.get("sheet_name", "SHAP_Annotations")
                    ws_iaa = get_or_create_worksheet(gc, sheet_name, "iaa_annotations")
                    st.session_state.ws_iaa = ws_iaa
                except Exception as e:
                    st.sidebar.error(f"IAA worksheet error: {e}")

        # Load data
        normal_data = load_normal_shap()
        relational_data = load_relational_shap()
        relation_data = load_relation_data()

        tab_annotate, tab_dashboard = st.tabs(["Annotate", "Dashboard"])
        with tab_annotate:
            render_iaa_annotation(
                annotator_name, normal_data, relational_data,
                relation_data, ws_iaa, sheets_connected)
        with tab_dashboard:
            render_iaa_dashboard(ws_iaa)
        return

    # === REGULAR MODE (existing flow, unchanged) ===

    # --- Auto-save info ---
    if sheets_connected:
        save_info = (
            "Your annotations are **saved to Google Sheets** each time you click "
            "**Save**. You can close the browser anytime — when you come back, "
            "all your work will be loaded automatically. No export needed."
        )
    else:
        save_info = (
            "Your annotations are **saved to a local CSV file** each time you click "
            "**Save**. You can close and reopen — your progress is preserved on this machine."
        )
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

    if explicit_ids is not None:
        my_ids = sorted([sid for sid in explicit_ids if sid in normal_data])
    else:
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
    jump_min = min(my_ids) if my_ids else 1
    jump_max = max(my_ids) if my_ids else 1
    jump_id = st.sidebar.number_input(
        "Jump to sample ID",
        min_value=jump_min, max_value=jump_max,
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
