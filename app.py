import gzip
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Protein Function Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================
# 2. FILE PATHS
# Keep these files in the repository root beside app.py:
# protein_model.joblib
# go-basic.obo.gz
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "protein_model.joblib"
GO_PATH = BASE_DIR / "go-basic.obo.gz"


# =====================================================
# 3. MODEL LOADING
# =====================================================

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


try:
    model_bundle = load_model(str(MODEL_PATH))

    model = model_bundle["model"]
    mlb = model_bundle["mlb"]
    amino_acids = list(model_bundle["amino_acids"])

    if not amino_acids:
        raise ValueError("The model bundle contains no amino-acid features.")

except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.info(
        "Check that protein_model.joblib exists in the same repository "
        "directory as app.py and that its bundle contains model, mlb, "
        "and amino_acids."
    )
    st.stop()


# =====================================================
# 4. GO DATABASE LOADING
# =====================================================

@st.cache_data(show_spinner="Loading Gene Ontology database...")
def load_go_database(go_path: str):
    """Load non-obsolete GO terms from a compressed OBO file."""

    go_terms = {}
    current_term = {}

    with gzip.open(go_path, "rt", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if line == "[Term]":
                if (
                    current_term.get("id")
                    and not current_term.get("obsolete", False)
                ):
                    go_terms[current_term["id"]] = current_term

                current_term = {}

            elif line.startswith("id: "):
                current_term["id"] = line[4:]

            elif line.startswith("name: "):
                current_term["name"] = line[6:]

            elif line.startswith("namespace: "):
                current_term["namespace"] = line[11:]

            elif line.startswith("def: "):
                definition_text = line[5:]
                if '"' in definition_text:
                    current_term["definition"] = (
                        definition_text.split('"', 2)[1]
                    )
                else:
                    current_term["definition"] = ""

            elif line == "is_obsolete: true":
                current_term["obsolete"] = True

    # Save the final term in the file.
    if (
        current_term.get("id")
        and not current_term.get("obsolete", False)
    ):
        go_terms[current_term["id"]] = current_term

    return go_terms


try:
    if not GO_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {GO_PATH.name} in the repository root."
        )

    go_database = load_go_database(str(GO_PATH))

except Exception as exc:
    go_database = {}
    st.warning(
        f"Gene Ontology annotations could not be loaded: {exc}. "
        "Predictions will still run, but GO names and definitions "
        "may be unavailable."
    )


# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================

def clear_sequence():
    st.session_state["protein_sequence"] = ""


def parse_sequence(raw_input: str, input_format: str) -> str:
    """Normalize raw or FASTA input into an uppercase sequence."""

    lines = raw_input.strip().splitlines()

    if input_format == "FASTA":
        sequence_lines = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith(">")
        ]
        sequence = "".join(sequence_lines)
    else:
        sequence = "".join(raw_input.split())

    return sequence.upper()


def calculate_aac(sequence: str) -> pd.DataFrame:
    """Calculate amino-acid composition in the model's feature order."""

    if not sequence:
        raise ValueError("The protein sequence is empty.")

    features = [
        sequence.count(aa) / len(sequence)
        for aa in amino_acids
    ]

    return pd.DataFrame([features], columns=amino_acids)


def analyze_sequence(sequence: str):
    """Calculate simple composition and residue-class summaries."""

    total = len(sequence)

    counts = {
        aa: sequence.count(aa)
        for aa in amino_acids
    }

    composition = pd.DataFrame({
        "Amino Acid": list(counts.keys()),
        "Count": list(counts.values()),
    })

    composition["Percentage"] = (
        composition["Count"] / total * 100
    )

    # Broad, simplified residue groupings.
    # These are descriptive categories, not a full biochemical model.
    hydrophobic = set("AVILMFWY")
    polar = set("STNQCY")
    charged = set("DEKRH")

    groups = {
        "Hydrophobic": sum(aa in hydrophobic for aa in sequence),
        "Polar": sum(aa in polar for aa in sequence),
        "Charged": sum(aa in charged for aa in sequence),
    }

    properties = pd.DataFrame({
        "Residue Group": list(groups.keys()),
        "Count": list(groups.values()),
    })

    properties["Percentage"] = (
        properties["Count"] / total * 100
    )

    return composition, properties


def get_go_info(go_id: str):
    """Return annotation fields, with sensible fallbacks."""

    term = go_database.get(go_id, {})

    return {
        "name": term.get("name", go_id),
        "namespace": term.get("namespace", "Not available"),
        "definition": term.get(
            "definition",
            "No definition is available in the loaded GO database.",
        ),
    }


# =====================================================
# 6. GLOBAL DESIGN
# =====================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 10% 5%,
                rgba(0, 210, 220, 0.10), transparent 40%),
            radial-gradient(ellipse at 90% 80%,
                rgba(100, 70, 220, 0.10), transparent 40%),
            #080D16;
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    .stApp h1 {
        color: #EAF7FF;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -1.5px;
    }

    .stApp h2, .stApp h3 {
        color: #C8E6F5;
        font-weight: 700;
    }

    .stApp p, .stApp label {
        color: #A8BACD;
    }

    .eyebrow {
        display: inline-block;
        background: rgba(0, 220, 210, 0.10);
        color: #00E5D4;
        border: 1px solid rgba(0, 220, 210, 0.28);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }

    .hero-subtitle {
        color: #9FB6C9;
        font-size: 1.05rem;
        margin-top: -8px;
    }

    .hero-line {
        height: 3px;
        width: 100%;
        background: linear-gradient(
            90deg, transparent, #00D9D9, #7C5CFC, transparent
        );
        background-size: 200% 100%;
        animation: gradientMove 6s linear infinite;
        border-radius: 5px;
        margin: 18px 0 25px 0;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(90, 180, 210, 0.19);
        border-radius: 14px;
        background: rgba(14, 25, 39, 0.58);
        transition: border-color 0.25s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(0, 220, 230, 0.40);
    }

    div[data-testid="stMetric"] {
        background: rgba(14, 30, 45, 0.72);
        border: 1px solid rgba(65, 145, 180, 0.18);
        border-radius: 12px;
        padding: 13px;
    }

    div[data-testid="stMetricValue"] {
        color: #00E5D4;
        font-weight: 800;
    }

    .stTextArea textarea {
        background: #0B1521 !important;
        color: #E2F1FF !important;
        border: 1px solid #29465A !important;
        border-radius: 11px !important;
        font-family: monospace !important;
        font-size: 14px !important;
        line-height: 1.8 !important;
    }

    .stTextArea textarea:focus {
        border-color: #00D9D9 !important;
        box-shadow: 0 0 0 2px rgba(0, 217, 217, 0.12) !important;
    }

    .stButton button, .stDownloadButton button {
        background: linear-gradient(110deg, #063F4A, #123A62);
        color: #E5FFFF;
        border: 1px solid #087E8B;
        border-radius: 9px;
        min-height: 43px;
        font-weight: 700;
        transition: all 0.22s ease;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        border-color: #00E5D4;
        box-shadow: 0 0 16px rgba(0, 220, 220, 0.18);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"] {
        color: #A8BACD;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #00E5D4 !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    section.main > div {
        animation: fadeIn 0.55s ease-out;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 7. PAGE INTRODUCTION
# =====================================================

st.markdown(
    """
    <div class="eyebrow">COMPUTATIONAL BIOLOGY · MACHINE LEARNING</div>
    """,
    unsafe_allow_html=True,
)

st.title("Protein Function Analysis")

st.markdown(
    """
    <div class="hero-subtitle">
        From amino acid sequences to predicted molecular functions.
    </div>
    <div class="hero-line"></div>
    """,
    unsafe_allow_html=True,
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Prediction Model", "Logistic Regression")

with info2:
    st.metric("Feature Extraction", "AAC")

with info3:
    st.metric("Annotation Resource", "Gene Ontology")

with st.expander("ABOUT THIS PROJECT"):
    st.markdown(
        """
        This application explores protein function prediction using
        machine learning and amino acid composition.

        - **Model:** One-vs-Rest Logistic Regression
        - **Features:** Amino Acid Composition (AAC)
        - **Prediction target:** Molecular Function (MF)
        - **Annotation resource:** Gene Ontology
        - **Input:** Protein amino acid sequence

        AAC represents each sequence using the relative frequencies
        of its amino acids. It does not preserve residue order or
        directly model protein structure.
        """
    )


# =====================================================
# 8. SEQUENCE INPUT AND VALIDATION
# =====================================================

with st.container(border=True):

    header_col, status_col = st.columns([4, 1])

    with header_col:
        st.markdown("### 🧬 Sequence Analysis")
        st.caption("Enter a protein sequence to inspect and analyze it.")

    with status_col:
        st.caption("INPUT MODULE / 01")

    st.divider()

    format_col, target_col = st.columns(2)

    with format_col:
        input_format = st.selectbox(
            "INPUT FORMAT",
            ["Raw sequence", "FASTA"],
        )

    with target_col:
        st.selectbox(
            "PREDICTION TARGET",
            ["Molecular Function (MF)"],
            disabled=True,
        )

    st.markdown("**AMINO ACID SEQUENCE**")

    sequence_input = st.text_area(
        "Sequence",
        placeholder=(
            "Paste your protein sequence here...\n"
            "FASTA example: >protein_name"
        ),
        height=170,
        label_visibility="collapsed",
        key="protein_sequence",
    )

    sequence = parse_sequence(sequence_input, input_format)

    valid_chars = set(amino_acids)
    invalid_chars = sorted(set(sequence) - valid_chars)

    valid = bool(sequence) and not invalid_chars

    st.divider()
    st.markdown("**SEQUENCE OVERVIEW**")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Residues", f"{len(sequence):,}")

    with metric2:
        st.metric("Input characters", f"{len(sequence_input):,}")

    with metric3:
        st.metric(
            "Validation",
            "Valid" if valid else ("Invalid" if sequence else "Pending"),
        )

    if sequence:
        if valid:
            st.success("Sequence validation successful.")
        else:
            st.error(
                "Invalid sequence. Unsupported characters: "
                + ", ".join(invalid_chars)
            )
            st.caption(
                "This model accepts only the amino-acid letters "
                "included in its trained feature set."
            )

    st.divider()

    action_col, clear_col = st.columns([3, 1])

    with action_col:
        predict_clicked = st.button(
            "ANALYZE SEQUENCE",
            use_container_width=True,
            type="primary",
        )

    with clear_col:
        st.button(
            "CLEAR",
            on_click=clear_sequence,
            use_container_width=True,
        )


# =====================================================
# 9. LIVE SEQUENCE ANALYTICS
# =====================================================

if valid:

    composition, properties_df = analyze_sequence(sequence)

    st.markdown("## Sequence Analytics")
    st.caption(
        "Descriptive statistics calculated directly from the input sequence."
    )

    tab1, tab2 = st.tabs([
        "Amino Acid Composition",
        "Residue Properties",
    ])

    with tab1:

        st.markdown("### Amino Acid Frequency")

        composition_chart = (
            composition
            .set_index("Amino Acid")[["Percentage"]]
            .sort_values("Percentage", ascending=False)
        )

        st.bar_chart(
            composition_chart,
            color="#00CFC8",
        )

        st.dataframe(
            composition.sort_values("Count", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:

        st.markdown("### Broad Residue Categories")

        properties_chart = properties_df.set_index(
            "Residue Group"
        )[["Percentage"]]

        st.bar_chart(
            properties_chart,
            color="#7C72F2",
        )

        st.dataframe(
            properties_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Residue categories are simplified descriptive groupings. "
            "They are not a complete measure of protein hydrophobicity, "
            "charge state, folding, or structure."
        )


# =====================================================
# 10. PREDICTION OUTPUT
# =====================================================

if predict_clicked:

    if not valid:
        st.warning("Please enter a valid protein amino-acid sequence.")

    else:

        try:
            with st.spinner("Analyzing sequence and calculating predictions..."):

                features_df = calculate_aac(sequence)
                probabilities = model.predict_proba(features_df)[0]

                scored_terms = sorted(
                    zip(mlb.classes_, probabilities),
                    key=lambda item: item[1],
                    reverse=True,
                )

                # Always show the five highest-scoring terms.
                top_predictions = scored_terms[:5]

            st.divider()
            st.markdown("## Prediction Report")
            st.caption(
                "Ranked molecular-function predictions generated by the model."
            )

            highest_score = float(top_predictions[0][1]) if top_predictions else 0.0
            displayed_count = sum(
                float(score) >= 0.15
                for go_id, score in top_predictions
            )

            result1, result2, result3 = st.columns(3)

            with result1:
                st.metric("Sequence Length", f"{len(sequence):,} aa")

            with result2:
                st.metric("Top Model Score", f"{highest_score * 100:.2f}%")

            with result3:
                st.metric("Top 5 Above 15%", displayed_count)

            st.info(
                "Interpretation note: these values are model-estimated "
                "scores. They should not be treated as experimentally "
                "validated probabilities or confirmed protein functions."
            )

            # Score chart
            st.markdown("### Prediction Score Distribution")

            chart_rows = []

            for go_id, score in top_predictions:
                go_info = get_go_info(go_id)
                chart_rows.append({
                    "GO Term": go_info["name"],
                    "Model Score (%)": float(score) * 100,
                })

            chart_data = pd.DataFrame(chart_rows)

            if not chart_data.empty:
                st.bar_chart(
                    chart_data.set_index("GO Term"),
                    color="#00D9C8",
                )

            # Detailed result cards
            st.markdown("### Ranked Molecular Functions")

            report_rows = []

            for rank, (go_id, score) in enumerate(
                top_predictions,
                start=1,
            ):

                score = float(score)
                go_info = get_go_info(go_id)

                name = go_info["name"]
                namespace = go_info["namespace"]
                definition = go_info["definition"]

                report_rows.append({
                    "Rank": rank,
                    "GO ID": go_id,
                    "Function": name,
                    "Namespace": namespace,
                    "Model Score (%)": round(score * 100, 2),
                    "Definition": definition,
                })

                with st.container(border=True):

                    card_left, card_right = st.columns([4, 1])

                    with card_left:
                        st.markdown(f"### {rank}. {name}")
                        st.caption(f"{go_id} · {namespace}")

                    with card_right:
                        st.metric(
                            "Model Score",
                            f"{score * 100:.1f}%",
                        )

                    st.progress(max(0.0, min(1.0, score)))

                    st.write(definition)

                    st.markdown(
                        f"[View this Gene Ontology term]"
                        f"(https://amigo.geneontology.org/amigo/term/{go_id})"
                    )

            # Export results
            st.divider()
            st.markdown("### Export Analysis")

            report_df = pd.DataFrame(report_rows)

            download_col, fasta_col = st.columns(2)

            with download_col:
                st.download_button(
                    "DOWNLOAD PREDICTION REPORT",
                    data=report_df.to_csv(index=False).encode("utf-8"),
                    file_name="protein_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with fasta_col:
                fasta_data = f">Protein_Sequence\n{sequence}\n"

                st.download_button(
                    "DOWNLOAD FASTA",
                    data=fasta_data,
                    file_name="protein_sequence.fasta",
                    mime="text/plain",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption(
                "Check that the model's expected feature columns match "
                "the amino_acids list stored in protein_model.joblib."
            )


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown(
    """
    <div style="
        margin-top: 55px;
        padding-top: 18px;
        border-top: 1px solid rgba(100,160,190,.18);
        text-align: center;
        color: #71869A;
        font-size: 11px;
        letter-spacing: .5px;">
        PROTEIN FUNCTION ANALYSIS · MACHINE LEARNING · GENE ONTOLOGY
    </div>
    """,
    unsafe_allow_html=True,
)
