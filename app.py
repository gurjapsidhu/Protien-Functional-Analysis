import streamlit as st
import pandas as pd
import joblib
import re


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Protein Function Predictor",
    page_icon="🧬",
    layout="wide"
)


# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load("protein_model.joblib")


try:
    model_bundle = load_model()

    model = model_bundle["model"]
    mlb = model_bundle["mlb"]
    amino_acids = model_bundle["amino_acids"]

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =====================================================
# 3. AAC FEATURE EXTRACTION
# =====================================================

def calculate_aac(sequence):

    features = []

    for aa in amino_acids:

        frequency = sequence.count(aa) / len(sequence)
        features.append(frequency)

    return pd.DataFrame(
        [features],
        columns=amino_acids
    )


# =====================================================
# 4. CLEAR INPUT CALLBACK
# =====================================================

def clear_sequence():

    st.session_state["protein_sequence"] = ""


@st.cache_data
def load_go_database():

    go_terms = {}
    current_term = {}

    with open("go-basic.obo", "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line == "[Term]":

                if current_term:
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
                current_term["definition"] = line[5:].split('"')[1] if '"' in line else ""

            elif line == "is_obsolete: true":
                current_term["obsolete"] = True

    if current_term:
        if (
            current_term.get("id")
            and not current_term.get("obsolete", False)
        ):
            go_terms[current_term["id"]] = current_term

    return go_terms


go_database = load_go_database()


# =====================================================
# 5. GLOBAL DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(0, 210, 255, 0.12),
            transparent 35%
        ),
        radial-gradient(
            circle at 85% 75%,
            rgba(125, 60, 255, 0.12),
            transparent 35%
        ),
        #080D16;
}

/* Main heading */

.stApp h1 {
    font-family: Arial, sans-serif;
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #E8F4FF;
}

/* Subheadings */

.stApp h2, .stApp h3 {
    font-family: Arial, sans-serif;
    color: #B9D8EA;
    font-weight: 600;
}

/* General text */

.stApp p {
    font-family: Arial, sans-serif;
    color: #A9B9CC;
}

/* Input fields */

.stTextArea textarea {
    background: #0B1521 !important;
    color: #DCEBFA !important;
    border: 1px solid #263D50 !important;
    border-radius: 9px !important;
    font-family: monospace !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 14px !important;
}

.stTextArea textarea:focus {
    border-color: #00CFE8 !important;
    box-shadow: 0 0 0 1px rgba(0, 207, 232, 0.2) !important;
}

/* Dropdowns */

.stSelectbox div[data-baseweb="select"] > div {
    background: #0B1521;
    border: 1px solid #263D50;
    border-radius: 8px;
    color: #DCEBFA;
}

/* Buttons */

div.stButton > button {
    background: rgba(0, 229, 255, 0.07);
    color: #00E5FF;
    border: 1px solid rgba(0, 229, 255, 0.5);
    border-radius: 8px;
    height: 46px;
    font-size: 13px;
    font-weight: 650;
    letter-spacing: 1px;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: rgba(0, 229, 255, 0.16);
    border-color: #00E5FF;
    color: white;
    box-shadow: 0 0 16px rgba(0, 229, 255, 0.15);
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 6. PAGE INTRODUCTION
# =====================================================

st.title("Protein Function Prediction & Analysis")

st.subheader(
    "Predict biological functions from protein sequences"
)

st.markdown("""
**Machine Learning for Protein Function Prediction**

*Understanding the molecular functions of proteins through
computational methods and machine learning.*

---

## Project Information

- **Dataset:** AmiGO
- **Prediction Category:** Molecular Function (MF)
- **Features:** Amino Acid Composition (AAC)
- **Model:** Logistic Regression with One-vs-Rest classification

---

*An exploration of machine learning and computational biology.*
""")


# =====================================================
# 7. PROTEIN INPUT TOOL
# =====================================================

with st.container(border=True):

    header, status = st.columns([4, 1])

    with header:

        st.markdown("### 🧬 Sequence Analysis")

        st.caption(
            "Configure and submit a protein sequence."
        )

    with status:

        st.caption("INPUT MODULE / 01")

    st.divider()

    # Input configuration

    format_col, target_col = st.columns(2)

    with format_col:

        input_format = st.selectbox(
            "INPUT FORMAT",
            ["Raw sequence", "FASTA"]
        )

    with target_col:

        target = st.selectbox(
            "PREDICTION TARGET",
            ["Molecular Function (MF)"]
        )

    # Sequence input

    st.markdown("**AMINO ACID SEQUENCE**")

    sequence_input = st.text_area(
        "Sequence",
        placeholder="Paste your protein sequence here...",
        height=170,
        label_visibility="collapsed",
        key="protein_sequence"
    )

    # -------------------------------------------------
    # PARSE SEQUENCE
    # -------------------------------------------------

    if input_format == "FASTA":

        lines = sequence_input.strip().splitlines()

        sequence = "".join(
            line.strip()
            for line in lines
            if not line.strip().startswith(">")
        )

    else:

        sequence = "".join(sequence_input.split())

    sequence = sequence.upper()

    # -------------------------------------------------
    # VALIDATE SEQUENCE
    # -------------------------------------------------

    valid_chars = set(amino_acids)

    valid = (
        len(sequence) > 0
        and all(aa in valid_chars for aa in sequence)
    )

    # -------------------------------------------------
    # SEQUENCE OVERVIEW
    # -------------------------------------------------

    st.divider()

    st.markdown("**SEQUENCE OVERVIEW**")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Residues",
            f"{len(sequence):,}"
        )

    with metric2:

        st.metric(
            "Input characters",
            f"{len(sequence_input):,}"
        )

    with metric3:

        st.metric(
            "Validation",
            "Valid" if valid else "Pending"
        )

    # Validation feedback

    if sequence:

        if valid:

            st.success("Sequence validation successful.")

        else:

            st.error(
                "Invalid sequence. Use the 20 standard amino acid letters."
            )

    # -------------------------------------------------
    # ACTION BUTTONS
    # -------------------------------------------------

    st.divider()

    action_col, clear_col = st.columns([3, 1])

    with action_col:

        predict_clicked = st.button(
            "ANALYZE SEQUENCE",
            use_container_width=True
        )

    with clear_col:

        st.button(
            "CLEAR",
            on_click=clear_sequence,
            use_container_width=True
        )


# =====================================================
# 8. PREDICTION OUTPUT
# =====================================================

if predict_clicked:

    if not valid:

        st.warning("Please enter a valid amino acid sequence.")

    else:

        features_df = calculate_aac(sequence)
        probs = model.predict_proba(features_df)[0]

        scored_terms = sorted(
            zip(mlb.classes_, probs),
            key=lambda x: x[1],
            reverse=True
        )

        top_predictions = [
            (go_id, prob)
            for go_id, prob in scored_terms[:5]
            if prob >= 0.15
        ]

        st.divider()
        st.subheader("Predicted Protein Functions")
        st.metric("Sequence Length", f"{len(sequence)} residues")

        for rank, (go_id, prob) in enumerate(top_predictions, start=1):

            term_info = go_database.get(go_id, {})
            name = term_info.get("name", go_id).title()
            definition = term_info.get("definition", "")

            with st.container(border=True):

                st.markdown(f"### {rank}. {name}")
                st.code(go_id)
                st.write(definition)
                st.markdown(f"Model estimated probability: **{prob * 100:.1f}%**")
