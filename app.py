import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
import json

st.set_page_config(page_title="Obesity Class Predictor", page_icon="🍏", layout="centered")

APP_DIR = Path(__file__).resolve().parent
TRAINED_DIR = APP_DIR / "Trained"
PREPROC_DIR = APP_DIR / "Preprocessing" / "output"
CSV_FALLBACK = PREPROC_DIR / "cleaned_obesity_level.csv"

MODEL_PATH = TRAINED_DIR / "RF_model.pkl"
META_PATHS = [TRAINED_DIR / "RF_model_meta.json", APP_DIR / "RF_model_meta.json"]

def _load_json_first(paths):
    for p in paths:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8")), p
        except Exception:
            continue
    return None, None

@st.cache_resource
def load_artifacts(model_path: Path, meta_paths):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    pipe = joblib.load(model_path)
    meta, meta_path = _load_json_first(meta_paths)
    return pipe, meta, meta_path

try:
    pipe, meta, meta_path = load_artifacts(MODEL_PATH, META_PATHS)
except Exception as e:
    st.error(f"Could not load model from **{MODEL_PATH}**. {e}")
    st.stop()

def final_estimator(p):
    try:
        return p.steps[-1][1]
    except Exception:
        return p

def get_column_transformer(pipeline):
    try:
        if hasattr(pipeline, "named_steps"):
            for key in ["prep", "preprocess", "preprocessor", "column_transformer"]:
                if key in pipeline.named_steps:
                    return pipeline.named_steps[key]
    except Exception:
        pass
    return None

def expected_columns(pipe, meta, csv_fallback: Path) -> list:
    if isinstance(meta, dict) and "feature_order" in meta and meta["feature_order"]:
        return list(meta["feature_order"])

    try:
        ct = get_column_transformer(pipe)
        if ct is not None and hasattr(ct, "transformers_"):
            cols = []
            for name, trans, selected in ct.transformers_:
                if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend(list(selected))
            if cols:
                seen = set()
                ordered = []
                for c in cols:
                    if c not in seen:
                        ordered.append(c)
                        seen.add(c)
                return ordered
    except Exception:
        pass

    try:
        if hasattr(pipe, "feature_names_in_"):
            return list(pipe.feature_names_in_)
    except Exception:
        pass

    try:
        fe = final_estimator(pipe)
        if hasattr(fe, "feature_names_in_"):
            return list(fe.feature_names_in_)
    except Exception:
        pass

    try:
        if csv_fallback.exists():
            df = pd.read_csv(csv_fallback)
            drop = {"id", "obesity_class"}
            return [c for c in df.columns if c not in drop]
    except Exception:
        pass

    return [
        "Gender","Age","family_hist","highcalorie","vegtables","main_meals","snacks",
        "smokes","water_intake","monitors_calories","physical_activity","screen_time",
        "alcohol","transport","BMI"
    ]

EXPECTED_COLS = expected_columns(pipe, meta, CSV_FALLBACK)

def get_class_labels(pipe, meta):
    if isinstance(meta, dict) and meta.get("classes"):
        return [str(c) for c in meta["classes"]]
    try:
        fe = final_estimator(pipe)
        if hasattr(fe, "classes_"):
            return [str(c) for c in fe.classes_]
    except Exception:
        pass
    return None

CLASS_LABELS = get_class_labels(pipe, meta)

# def scale_to_normalized(value, original_min, original_max, normalized_min, normalized_max):
#     return normalized_min + (value - original_min) * (normalized_max - normalized_min) / (original_max - original_min)

st.title("🍏 Obesity Class Predictor")

with st.form("input_form"):
    st.subheader("Basics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", min_value=14, max_value=100, value=25, step=1)
    with col2:
        height = st.number_input(
            "Height (meters)",
            min_value=1.0, max_value=2.5, value=1.70, step=0.01, format="%.2f"
        )
        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0, max_value=250.0, value=70.0, step=0.5
        )

    try:
        bmi = float(round(weight / (height ** 2), 2))
    except Exception:
        bmi = 24.0

    st.subheader("Habits & lifestyle")
    c1, c2, c3 = st.columns(3)
    with c1:
        family_hist = st.selectbox("Family history of overweight", ["No", "Yes"])
        highcalorie = st.selectbox("High-calorie food (FAVC)", ["No", "Yes"])
        smokes = st.selectbox("Smokes", ["No", "Yes"])
        monitors_calories = st.selectbox("Monitors calories (SCC)", ["No", "Yes"])
    with c2:
        fcvc = st.slider("Vegetables frequency per meal (FCVC)", 1, 6, 2, 1)
        ncp = st.slider("Number of main meals (NCP)", 1, 6, 3, 1)
        water_intake = st.slider("Daily water intake (litres)", 1.0, 6.0, 2.0, 0.5)
    with c3:
        faf = st.slider("Physical activity (hours/day)", 0.0, 12.0, 1.0, 0.5)
        screen_time = st.slider("Screen/tech time (hours/day)", 0.0, 12.0, 1.0, 0.5)

    st.subheader("Choices")
    c4, c5 = st.columns(2)
    with c4:
        snacks = st.selectbox("Snacking (CAEC)", ["Never", "Sometimes", "Frequently", "Always"])
        alcohol = st.selectbox("Alcohol (CALC)", ["Never", "Sometimes", "Frequently"])
    with c5:
        transport = st.selectbox(
            "Primary transport (MTRANS)",
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
        )

    submitted = st.form_submit_button("Predict")

# fcvc = scale_to_normalized(fcvc, 1, 6, 1.0, 3.0)
# ncp = scale_to_normalized(ncp, 1, 6, 1.0, 3.0)
# water_intake = scale_to_normalized(water_intake, 1.0, 6.0, 1.0, 3.0)
# faf = scale_to_normalized(faf, 0, 12, 0.0, 3.0)
# screen_time = scale_to_normalized(screen_time, 0, 12, 0.0, 3.0)

def bool_to_int(x: str) -> int:
    return 1 if str(x).lower() in ["yes", "1", "true"] else 0

def never_to_zero(x: str) -> str:
    return "0" if str(x).strip().lower() == "never" else x

canonical_values = {
    "Gender": gender,
    "Age": int(age),
    "BMI": float(bmi),
    "family_hist": bool_to_int(family_hist),
    "highcalorie": bool_to_int(highcalorie),
    "vegtables": float(fcvc),
    "main_meals": int(ncp),
    "snacks": never_to_zero(snacks),
    "smokes": bool_to_int(smokes),
    "water_intake": float(water_intake),
    "monitors_calories": bool_to_int(monitors_calories),
    "physical_activity": float(faf),
    "screen_time": float(screen_time),
    "alcohol": never_to_zero(alcohol),
    "transport": transport,

    "FAVC": bool_to_int(highcalorie),
    "FCVC": float(fcvc),
    "NCP": int(ncp),
    "CAEC": never_to_zero(snacks),
    "SMOKE": bool_to_int(smokes),
    "CH2O": float(water_intake),
    "SCC": bool_to_int(monitors_calories),
    "FAF": float(faf),
    "TUE": float(screen_time),

    "gender": gender,
    "age": int(age),
    "bmi": float(bmi),
    "Height": float(height),
    "Weight": float(weight),
    "height": float(height),
    "weight": float(weight),
    "water": float(water_intake),
    "tech_time": float(screen_time),
}

def assemble_row(expected_cols: list, provided: dict) -> pd.DataFrame:
    if not expected_cols:
        return pd.DataFrame([{k: provided.get(k, np.nan) for k in provided.keys()}])
    row = {}
    for col in expected_cols:
        val = provided.get(col, np.nan)
        if col in ("snacks", "CAEC", "alcohol", "CALC"):
            val = never_to_zero(val)
        row[col] = val
    return pd.DataFrame([row])

if submitted:
    X_row = assemble_row(EXPECTED_COLS, canonical_values)
    try:
        proba = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_row)[0]

        y_pred = pipe.predict(X_row)[0]
        label = str(y_pred)
        st.success(f"**Predicted class:** {label}")

        if proba is not None:
            classes = CLASS_LABELS
            if classes is None:
                try:
                    fe = final_estimator(pipe)
                    if hasattr(fe, "classes_"):
                        classes = [str(c) for c in fe.classes_]
                except Exception:
                    classes = None
            if classes is None or len(classes) != len(proba):
                classes = [f"class_{i}" for i in range(len(proba))]

            dfp = pd.DataFrame({"Class": classes, "Probability": proba})
            dfp = dfp.sort_values("Probability", ascending=False).reset_index(drop=True)

            st.write("Top probabilities:")
            st.dataframe(dfp.astype({"Class": "string"}).style.format({"Probability": "{:.3f}"}),
                         use_container_width=True)
            st.bar_chart(dfp.set_index("Class"))

        with st.expander("Show raw input used"):
            st.dataframe(X_row.T.astype("string"), use_container_width=True)

        with st.expander("Model & columns (debug)"):
            st.write("Model path:", str(MODEL_PATH))
            st.write("Meta path used:", str(meta_path) if meta_path else None)
            st.code(", ".join(map(str, EXPECTED_COLS)), language="text")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        with st.expander("Debug info"):
            st.write("Expected columns:", EXPECTED_COLS)
            st.write("Input row columns:", list(X_row.columns))
            st.write("Input row:", X_row.to_dict(orient="records")[0])
