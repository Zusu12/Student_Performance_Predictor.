import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ==========================
# CUSTOM BLUE & WHITE THEME + SIDEBAR TITLE COLOR
# ==========================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #e5f0ff;
    }

    /* Sidebar header "Enter Student Details" */
    section[data-testid="stSidebar"] h2 {
        color: #1e3a8a !important;   /* Change this color */
        font-weight: 700 !important;
        text-align: center;
        font-size: 22px !important;
    }

    /* Main headings */
    h1, h2, h3, h4 {
        color: #1e3a8a;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #1d4ed8;
        color: white;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1e40af;
        color: white;
    }

    /* Slider + Select label colors */
    .stSlider label, .stSelectbox label {
        color: #1e3a8a !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# LOAD DATA
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("CollegeStudentsPerformance.csv")

df = load_data()

# ==========================
# FEATURE ENGINEERING
# ==========================
df["total_marks"] = df["internal_marks"] + df["external_marks"]
df["avg_marks"] = df["total_marks"] / 2

def performance_label(x):
    if x < 60:
        return 0  # Low
    elif x < 80:
        return 1  # Mid
    return 2      # High

df["target"] = df["total_marks"].apply(performance_label)

# ==========================
# ENCODE CATEGORICAL FEATURES
# ==========================
cat_cols = df.select_dtypes(include=["object"]).columns
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ==========================
# MODEL TRAINING
# ==========================
X = df.drop(columns=["target"])
y = df["target"]

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X, y)

# ==========================
# UI HEADER
# ==========================
st.markdown("<h1 style='text-align: center;color:#1e3a8a;'>🎓 College Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#1e3a8a;'>Low / Mid / High Performance Classifier</h4>", unsafe_allow_html=True)

# ==========================
# SIDEBAR INPUTS
# ==========================
st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", encoders["gender"].classes_)
department = st.sidebar.selectbox("Department", encoders["department"].classes_)
test_prep_choice = st.sidebar.selectbox("Test Preparation", ["None", "Coaching", "Self"])

study_hours = st.sidebar.slider("Study Hours per Day", 0, 8, 3)
attendance = st.sidebar.slider("Attendance (%)", 50, 100, 85)
cgpa = st.sidebar.slider("Previous Sem CGPA", 5.0, 10.0, 7.5, step=0.1)
internal = st.sidebar.slider("Internal Marks (out of 40)", 0, 40, 28)
external = st.sidebar.slider("External Marks (out of 60)", 0, 60, 45)
backlogs = st.sidebar.selectbox("Backlogs History", [0, 1])

# ==========================
# SAFE MAPPING FOR TEST PREPARATION
# ==========================
tp_classes = [str(c).strip() for c in encoders["test_preparation"].classes_]
user_choice = test_prep_choice.lower()

if user_choice == "self":
    if "none" in [c.lower() for c in tp_classes]:
        mapped_label = "None"
    else:
        mapped_label = tp_classes[0]
else:
    mapped_label = test_prep_choice

matched_label = None
for c in tp_classes:
    if c.lower() == mapped_label.lower():
        matched_label = c
        break

if matched_label is None:
    matched_label = tp_classes[0]

test_prep_encoded = encoders["test_preparation"].transform([matched_label])[0]

# ==========================
# CREATE INPUT DATAFRAME
# ==========================
input_data = pd.DataFrame([{
    "gender": encoders["gender"].transform([gender])[0],
    "department": encoders["department"].transform([department])[0],
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "previous_sem_cgpa": cgpa,
    "internal_marks": internal,
    "external_marks": external,
    "test_preparation": test_prep_encoded,
    "backlogs_history": backlogs,
    "total_marks": internal + external,
    "avg_marks": (internal + external) / 2
}])

# ==========================
# PREDICTION BUTTON
# ==========================
if st.button("Predict Performance"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    label_map = {0: "LOW", 1: "MID", 2: "HIGH"}
    colors = {0: "red", 1: "orange", 2: "green"}

    st.markdown(f"<h3 style='color:{colors[pred]};'>Predicted Performance: {label_map[pred]}</h3>", unsafe_allow_html=True)

    st.subheader("Prediction Confidence")
    st.bar_chart(pd.DataFrame({
        "Performance": ["Low", "Mid", "High"],
        "Probability": proba
    }).set_index("Performance"))

    st.subheader("Entered Details")
    st.json({
        "Gender": gender,
        "Department": department,
        "Study Hours": study_hours,
        "Attendance %": attendance,
        "Previous CGPA": cgpa,
        "Internal Marks": internal,
        "External Marks": external,
        "Test Prep": test_prep_choice,
        "Backlogs": backlogs
    })
