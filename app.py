import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Heart Stroke detected by VORTEXNISH", page_icon="❤️", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>❤️ Heart Stroke detected by VORTEXNISH</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Next-gen cardiovascular risk intelligence</p>", unsafe_allow_html=True)

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🧠 Prediction", "📊 Analytics", "📁 Reports"])

# ================= TAB 1 =================
with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 100, 40)
        resting_bp = st.number_input("BP", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 600, 200)

    with col2:
        max_hr = st.slider("Max HR", 60, 220, 150)
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        fasting_bs = st.selectbox("Fasting Blood Sugar>120mg/dl", [0, 1])

    with col3:
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "TA", "ASY"])
        exercise_angina = st.selectbox("Angina", ["Y", "N"])
        resting_ecg = st.selectbox("ECG", ["Normal", "ST", "LVH"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("🚀 Run AI Analysis"):

        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]
        scaled = scaler.transform(df)

        prob = model.predict_proba(scaled)[0][1]

        # Save history
        st.session_state.history.append(prob)

        st.divider()
        st.subheader("📊 AI Result")

        # ----------- GAUGE -----------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ----------- MESSAGE -----------
        if prob > 0.7:
            st.error("🔴 High Risk! Consult doctor.")
        elif prob > 0.4:
            st.warning("🟠 Moderate Risk.")
        else:
            st.success("🟢 Low Risk.")


with tab2:

    st.subheader("📊 Patient Profile Radar")

    radar = go.Figure()

    radar.add_trace(go.Scatterpolar(
        r=[age/100, resting_bp/200, cholesterol/600, max_hr/220],
        theta=["Age", "BP", "Cholesterol", "HR"],
        fill='toself'
    ))

    radar.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(radar, use_container_width=True)

    # History chart
    st.subheader("📈 Prediction History")
    st.line_chart(st.session_state.history)


with tab3:

    st.subheader("📁 Export Report")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history, columns=["Risk"])

        st.download_button(
            label="⬇️ Download Report",
            data=df_hist.to_csv(index=False),
            file_name="heart_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No data yet")

st.info("⚠️ THIS IS AN AI BASED MODEL WITH A ACCURACY OF 88% USED FOR PREDICTION ONLY  Not a medical diagnosis.")