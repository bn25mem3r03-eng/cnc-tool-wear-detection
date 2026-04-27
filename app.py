import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="CNC Tool Wear Monitor",
    page_icon="⚙️",
    layout="wide"
)

# ── Load model ───────────────────────────────
model = joblib.load('tool_wear_model.pkl')
features = joblib.load('selected_features.pkl')

# ── Header ───────────────────────────────────
st.title("⚙️ CNC Tool Wear Detection System")
st.markdown("**Smart Manufacturing | University of Michigan Dataset | KNN Classifier**")
st.divider()

# ── Sidebar inputs ───────────────────────────
st.sidebar.header("🔧 Machine Parameters")
feed_rate = st.sidebar.selectbox("Feed Rate (mm/s)", [0.5, 1.0, 1.5])
clamp     = st.sidebar.selectbox("Clamp Pressure (bar)", [2.5, 3.0, 4.0])

st.sidebar.divider()
st.sidebar.markdown("### 📂 Upload Experiment File")
uploaded = st.sidebar.file_uploader("Upload experiment CSV", type=['csv'])

# ── Main logic ───────────────────────────────
if uploaded:
    df = pd.read_csv(uploaded)
    
    active = ['Layer 1 Up','Layer 1 Down',
              'Layer 2 Up','Layer 2 Down',
              'Layer 3 Up','Layer 3 Down']
    df_active = df[df['Machining_Process'].isin(active)]
    
    # Extract features
    row = {}
    for col in features:
        sensor = col.rsplit('_', 1)[0]   # e.g. Y1_ActualVelocity
        stat   = col.rsplit('_', 1)[1]   # e.g. mean
        if sensor in df_active.columns:
            if stat == 'mean': row[col] = df_active[sensor].mean()
            elif stat == 'std': row[col] = df_active[sensor].std()
            elif stat == 'max': row[col] = df_active[sensor].max()
            elif stat == 'rms': row[col] = np.sqrt(np.mean(df_active[sensor]**2))
    
    X_input = pd.DataFrame([row])[features].fillna(0)
    prediction = model.predict(X_input)[0]
    label = "🔴 WORN" if prediction == 1 else "🟢 UNWORN"
    color = "red" if prediction == 1 else "green"
    
    # ── Results display ──────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Tool Status", label)
    col2.metric("Feed Rate", f"{feed_rate} mm/s")
    col3.metric("Clamp Pressure", f"{clamp} bar")
    
    st.divider()
    
    if prediction == 1:
        st.error("⚠️ WARNING: Tool wear detected! Schedule tool replacement.")
    else:
        st.success("✅ Tool condition is good. Continue machining.")
    
    # ── Signal plots ─────────────────────────
    st.subheader("📊 Sensor Signal Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df_active['Y1_ActualVelocity'].values[:200], color='royalblue')
        ax.set_title('Y-Axis Velocity Signal')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Velocity (mm/s)')
        st.pyplot(fig)
    
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(df_active['S1_ActualVelocity'].values[:200], color='coral')
        ax.set_title('Spindle Velocity Signal')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Velocity (mm/s)')
        st.pyplot(fig)

else:
    st.info("👈 Upload an experiment CSV file from the sidebar to get started.")
    st.markdown("""
    ### How to use:
    1. Upload any of the 18 experiment CSV files in the sidebar
    2. The system will automatically extract sensor features
    3. The ML model will predict tool condition instantly
    """)