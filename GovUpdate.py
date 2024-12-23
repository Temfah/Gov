import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
from sklearn.utils import resample

# สร้างข้อมูลจำลอง
np.random.seed(42)
n_samples = 20000

parameters = {
    "GV POSITION (%)": np.random.uniform(0, 100, n_samples),
    "RB POSITION (ｰ)": np.random.uniform(0, 90, n_samples),
    "GEN MW (%)": np.random.uniform(0, 100, n_samples),
    "GEN Hz (%)": np.random.uniform(47, 53, n_samples),
    "TURBINE SPEED (%)": np.random.uniform(95, 105, n_samples),
}

df = pd.DataFrame(parameters)

def generate_fault(row):
    if (
        row["RB POSITION (ｰ)"] > 85 or
        row["GEN MW (%)"] > 95 or
        row["GEN Hz (%)"] < 48.5 or row["GEN Hz (%)"] > 51.5 or
        row["TURBINE SPEED (%)"] > 103
    ):
        return 1
    return 0

df["fault"] = df.apply(generate_fault, axis=1)

majority_class = df[df['fault'] == 0]
minority_class = df[df['fault'] == 1]

if len(majority_class) > len(minority_class):
    minority_upsampled = resample(
        minority_class,
        replace=True,
        n_samples=len(majority_class),
        random_state=42
    )
    balanced_data = pd.concat([majority_class, minority_upsampled])
else:
    majority_upsampled = resample(
        majority_class,
        replace=True,
        n_samples=len(minority_class),
        random_state=42
    )
    balanced_data = pd.concat([majority_upsampled, minority_class])

balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

balanced_data.to_csv("balanced_data.csv", index=False)

X = balanced_data.drop(columns=["fault"])
y = balanced_data["fault"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=64, verbose=1)

y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

model.save("predictive_maintenance_model.h5")

st.title("Predictive Maintenance for Governor Control")

# Input แบบกรอกมือ
st.sidebar.subheader("Manual Input Parameters")

# ใช้ None เป็นค่าเริ่มต้น
params = {
    "GV POSITION (%)": st.sidebar.number_input("GV POSITION (%)", min_value=0, max_value=100),
    "RB POSITION (ｰ)": st.sidebar.number_input("RB POSITION (ｰ)", min_value=0, max_value=90),
    "GEN MW (%)": st.sidebar.number_input("GEN MW (%)", min_value=0, max_value=100),
    "GEN Hz (%)": st.sidebar.number_input("GEN Hz (%)", min_value=47, max_value=53),
    "TURBINE SPEED (%)": st.sidebar.number_input("TURBINE SPEED (%)", min_value=95, max_value=105),
}

# ตรวจสอบว่าทุกพารามิเตอร์มีค่าก่อนการทำนาย
if all(value is not None for value in params.values()):
    if st.sidebar.button("Predict from Manual Input"):
        manual_df = pd.DataFrame([params])

        manual_scaled = scaler.transform(manual_df)
        manual_prediction = (model.predict(manual_scaled) > 0.5).astype(int)[0][0]
        status = "Repair Needed" if manual_prediction == 1 else "Normal"
        st.sidebar.write(f"Prediction: {status}")

        history_file = "prediction_history.csv"

        if not os.path.exists(history_file):
            history_df = pd.DataFrame(columns=list(params.keys()) + ["Prediction", "Status"])
        else:
            history_df = pd.read_csv(history_file)

        new_data = pd.DataFrame([{**params, "Prediction": manual_prediction, "Status": status}])
        history_df = pd.concat([history_df, new_data], ignore_index=True)

        if len(history_df) > 1000:
            history_df = history_df.tail(1000)

        history_df.to_csv(history_file, index=False)

        st.subheader("Prediction History (Last 1000 Records)")
        st.line_chart(history_df[list(params.keys())])

uploaded_file = st.file_uploader("Upload Parameters (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        uploaded_data = pd.read_csv(uploaded_file)
    else:
        uploaded_data = pd.read_excel(uploaded_file)

    uploaded_scaled = scaler.transform(uploaded_data)
    predictions = (model.predict(uploaded_scaled) > 0.5).astype(int)
    uploaded_data["Prediction"] = predictions
    uploaded_data["Status"] = uploaded_data["Prediction"].apply(lambda x: "Repair Needed" if x == 1 else "Normal")

    st.subheader("Prediction Results")
    st.write(uploaded_data)

    st.subheader("Performance Governor - Uploaded File")
    st.line_chart(uploaded_data.drop(columns=["Prediction", "Status"]))

st.download_button(
    label="Download Example Dataset",
    data=df.to_csv(index=False),
    file_name="example_hydropower_data.csv",
    mime="text/csv"
)
