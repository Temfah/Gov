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

model.save("predictive_maintenance_model.h5")

st.title("Predictive Maintenance for Governor Control")

# ประวัติการพยากรณ์
history_file = "prediction_history.csv"
if not os.path.exists(history_file):
    history_df = pd.DataFrame(columns=["GV POSITION (%)", "RB POSITION (ｰ)", "GEN MW (%)", "GEN Hz (%)", "TURBINE SPEED (%)", "Prediction", "Status"])
    history_df.to_csv(history_file, index=False)
else:
    history_df = pd.read_csv(history_file)

st.subheader("Prediction History Overview")
st.write(f"Total Predictions Made: {len(history_df)}")
st.write(history_df.tail(10))  # แสดงประวัติ 10 รายการล่าสุด

if not history_df.empty:
    st.line_chart(history_df[["GV POSITION (%)", "RB POSITION (ｰ)", "GEN MW (%)", "GEN Hz (%)", "TURBINE SPEED (%)"]])

# Input Parameters
st.sidebar.subheader("Manual Input Parameters")
gv_position = st.sidebar.number_input("GV POSITION (%)")
rb_position = st.sidebar.number_input("RB POSITION (ｰ)")
gen_mw = st.sidebar.number_input("GEN MW (%)")
gen_hz = st.sidebar.number_input("GEN Hz (%)")
turbine_speed = st.sidebar.number_input("TURBINE SPEED (%)")

data_for_graph = pd.DataFrame({
    "GV POSITION (%)": [gv_position],
    "RB POSITION (ｰ)": [rb_position],
    "GEN MW (%)": [gen_mw],
    "GEN Hz (%)": [gen_hz],
    "TURBINE SPEED (%)": [turbine_speed]
})

st.subheader("Input Parameter Values")
st.write(data_for_graph)
st.subheader("Graph for Input Parameters")
st.line_chart(data_for_graph)

if st.sidebar.button("Predict from Manual Input"):
    manual_df = pd.DataFrame([{
        "GV POSITION (%)": gv_position,
        "RB POSITION (ｰ)": rb_position,
        "GEN MW (%)": gen_mw,
        "GEN Hz (%)": gen_hz,
        "TURBINE SPEED (%)": turbine_speed
    }])

    manual_scaled = scaler.transform(manual_df)
    manual_prediction = (model.predict(manual_scaled) > 0.5).astype(int)[0][0]
    status = "Repair Needed" if manual_prediction == 1 else "Normal"

    st.sidebar.write(f"Prediction: {status}")

    new_data = pd.DataFrame([{
        "GV POSITION (%)": gv_position,
        "RB POSITION (ｰ)": rb_position,
        "GEN MW (%)": gen_mw,
        "GEN Hz (%)": gen_hz,
        "TURBINE SPEED (%)": turbine_speed,
        "Prediction": manual_prediction,
        "Status": status
    }])

    history_df = pd.concat([history_df, new_data], ignore_index=True)

    if len(history_df) > 1000:
        history_df = history_df.tail(1000)

    history_df.to_csv(history_file, index=False)

    st.subheader("Updated Prediction History")
    st.write(history_df.tail(10))
    st.line_chart(history_df[["GV POSITION (%)", "RB POSITION (ｰ)", "GEN MW (%)", "GEN Hz (%)", "TURBINE SPEED (%)"]])
# เพิ่มการอัปโหลดไฟล์ CSV/Excel
st.subheader("Upload Parameters for Prediction")
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        uploaded_data = pd.read_csv(uploaded_file)
    else:
        uploaded_data = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(uploaded_data)

    # ตรวจสอบและพยากรณ์จากข้อมูลที่อัปโหลด
    try:
        uploaded_scaled = scaler.transform(uploaded_data)
        uploaded_predictions = (model.predict(uploaded_scaled) > 0.5).astype(int)

        uploaded_data["Prediction"] = uploaded_predictions
        uploaded_data["Status"] = uploaded_data["Prediction"].apply(lambda x: "Repair Needed" if x == 1 else "Normal")

        st.subheader("Prediction Results")
        st.write(uploaded_data)

        # บันทึกผลลัพธ์เป็นไฟล์ CSV สำหรับดาวน์โหลด
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_result = convert_df(uploaded_data)
        st.download_button(
            label="Download Prediction Results",
            data=csv_result,
            file_name="prediction_results.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error in processing uploaded file: {e}")

# ดาวน์โหลดตัวอย่างข้อมูล
st.subheader("Download Example Data")
example_data = pd.DataFrame({
    "GV POSITION (%)": [50, 60, 70],
    "RB POSITION (ｰ)": [40, 50, 60],
    "GEN MW (%)": [80, 90, 85],
    "GEN Hz (%)": [49, 50, 51],
    "TURBINE SPEED (%)": [98, 99, 97]
})

example_csv = example_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Example Dataset",
    data=example_csv,
    file_name="example_dataset.csv",
    mime="text/csv"
)
