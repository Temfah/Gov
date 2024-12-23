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

# พารามิเตอร์และการกระจายตัว
parameters = {
    "GV POSITION (%)": np.random.uniform(0, 100, n_samples),
    "RB POSITION (ｰ)": np.random.uniform(0, 90, n_samples),
    "GEN MW (%)": np.random.uniform(0, 100, n_samples),
    "GEN Hz (%)": np.random.uniform(47, 53, n_samples),
    "TURBINE SPEED (%)": np.random.uniform(95, 105, n_samples),
}

df = pd.DataFrame(parameters)

# กำหนดกฎสำหรับค่าผิดปกติ
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

# การสมดุลข้อมูล
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

# Shuffle the data to ensure random distribution
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# เก็บข้อมูลที่ถูกสมดุลไว้ในไฟล์ CSV
balanced_data.to_csv("balanced_data.csv", index=False)

# แบ่งข้อมูลใหม่หลังจากทำการสมดุล
X = balanced_data.drop(columns=["fault"])
y = balanced_data["fault"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# มาตรฐานข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้างโมเดล
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, batch_size=64, verbose=1)

# ประเมินผล
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# บันทึกโมเดลที่ฝึกแล้วลงในไฟล์
model.save("predictive_maintenance_model.h5")

# โปรแกรม Streamlit
st.title("Predictive Maintenance for Governor Control")

# Input แบบกรอกมือ
st.sidebar.subheader("Manual Input Parameters")
man_gv = st.sidebar.number_input("GV POSITION (%)")
man_rb = st.sidebar.number_input("RB POSITION (ｰ)")
man_gen_mw = st.sidebar.number_input("GEN MW (%)")
man_gen_hz = st.sidebar.number_input("GEN Hz (%)")
man_turbine_speed = st.sidebar.number_input("TURBINE SPEED (%)")

if st.sidebar.button("Predict from Manual Input"):
    manual_df = pd.DataFrame([{
        "GV POSITION (%)": man_gv,
        "RB POSITION (ｰ)": man_rb,
        "GEN MW (%)": man_gen_mw,
        "GEN Hz (%)": man_gen_hz,
        "TURBINE SPEED (%)": man_turbine_speed
    }])
    manual_scaled = scaler.transform(manual_df)
    manual_prediction = (model.predict(manual_scaled) > 0.5).astype(int)[0][0]
    status = "Repair Needed" if manual_prediction == 1 else "Normal"
    st.sidebar.write(f"Prediction: {status}")

    # เก็บประวัติการกรอกค่าทำนาย
    history_file = "prediction_history.csv"
    
    # ถ้าไฟล์ไม่พบบันทึกให้สร้างไฟล์ใหม่
    if not os.path.exists(history_file):
        history_df = pd.DataFrame(columns=["GV POSITION (%)", "RB POSITION (ｰ)", "GEN MW (%)", "GEN Hz (%)", "TURBINE SPEED (%)", "Prediction", "Status"])
    else:
        history_df = pd.read_csv(history_file)
    
    # บันทึกข้อมูลการกรอก
    new_data = pd.DataFrame([{
        "GV POSITION (%)": man_gv,
        "RB POSITION (ｰ)": man_rb,
        "GEN MW (%)": man_gen_mw,
        "GEN Hz (%)": man_gen_hz,
        "TURBINE SPEED (%)": man_turbine_speed,
        "Prediction": manual_prediction,
        "Status": status
    }])
    
    history_df = pd.concat([history_df, new_data], ignore_index=True)
    
    # เก็บข้อมูลประวัติย้อนหลังไม่เกิน 1000 ครั้ง
    if len(history_df) > 1000:
        history_df = history_df.tail(1000)
    
    # เก็บไฟล์ CSV
    history_df.to_csv(history_file, index=False)

    # แสดงกราฟประวัติการทำนาย
    st.subheader("Prediction History (Last 1000 Records)")
    st.line_chart(history_df[["GV POSITION (%)", "RB POSITION (ｰ)", "GEN MW (%)", "GEN Hz (%)", "TURBINE SPEED (%)"]])

# ตัวอย่างการบันทึกข้อมูลจำลอง
st.download_button(
    label="Download Example Dataset",
    data=df.to_csv(index=False),
    file_name="example_hydropower_data.csv",
    mime="text/csv"
)
