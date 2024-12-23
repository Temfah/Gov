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

# เก็บข้อมูล X_train, X_test, y_train, y_test ลงในไฟล์ CSV
train_data = pd.DataFrame(X_train, columns=X.columns)
train_data["fault"] = y_train
train_data.to_csv("X_train_data.csv", index=False)

test_data = pd.DataFrame(X_test, columns=X.columns)
test_data["fault"] = y_test
test_data.to_csv("X_test_data.csv", index=False)

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

    # แสดงกราฟสำหรับข้อมูลที่ทำนายเป็นทั้ง Normal และ Repair Needed จากการกรอกมือ
    manual_df["Prediction"] = manual_prediction
    manual_df["Status"] = manual_df["Prediction"].apply(lambda x: "Repair Needed" if x == 1 else "Normal")
    st.subheader("Performance Governor - Manual Input")
    st.line_chart(manual_df.drop(columns=["Prediction", "Status"]))

    # เก็บข้อมูลที่กรอกมือและทำนายลงในไฟล์ CSV
    manual_df["Input Source"] = "Manual Input"
    manual_df.to_csv("manual_input_predictions.csv", mode='a', header=not os.path.exists("manual_input_predictions.csv"), index=False)

# Upload File (CSV หรือ Excel)
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

    # แสดงกราฟ Performance Governor สำหรับข้อมูลที่ทำนายเป็นทั้ง Normal และ Repair Needed จากการอัปโหลดไฟล์
    st.subheader("Performance Governor - Uploaded File")
    st.line_chart(uploaded_data.drop(columns=["Prediction", "Status"]))

    # เก็บข้อมูลที่อัปโหลดและทำนายลงในไฟล์ CSV
    uploaded_data["Input Source"] = "Uploaded File"
    uploaded_data.to_csv("uploaded_file_predictions.csv", mode='a', header=not os.path.exists("uploaded_file_predictions.csv"), index=False)

    st.subheader("Fault Analysis Graphs")
    st.line_chart(uploaded_data.drop(columns=["Prediction", "Status"]))

    # สร้าง folder ถ้ายังไม่มี
    output_folder = "C:/Users/598667/GOVRecent"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # บันทึกไฟล์ CSV ลงใน folder
    output_file = os.path.join(output_folder, "predicted_maintenance_data.csv")
    uploaded_data.to_csv(output_file, index=False)
    st.write(f"File has been saved to {output_file}")

# ตัวอย่างการบันทึกข้อมูลจำลอง
st.download_button(
    label="Download Example Dataset",
    data=df.to_csv(index=False),
    file_name="example_hydropower_data.csv",
    mime="text/csv"
)
