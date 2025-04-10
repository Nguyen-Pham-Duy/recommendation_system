import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Open and read file to cosine_sim_new
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Giao diện nhập liệu
st.title("Phân nhóm khách hàng bằng RFM")

r = st.number_input("Recency (R)", min_value=0.0, step=1.0)
f = st.number_input("Frequency (F)", min_value=0.0, step=1.0)
m = st.number_input("Monetary (M)", min_value=0.0, step=1.0)

if st.button("Dự đoán nhóm"):
    # Tạo array RFM
    user_rfm = np.array([[r, f, m]])
    
    # Dự đoán nhóm
    cluster = kmeans_model.predict(user_rfm)
    
    st.success(f"Khách hàng này thuộc nhóm: {cluster[0]}")