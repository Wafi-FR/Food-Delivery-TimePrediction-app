# food_delivery_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Food Delivery EDA", layout="wide")

# 1. Upload & Load Data
st.title('Food Delivery Time Exploratory Analysis')
st.markdown("Upload file **Food_Delivery_Times.csv** untuk mulai eksplorasi:")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 2. Data Preprocessing
    st.header('Data Preprocessing')
    # Missing value info
    st.write("Jumlah missing value per kolom:")
    st.write(df.isnull().sum())
    
    # Drop NA (critical cols only)
    df_clean = df.dropna(subset=['Delivery_Time_min', 'Distance_km', 'Vehicle_Type', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Preparation_Time_min'])
    df_clean['Courier_Experience_yrs'] = df_clean.groupby('Vehicle_Type')['Courier_Experience_yrs'].transform(lambda x: x.fillna(x.median()))
    
    # 3. Data Overview
    st.header('Statistik Dasar')
    st.dataframe(df_clean.head(10))
    st.write(df_clean.describe())
    st.write("Jumlah data:", df_clean.shape[0])
    
    # 4. Chart Sidebar Filter
    st.sidebar.header("Filter Visualisasi")
    weather_filter = st.sidebar.multiselect("Weather", options=df_clean['Weather'].unique(), default=list(df_clean['Weather'].unique()))
    traffic_filter = st.sidebar.multiselect("Traffic Level", options=df_clean['Traffic_Level'].unique(), default=list(df_clean['Traffic_Level'].unique()))
    vehicle_filter = st.sidebar.multiselect("Vehicle Type", options=df_clean['Vehicle_Type'].unique(), default=list(df_clean['Vehicle_Type'].unique()))
    
    # Apply filter
    df_vis = df_clean[
        df_clean['Weather'].isin(weather_filter) &
        df_clean['Traffic_Level'].isin(traffic_filter) &
        df_clean['Vehicle_Type'].isin(vehicle_filter)
    ]
    
    # 5. EDA & Visualisasi
    st.header('Exploratory Data Analysis')
    
    # Distribusi Delivery Time
    st.subheader("Distribusi Delivery Time (menit)")
    fig1 = px.histogram(df_vis, x="Delivery_Time_min", color="Weather", marginal="box", nbins=40)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Korelasi numerik
    st.subheader("Korelasi Fitur Numerik")
    num_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']
    correlation = df_vis[num_cols].corr()
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
    
    # Scatter: jarak vs waktu delivery
    st.subheader("Jarak vs Delivery Time (warna Traffic)")
    fig4 = px.scatter(df_vis, x="Distance_km", y="Delivery_Time_min", color="Traffic_Level", hover_data=['Vehicle_Type', 'Weather'])
    st.plotly_chart(fig4, use_container_width=True)
    
    # Boxplot: Delivery Time vs Vehicle Type
    st.subheader("Delivery Time menurut Vehicle Type")
    fig5 = px.box(df_vis, x="Vehicle_Type", y="Delivery_Time_min", color="Vehicle_Type", points="all")
    st.plotly_chart(fig5)
    
    # Boxplot Delivery Time vs Weather
    st.subheader("Delivery Time menurut Weather")
    fig6 = px.box(df_vis, x="Weather", y="Delivery_Time_min", color="Weather", points="all")
    st.plotly_chart(fig6)
    
    # Rekomendasi bisnis (statik—bisa dibuat lebih dinamis dengan insight otomatis)
    st.header("Business Insight & Rekomendasi")
    st.markdown("""
    - **Cuaca & traffic buruk tingkatkan waktu kirim:** Cek dan atur ekstra buffer pada perkiraan waktu tiba.
    - **Persiapan restoran lama juga menambah delay:** Prioritaskan pesanan dengan persiapan lama saat area/kondisi buruk.
    - **Jarak pengiriman sangat linear dengan waktu:** Optimasi rute penting di jam sibuk dan kondisi ekstrim.
    - **Pengalaman kurir berpengaruh:** Tugas di area sibuk sebaiknya diutamakan untuk kurir berpengalaman.
    """)

else:
    st.info("Silakan upload file dataset untuk mulai analisis.")

