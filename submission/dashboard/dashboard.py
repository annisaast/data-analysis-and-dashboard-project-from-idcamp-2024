import pandas as pd
import numpy as np  # Pastikan numpy diimpor
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Judul Aplikasi
st.title('Dashboard Interaktif Streamlit')

# Menampilkan Teks
st.header('Visualisasi Data')

# Menggunakan slider untuk input pengguna
st.subheader('Pilih Rentang Angka')
start = st.slider('Mulai:', 1, 100, 1)
end = st.slider('Selesai:', 1, 100, 100)

# Membuat Dataframe berdasarkan input pengguna
data = pd.DataFrame({
    'Angka': np.arange(start, end + 1),
    'Kuadrat': np.arange(start, end + 1) ** 2,
    'Kubik': np.arange(start, end + 1) ** 3
})

# Menampilkan Tabel Data
st.subheader('Tabel Data')
st.write(data)

# Menampilkan Grafik
st.subheader('Grafik Kuadrat dan Kubik')
fig, ax = plt.subplots()
ax.plot(data['Angka'], data['Kuadrat'], label='Kuadrat')
ax.plot(data['Angka'], data['Kubik'], label='Kubik')

ax.set_xlabel('Angka')
ax.set_ylabel('Nilai')
ax.legend()
st.pyplot(fig)
