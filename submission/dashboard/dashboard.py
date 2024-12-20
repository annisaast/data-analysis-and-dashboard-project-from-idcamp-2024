pip install numpy pandas matplotlib seaborn statsmodels dash plotly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objs as go

sns.set(style='dark')

# Helper function yang dibutuhkan untuk menyiapkan berbagai dataframe
def create_daily_user_df(df):
    daily_user_df = df.groupby('dteday').agg({
        'casual': 'sum',
        'registered': 'sum',
        'cnt': 'sum'
    }).reset_index()
    return daily_user_df

def create_bymonthyear_df(df):
  df['dteday'] = pd.to_datetime(df['dteday'])
  df['month'] = df['dteday'].dt.month
  bymonthyear_df = df.groupby(['month']).agg({
    'casual': 'mean',
    'registered': 'mean',
    'cnt': 'mean'
  }).reset_index()
  return bymonthyear_df

def create_byseason_df(df):
  df['season'] = df['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'})
  byseason_df = df.groupby(['season']).agg({
    'casual': 'mean',
    'registered': 'mean',
    'cnt': 'mean'
  }).reset_index()
  return byseason_df

def create_byweekday_df(df):
  byweekday_df = df.groupby('weekday').agg({
    'casual': 'mean',
    'registered': 'mean',
    'cnt': 'mean'
  }).reset_index()
  return byweekday_df

def create_byworkingday_df(df):
  byworkingday_df = df.groupby('workingday').agg({
    'casual': 'mean',
    'registered': 'mean',
    'cnt': 'mean'
  }).reset_index()
  return byworkingday_df

def create_byusertype_df(df):
  user_type_mean = df[['casual', 'registered']].mean()
  byusertype_df = pd.DataFrame({
    'user_type': ['Casual', 'Registered'],
    'total': [user_type_mean['casual'], user_type_mean['registered']]
  })
  return byusertype_df

# Decomposition of Time Series
def create_decompose_time_series_df(df):
  df['dteday'] = pd.to_datetime(df['dteday'])
  df.set_index('dteday', inplace=True, drop=False)
  decompose_time_series_df = seasonal_decompose(df['cnt'], model='additive', period=365)
  return decompose_time_series_df

all_df = pd.read_csv("all_data.csv")

# Menyortir dan memastikan kolom datetime
datetime_columns = ["dteday"]
all_df.sort_values(by="dteday", inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
  all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df["dteday"].min()
max_date = all_df["dteday"].max()

with st.sidebar:
  start_date, end_date = st.date_input(
  label = 'Time Range', min_value = min_date,
  max_value = max_date,
  value = [min_date, max_date]
  )

main_df = all_df[(all_df["dteday"] >= str(start_date)) &
         (all_df["dteday"] <= str(end_date))]

# Menyiapkan berbagai DataFrame
daily_user_df = create_daily_user_df(main_df)
bymonthyear_df = create_bymonthyear_df(main_df)
byseason_df = create_byseason_df(main_df)
byweekday_df = create_byweekday_df(main_df)
byworkingday_df = create_byworkingday_df(main_df)
byusertype_df = create_byusertype_df(main_df)
decompose_time_series_df = create_decompose_time_series_df(main_df)

st.header('Bike Sharing Dashboard :sparkles:')
st.subheader('Daily User')

col1, col2, col3 = st.columns(3)
with col1:
  average_cnt = round(daily_user_df.cnt.mean())
  st.metric("Average Count", value=average_cnt)

with col2:
  average_casual = round(daily_user_df.casual.mean())
  st.metric("Average Casual", value=average_casual)

with col3:
  average_registered = round(daily_user_df.registered.mean())
  st.metric("Average Registered", value=average_registered)

# Daily User Visualization
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(
    daily_user_df["dteday"],
    daily_user_df["cnt"],
    marker='o',
    linewidth=2,
    color="#1a569c"
)
ax.set_xlabel('Date', fontsize=24)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)

st.pyplot(fig)

# User Demographics
st.subheader("User Demographics")

# Demographics by Month
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(
    bymonthyear_df["month"],
    bymonthyear_df["cnt"],
    color = "#32aad9"
)
ax.set_title('Average Number of Users by Month', fontsize=28)
ax.set_xlabel('Month', fontsize=24)
ax.set_xticks(bymonthyear_df["month"])
ax.set_xticklabels(month_names[:len(bymonthyear_df)], ha='center', fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)

st.pyplot(fig)

# Demographics by Season
season_names = ['Spring', 'Summer', 'Fall', 'Winter']
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(
    byseason_df["season"],
    byseason_df["cnt"],
    color = "#32aad9"
)
ax.set_title('Average Number of Users by Season', fontsize=28)
ax.set_xlabel('Season', fontsize=24)
ax.set_xticklabels(season_names, ha='center', fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)

st.pyplot(fig)

# Demographics by Weekday (Day of The Week)
weekday_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(
    byweekday_df["weekday"],
    byweekday_df["cnt"],
    color = "#32aad9"
)
ax.set_title('Average Number of Users by Day of The Week', fontsize=28)
ax.set_xlabel('Day of The Week', fontsize=24)
ax.set_xticks(byweekday_df["weekday"])
ax.set_xticklabels(weekday_names[:len(byweekday_df)], ha='center', fontsize=18)
ax.set_yticklabels(ax.get_yticks(), fontsize=18)

st.pyplot(fig)

# Demographics by Workingday
workingday_name = ['Non-Working Day', 'Working Day']
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(
  byworkingday_df["cnt"],
  labels = workingday_name,
  autopct = '%1.2f%%',
  colors = ["#5fd0db", "#ff8001"],
  startangle = 90,
  textprops = {'fontsize': 14},
)
ax.set_title('Average Percentage of Users by Type of Working Day', fontsize=20, pad=20)

st.pyplot(fig)

# Demographics by User Type
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
  byusertype_df["total"],
  labels = byusertype_df["user_type"],
  autopct = '%1.2f%%',
  colors = ["#5fd0db", "#ff8001"],
  startangle = 90,
  textprops = {'fontsize': 12},
)
ax.set_title('Average Percentage of Users by User Type', fontsize=18, pad=20)

st.pyplot(fig)


# Decomposition of Time Series
st.subheader('Decomposition of Time Series')

# Dekomposisi Time Series untuk kolom 'cnt'
result = seasonal_decompose(main_df['cnt'], model='additive', period=365)  # Misalnya period 365 hari untuk data harian

# Buat plot untuk setiap komponen
fig_observed = go.Figure(data=[go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observed', line=dict(color="#1a569c"))])
fig_observed.update_layout(title='Observed Time Series', xaxis_title='Date', yaxis_title='Observed')
st.plotly_chart(fig_observed)

fig_trend = go.Figure(data=[go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend', line=dict(color="#1a569c"))])
fig_trend.update_layout(title='Trend Component', xaxis_title='Date', yaxis_title='Trend')
st.plotly_chart(fig_trend)

fig_seasonal = go.Figure(data=[go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal', line=dict(color="#1a569c"))])
fig_seasonal.update_layout(title='Seasonal Component', xaxis_title='Date', yaxis_title='Seasonal')
st.plotly_chart(fig_seasonal)

fig_residual = go.Figure(data=[go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual', line=dict(color="#1a569c"))])
fig_residual.update_layout(title='Residual Component', xaxis_title='Date', yaxis_title='Residual')
st.plotly_chart(fig_residual)
