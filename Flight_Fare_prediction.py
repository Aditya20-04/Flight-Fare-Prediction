import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

st.set_page_config(layout="wide")
st.title("🌟 Flight Fare Prediction Web App")

warnings.filterwarnings("ignore")

@st.cache_data

def load_data():
    df = pd.read_csv("Clean_Dataset.csv.zip")
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df

# Load dataset
df = load_data()
st.subheader("📃 Dataset Preview")
st.dataframe(df.head())

if st.checkbox("Show data summary"):
    st.write(df.describe())
    st.write("Missing Values:", df.isnull().sum())
    st.write(df.info())

# EDA Plots
st.subheader("📊 Exploratory Data Analysis")

if st.checkbox("Show Scatter Plot: Duration vs Price"):
    fig, ax = plt.subplots()
    sns.scatterplot(x='duration', y='price', data=df, ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Count Plot: Flights per Airline"):
    df1 = df.groupby(['flight', 'airline'], as_index=False).count()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(data=df1, y='airline', palette='hls', ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Pie Chart: Class Distribution"):
    df2 = df.groupby(['airline', 'flight', 'class'], as_index=False).count()
    fig, ax = plt.subplots()
    df2['class'].value_counts().plot.pie(autopct='%.2f', cmap="cool", ax=ax)
    st.pyplot(fig)

# Boxplots
box_cols = ['airline', 'class', 'stops', 'departure_time', 'arrival_time', 'source_city', 'destination_city']
st.subheader("🔍 Boxplots")
for col in box_cols:
    if st.checkbox(f"Show Boxplot: {col} vs Price"):
        fig, ax = plt.subplots()
        sns.boxplot(x=col, y='price', data=df, palette='hls', ax=ax)
        st.pyplot(fig)

# Lineplots
st.subheader("📈 Line Plots")
if st.checkbox("Show Lineplot: Duration vs Price by Class"):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(data=df, x='duration', y='price', hue='class', palette='hls', ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Lineplot: Days Left vs Price"):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(data=df, x='days_left', y='price', palette='hls', ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Lineplot: Days Left vs Price by Airline"):
    fig, ax = plt.subplots(figsize=(20,8))
    sns.lineplot(data=df, x='days_left', y='price', hue='airline', palette='hls', ax=ax)
    st.pyplot(fig)

# Encoding and Splitting
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = le.fit_transform(df_encoded[col])

x = df_encoded.drop('price', axis=1)
y = df_encoded['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scaling
scaler1 = MinMaxScaler()
x_train = scaler1.fit_transform(x_train)
x_test = scaler1.transform(x_test)

scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

# Model Training
st.subheader("🚀 Model Training & Evaluation")
model_option = st.selectbox("Choose Regression Model", ("Linear Regression", "Decision Tree", "Random Forest"))

if st.button("Train Model"):
    if model_option == "Linear Regression":
        model = LinearRegression()
    elif model_option == "Decision Tree":
        model = DecisionTreeRegressor()
    else:
        model = RandomForestRegressor()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    st.success(f"{model_option} trained successfully!")
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    st.write("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    st.write("R² Score:", r2)
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - x.shape[1] - 1)
    st.write("Adjusted R² Score:", round(adj_r2, 6))

    # Heatmap
    st.subheader("🌡 Heatmap of Correlation")
    fig, ax = plt.subplots(figsize=(16,5))
    sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='viridis', ax=ax)
    st.pyplot(fig)

    # Actual vs Predicted
    st.subheader("📊 Actual vs Predicted")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.lineplot(x=range(len(y_test)), y=y_test, label='Actual', ax=ax2)
    sns.lineplot(x=range(len(y_pred)), y=y_pred, label='Predicted', ax=ax2)
    ax2.set_title("Actual vs Predicted Over Index")
    st.pyplot(fig2)
