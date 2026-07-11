import seaborn as sns

import matplotlib.pyplot as plt

import streamlit as st

import pandas as pd

df = pd.read_csv("./outputs/engineered_house_prices.csv")

st.set_page_config(
    page_title="Heritage Housing Analysis",
    page_icon="🏠",
    layout="wide",
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "Project Overview",
        "Business Understanding",
        "Data Exploration",
        "Model Performance",
    ],
)

if page == "Project Overview":
    st.title("🏠 Heritage Housing Analysis")

    st.write("""
    Welcome to the Heritage Housing Analysis dashboard.

    This project explores residential property data from Ames, Iowa,
    identifies the factors that influence house prices,
    and develops a machine learning model to predict sale prices.
    """)

elif page == "Business Understanding":
    st.title("Business Understanding")

    st.subheader("Business Problem")

    st.write("""
    The client inherited four houses in Ames, Iowa and would like to estimate
    their market value using historical housing data.
    """)

    st.subheader("Project Hypotheses")

    st.markdown("""
    - Houses with higher overall quality are expected to have higher sale prices.
    - Larger living areas are expected to increase house prices.
    - Newer houses are expected to sell for higher prices.
    - Houses with garages are expected to have higher sale prices.
    - Feature engineering is expected to improve model performance.
    """)

elif page == "Data Exploration":
    st.title("📊 Data Exploration")

    st.write(
        "The dataset below contains the cleaned and engineered housing data "
        "used to train the machine learning model."
    )

    st.dataframe(df.head(10))

    st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Distribution of House Sale Prices")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(df["SalePrice"], bins=30)

    ax.set_xlabel("Sale Price")
    ax.set_ylabel("Number of Houses")
    ax.set_title("Distribution of Sale Prices")

    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["number"])

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader("Top Features Correlated with Sale Price")

    corr = numeric_df.corr()["SalePrice"].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    corr.drop("SalePrice").head(10).plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Correlation")
    ax.set_title("Top 10 Features Correlated with Sale Price")

    st.pyplot(fig)

elif page == "Model Performance":
    st.title("🤖 Model Performance")

    st.write("""
    Linear Regression Results

    • MAE: 21,242.82

    • RMSE: 34,178.36

    • R² Score: 0.8477
    """)