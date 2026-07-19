import seaborn as sns

import matplotlib.pyplot as plt

import streamlit as st

import pandas as pd

import joblib

df = pd.read_csv("./outputs/engineered_house_prices.csv")
model = joblib.load("./outputs/models/best_regression_model.pkl")
model_features = joblib.load("./outputs/models/model_features.pkl")

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
        "House Price Prediction",
    ],
)

if page == "Project Overview":
    st.title("Heritage Housing Analysis")

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
    - Houses with higher overall quality are expected to have
                higher sale prices.
    - Larger living areas are expected to increase house prices.
    - Newer houses are expected to sell for higher prices.
    - Houses with garages are expected to have higher sale prices.
    - Feature engineering is expected to improve model performance.
    """)

    st.subheader("Hypothesis Validation")

    st.markdown("""
    ### Validation Results

    - **Higher Overall Quality:** Supported. Houses with higher quality
                ratings generally achieved higher sale prices.

    - **Larger Living Area:** Supported. GrLivArea was one of the strongest
                features correlated with SalePrice.

    - **Newer Houses:** Partially supported. Newer homes tended
                to sell for more, although other factors also influenced price.

    - **Garage Presence:** Supported. Houses with
                 garages generally sold for higher prices.

    - **Feature Engineering:** Supported. The engineered features
                contributed to a model with an R² score of **0.8477**.
    """)

elif page == "Data Exploration":
    st.title("Data Exploration")

    st.write(
        "The dataset below contains the cleaned and engineered housing data "
        "used to train the machine learning model."
    )

    st.dataframe(df.head(10))

    st.write(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Distribution of House Sale Prices")

    fig, ax = plt.subplots(figsize=(8, 4))

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

    st.subheader("Sale Price by Overall Quality")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="OverallQual",
        y="SalePrice",
        ax=ax
    )

    ax.set_xlabel("Overall Quality")
    ax.set_ylabel("Sale Price")

    st.pyplot(fig)

    st.info(
        "Higher Overall Quality ratings are associated"
        " with higher house sale"
        "prices, supporting one of the project's initial hypotheses."
    )

    st.subheader("Sale Price vs Living Area")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5)

    ax.set_xlabel("Ground Living Area")
    ax.set_ylabel("Sale Price")
    ax.set_title("Living Area vs Sale Price")

    st.pyplot(fig)

    st.info(
        "Homes with larger living areas generally sell for higher prices, "
        "supporting the second hypothesis."
    )

    st.subheader("Sale Price vs Year Built")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df["YearBuilt"], df["SalePrice"], alpha=0.5)

    ax.set_xlabel("Year Built")
    ax.set_ylabel("Sale Price")
    ax.set_title("Year Built vs Sale Price")

    st.pyplot(fig)

    st.info(
        "Newer houses generally achieve higher sale prices, "
        "although other factors also influence value."
    )

elif page == "Model Performance":
    st.title("Model Performance")

    st.write(
        "The Linear Regression model was evaluated on unseen test data."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training R²", "0.8141")
    col2.metric("Test R²", "0.8477")
    col3.metric("Mean Absolute Error", "$21,242.82")
    col4.metric("Root Mean Squared Error", "$34,178.36")

    st.subheader("Model Interpretation")

    st.write(
        "The model explains approximately 84.8% of the variation in house "
        "sale prices. The Mean Absolute Error indicates "
        "that predictions differ "
        "from actual sale prices by approximately 21,243 on average."
    )

    st.info(
        "The similar training and test R² scores suggest that the model "
        "generalises well and does not show strong evidence of overfitting."
    )

elif page == "House Price Prediction":
    st.title("House Price Prediction")

    st.write("""
    Enter the property characteristics below to estimate the predicted
    sale price of a house in Ames, Iowa.

    Only the most influential features are entered by the user. Remaining
    features are automatically assigned typical values from the training
    dataset.
    """)

    col1, col2 = st.columns(2)

    with col1:
        overall_quality = st.slider(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=5
        )

        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=300,
            max_value=6000,
            value=1500
        )

        garage_area = st.number_input(
            "Garage Area (sq ft)",
            min_value=0,
            max_value=1500,
            value=500
        )

    with col2:
        year_built = st.number_input(
            "Year Built",
            min_value=1872,
            max_value=2010,
            value=2000
        )

        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=1000,
            max_value=250000,
            value=9000
        )

        kitchen_quality = st.selectbox(
            "Kitchen Quality",
            ["Poor", "Fair", "Typical", "Good", "Excellent"]
        )

    if st.button("Predict Sale Price"):

        # Encode the same dataset structure used to train the model.
        model_data = pd.get_dummies(
            df.drop(columns=["SalePrice"]),
            drop_first=True
        )

        # Start with typical values for features not entered in the form.
        input_data = model_data.median().to_frame().T

        # Ensure the columns are in exactly the same order as during training.
        input_data = input_data.reindex(
            columns=model_features,
            fill_value=0
        )

        # Replace typical values with the user's entries.
        input_data.loc[0, "OverallQual"] = overall_quality
        input_data.loc[0, "GrLivArea"] = living_area
        input_data.loc[0, "GarageArea"] = garage_area
        input_data.loc[0, "YearBuilt"] = year_built
        input_data.loc[0, "LotArea"] = lot_area

        # Update simple engineered features when they are available.
        if "HasGarage" in input_data.columns:
            input_data.loc[0, "HasGarage"] = int(garage_area > 0)

        # Match the readable options to the dataset category codes.
        kitchen_codes = {
            "Poor": "Po",
            "Fair": "Fa",
            "Typical": "TA",
            "Good": "Gd",
            "Excellent": "Ex",
        }

        selected_code = kitchen_codes[kitchen_quality]

        # Reset all encoded kitchen-quality columns.
        kitchen_columns = [
            column
            for column in input_data.columns
            if column.startswith("KitchenQual_")
        ]

        input_data.loc[0, kitchen_columns] = 0

        # With drop_first=True, one category may not have its own column.
        selected_column = f"KitchenQual_{selected_code}"

        if selected_column in input_data.columns:
            input_data.loc[0, selected_column] = 1

        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Sale Price: ${prediction:,.0f}")

        st.info("""
        This estimate is generated using the trained Linear Regression model.
        Only six property characteristics are entered by the user;
                 all remaining
        features are automatically filled with
                typical values from the training dataset.
        """)

        st.caption(
            "Predictions are estimates based on historical housing data "
            "and should not be interpreted as "
            "professional property valuations."
        )
