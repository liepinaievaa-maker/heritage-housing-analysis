import seaborn as sns

import matplotlib.pyplot as plt

import streamlit as st

import pandas as pd

import joblib

df = pd.read_csv("./outputs/engineered_house_prices.csv")
model = joblib.load("./outputs/models/best_regression_model.pkl")
model_features = joblib.load("./outputs/models/model_features.pkl")
model_name = joblib.load("./outputs/models/model_name.pkl")

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

    st.write(
        """
        This dashboard presents the results of a predictive analytics project
        based on residential property data from Ames, Iowa.

        The project investigates the property characteristics associated with
        house sale prices and uses a Gradient Boosting regression model to
        estimate property values.
        """
    )

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Explore the relationships between property characteristics and
          sale price.
        - Test the main project hypotheses using data visualisations.
        - Compare several regression models.
        - Select the strongest-performing model for deployment.
        - Provide an interactive tool for estimating house sale prices.
        """
    )

    st.subheader("Dashboard Pages")

    st.markdown(
        """
        - **Project Overview:** Introduces the project, objectives and
          dashboard structure.
        - **Business Understanding:** Explains the client problem, business
          requirements and project hypotheses.
        - **Data Exploration:** Presents the main patterns and relationships
          identified in the housing data.
        - **Model Performance:** Shows the selected model and its evaluation
          results.
        - **House Price Prediction:** Allows users to enter property details
          and generate an estimated sale price.
        """
    )

    st.info(
        "Use the navigation menu in the sidebar to move between the "
        "dashboard pages."
    )

    st.subheader("Project Dataset")

    st.write(
        """
        The project uses the Ames Housing dataset, containing historical
        residential property information from Ames, Iowa.

        Following data cleaning and feature engineering, the processed dataset
        contains 1,460 observations and 24 variables used for model
        development.
        """
    )

elif page == "Business Understanding":
    st.title("Business Understanding")

    st.subheader("Client Requirement")

    st.write(
        """
        The client has inherited four residential properties in Ames, Iowa,
        and wants to estimate their likely sale values.

        Historical housing data is used to identify the property
        characteristics most strongly associated with sale price and to
        develop a predictive model that can support valuation decisions.
        """
    )

    st.subheader("Business Requirements")

    st.markdown(
        """
        1. Identify the property characteristics that are most strongly
           related to house sale prices.

        2. Develop a machine learning model capable of producing reliable
           sale-price estimates.

        3. Provide an interactive dashboard where users can explore the
           findings and generate property-price predictions.
        """
    )

    st.subheader("Project Hypotheses")

    st.markdown(
        """
        - Houses with higher overall quality are expected to have higher
          sale prices.

        - Houses with larger living areas are expected to have higher
          sale prices.

        - Newer houses are expected to achieve higher sale prices.

        - Properties with larger garages are expected to achieve higher
          sale prices.

        - The selected property features are expected to provide sufficient
          information to build a reliable house-price prediction model.
        """
    )

    st.subheader("Hypothesis Validation")

    st.markdown(
        """
        - **Overall Quality:** Supported. Properties with higher quality
          ratings generally achieved higher sale prices.

        - **Living Area:** Supported. Larger ground-floor living areas were
          positively associated with sale price.

        - **Year Built:** Partially supported. Newer properties generally
          achieved higher prices, although sale price was also influenced by
          quality, size and other characteristics.

        - **Garage Area:** Supported. Properties with larger garages generally
          achieved higher sale prices.

        - **Predictive Feature Set:** Supported. The selected original and
          engineered property features produced strong test and
          cross-validation performance.
        """
    )

    st.info(
        f"The final deployed model is {model_name}, selected after comparing "
        "multiple regression algorithms."
    )

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
    plt.close(fig)

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["number"])

    top_features = (
        numeric_df.corr()["SalePrice"]
        .abs()
        .sort_values(ascending=False)
        .head(12)
        .index
    )

    correlation_matrix = numeric_df[top_features].corr()

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Correlation Between Sale Price and Key Numerical Features")

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "The heatmap shows the strength and direction of relationships between"
        "SalePrice and the numerical features"
        "most strongly associated with it."
        "Values closer to 1 indicate a stronger positive relationship, while"
        "values closer to -1 indicate a stronger negative relationship."
    )

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
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)

    st.info(
        "Newer houses generally achieve higher sale prices, "
        "although other factors also influence value."
    )
    
    st.subheader("Exploration Summary")

    st.markdown(
        """
        The exploratory analysis identified several important findings:

        - Higher overall quality is strongly associated 
        with higher sale prices.
        - Larger above-ground living areas generally 
        correspond to higher prices.
        - Newer properties tend to achieve higher prices, although the
          relationship is influenced by other property characteristics.
        - Garage-related and total-area variables also show meaningful
          relationships with sale price.
        - The target variable contains a smaller number of high-value
          properties, making these properties potentially more difficult to
          predict accurately.
        """
    )

    st.success(
        """
        The analysis supports the use of property quality, size, age and
        garage-related features in the predictive model.
        """
    )

elif page == "Model Performance":
    st.title("Model Performance")

    st.write(
        f"The selected model, **{model_name}**, was evaluated on unseen test data."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training R²", "0.9633")
    col2.metric("Test R²", "0.8892")
    col3.metric("Mean Absolute Error", "$17,953.97")
    col4.metric("Root Mean Squared Error", "$29,149.13")

    st.subheader("Model Interpretation")

    st.write(
        "The selected Gradient Boosting model explains approximately 88.9% "
        "of the variation in sale prices on unseen test data."
        "Its Mean Absolute "
        "Error indicates that predictions differ from actual sale prices by "
        "approximately $17,954 on average."
    )

    st.info(
        "The model performs strongly on unseen test data. The higher training "
        "score indicates some difference between training and test performance, "
        "but the cross-validation R² score of 0.8607 suggests that the model "
        "generalises reliably across different data subsets."
    )

    st.subheader("Model Selection")

    st.write(
        f"""
        Four regression algorithms were evaluated during model development:

        - Linear Regression
        - Ridge Regression
        - Random Forest
        - Gradient Boosting

        {model_name} achieved the strongest overall performance and was
        therefore selected as the final deployed model.
        """
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

    st.subheader("How to Use")

    st.write(
        """
        Enter the available property characteristics below and click
        **Predict Sale Price**.

        Only the most influential variables are entered manually.
        Remaining variables are assigned representative values from the
        training dataset before the prediction is generated.
        """
    )

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

        st.info(
            f"""
            This estimate is generated using the trained {model_name} model.

            Only six property characteristics are entered by the user.
            All remaining features are automatically filled with typical
            values from the training dataset.
            """
        )

        st.caption(
            "Predictions are estimates based on historical housing data "
            "and should not be interpreted as "
            "professional property valuations."
        )
