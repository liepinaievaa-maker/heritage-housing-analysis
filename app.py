import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

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
        f"""
        This dashboard presents the results of a predictive analytics project
        based on residential property data from Ames, Iowa.

        The project investigates the property characteristics associated with
        house sale prices and uses a **{model_name}** regression model to
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

        - **Living Area:** Supported. Larger above-ground living areas were
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
        """
        This page presents the key findings from the exploratory data analysis
        carried out before developing the machine learning model.

        The visualisations below highlight important relationships between
        property characteristics and house sale prices, helping to evaluate the
        project hypotheses and identify the features
        most useful for prediction.
        """
    )

    st.subheader("Processed Dataset")

    st.write(
        """
        The table below shows a preview of the cleaned and engineered dataset
        used for model development.
        """
    )

    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    col1.metric("Properties", f"{df.shape[0]:,}")
    col2.metric("Variables", df.shape[1])

    st.subheader("Distribution of House Sale Prices")

    st.write(
        """
        This histogram shows how house sale prices are distributed across the
        dataset. It helps identify the most common price ranges and whether the
        dataset contains unusually expensive properties.
        """
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["SalePrice"], bins=30)

    ax.set_xlabel("Sale Price")
    ax.set_ylabel("Number of Houses")
    ax.set_title("Distribution of Sale Prices")
    ax.xaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        Most properties are concentrated in the lower and middle price ranges,
        while a smaller number of properties have much higher sale prices.

        This indicates a right-skewed distribution, where a small number of
        high-value properties extend the distribution towards larger prices.
        """
    )

    st.subheader("Correlation Heatmap")

    st.write(
        """
        Correlation measures the strength of the relationship between two
        numerical variables.

        Correlation values range from -1 to 1:

        - Values close to **1** indicate a strong positive relationship.
        - Values close to **-1** indicate a strong negative relationship.
        - Values close to **0** indicate little or no linear relationship.

        The heatmap below focuses on the numerical features that are most
        strongly related to house sale price.
        """
    )

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
        """
        The heatmap shows that several numerical variables have strong positive
        relationships with house sale price.

        Overall quality, living area and garage-related features appear among
        the strongest predictors, supporting their inclusion in the final
        machine learning model.
        """
    )

    st.subheader("Top Features Correlated with Sale Price")

    st.write(
        """
        This bar chart ranks the numerical features according to the strength
        of their positive correlation with house sale price.

        Features with larger correlation values tend to have a stronger
        relationship with the target variable and may be more useful for
        prediction.
        """
    )

    corr = numeric_df.corr()["SalePrice"].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    corr.drop("SalePrice").head(10).plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Correlation with Sale Price")
    ax.set_title("Top 10 Numerical Features Correlated with Sale Price")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        Overall quality has the strongest observed numerical correlation with
        sale price.

        Living-area, garage and total-area features also show meaningful
        positive relationships, suggesting that quality and usable space are
        important factors in determining property value.
        """
    )

    st.subheader("Relationship Between Overall Quality and Sale Price")

    st.write(
        """
        `OverallQual` is a rating from 1 (very poor) to 10 (excellent) that
        describes the overall material and finish quality of a property.

        The boxplot below compares the distribution of house sale prices for
        each quality rating.
        """
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="OverallQual",
        y="SalePrice",
        ax=ax
    )

    ax.set_xlabel("Overall Quality Rating")
    ax.set_ylabel("Sale Price")
    ax.set_title("Sale Price Distribution by Overall Quality")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        Sale prices generally increase as the overall quality rating becomes
        higher.

        Higher-quality properties consistently achieve higher sale prices,
        supporting the project's hypothesis that overall quality is one of the
        strongest predictors of property value.
        """
    )

    st.subheader("Relationship Between Living Area and Sale Price")

    st.write(
        """
        `GrLivArea` represents the above-ground living area of a property in
        square feet.

        The scatter plot below shows how house sale price changes as the amount
        of living space increases.
        """
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df["GrLivArea"],
        df["SalePrice"],
        alpha=0.5
    )

    ax.set_xlabel("Above-Ground Living Area (sq ft)")
    ax.set_ylabel("Sale Price")
    ax.set_title("Living Area vs Sale Price")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        Properties with larger above-ground living
        areas generally achieve higher sale prices.

        The relationship is positive, although there
        is some variation between properties of similar size.
        This suggests that living area is important,
        but other characteristics such as quality,
        age and location also influence
        the final sale price.
        """
    )

    st.subheader("Relationship Between Year Built and Sale Price")

    st.write(
        """
        The construction year provides an indication of the age of a property.

        The scatter plot below explores whether newer houses tend to achieve
        higher sale prices than older properties.
        """
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        df["YearBuilt"],
        df["SalePrice"],
        alpha=0.5
    )

    ax.set_xlabel("Year Built")
    ax.set_ylabel("Sale Price")
    ax.set_title("Year Built vs Sale Price")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        """
        Newer properties generally achieve higher sale prices than older
        properties.

        However, the relationship is weaker than for overall quality or living
        area, indicating that the age of a property is only one of several
        factors that influence its market value.
        """
    )

    st.subheader("Exploration Summary")

    st.write(
        """
        The exploratory data analysis identified several important patterns
        within the Ames Housing dataset that informed the development of the
        predictive model.
        """
    )

    st.markdown(
        """
        - **Overall quality** shows the strongest relationship with house sale
        price.
        - **Living area** is positively associated with property value, with
        larger homes generally selling for higher prices.
        - **Year built** has a positive relationship with sale price, although
        its influence is weaker than quality and living area.
        - **Garage-related** and **total-area** features also contribute useful
        information for predicting sale price.
        - The distribution of sale prices is **right-skewed**, with relatively
        few high-value properties compared with the majority of homes.
        """
    )

    st.success(
        """
        Overall, the exploratory analysis supports the project's hypotheses and
        confirms that property quality, size, age and garage
        characteristics are
        valuable predictors for estimating house sale prices.
        """
    )

elif page == "Model Performance":
    st.title("Model Performance")

    st.write(
        f"""
        This page presents the evaluation results of the final
        **{model_name}** regression model.

        The model was assessed using both training data and previously unseen
        test data to determine how accurately it predicts house sale prices
        and how well it generalises to new properties.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training R²", "0.9633")
    col2.metric("Test R²", "0.8892")
    col3.metric("Mean Absolute Error", "$17,953.97")
    col4.metric("Root Mean Squared Error", "$29,149.13")

    st.caption(
        """
        **Training R²** shows how well the model fits the training data.

        **Test R²** measures how well the model predicts previously
        unseen data.

        **Mean Absolute Error (MAE)** represents the average difference between
        predicted and actual sale prices.

        **Root Mean Squared Error (RMSE)** gives greater weight to larger
        prediction errors and provides an indication of the model's overall
        prediction accuracy.
        """
    )

    st.subheader("Model Interpretation")

    st.write(
        f"""
        The selected **{model_name}** model achieved a test R² score of
        **0.8892**, meaning that it explains approximately **88.9% of the
        variation in house sale prices** within the unseen test data.

        The model's Mean Absolute Error of **$17,953.97** indicates that its
        predicted sale prices differ from the actual sale
        prices by approximately **$17,954 on average**.

        The Root Mean Squared Error of **$29,149.13** is higher than the MAE
        because it gives greater weight to larger prediction errors.
        This suggests that although most predictions
        are reasonably accurate, some properties
        are more difficult to predict.
        """
    )

    st.subheader("Generalisation Performance")

    st.info(
        """
        The training R² score of **0.9633** is higher than the test R² score of
        **0.8892**, showing that the model performs better on the data it was
        trained on.

        This difference suggests a small amount of overfitting. However, the
        cross-validation R² score of **0.8607** remains
        strong and indicates that the model performs consistently
        across different subsets of the dataset.

        Overall, the model demonstrates good generalisation performance and is
        suitable for producing house-price estimates on unseen property data.
        """
    )

    st.subheader("Model Selection")

    st.write(
        """
        Several regression algorithms were evaluated during
        model development to identify the model that provided
        the most accurate and reliable house price predictions.

        The models compared were:

        - **Linear Regression**
        - **Ridge Regression**
        - **Random Forest Regression**
        - **Gradient Boosting Regression**
        """
    )

    st.info(
        f"""
        **{model_name}** achieved the strongest overall
        performance by combining:

        - the highest test R² score,
        - the lowest prediction errors,
        - and strong cross-validation performance.

        Based on these evaluation results, it was selected as the final model
        deployed within this dashboard.
        """
    )

elif page == "House Price Prediction":
    st.title("House Price Prediction")

    st.write(
        f"""
        This page allows users to generate an estimated sale price for a
        residential property in Ames, Iowa, using the trained
        **{model_name}** regression model.

        Enter the available property characteristics below.
        The application will combine these values with representative
        values for the remaining model
        features before generating the prediction.
        """
    )

    st.subheader("How to Use")

    st.markdown(
        """
        1. Enter the available property characteristics in the form below.
        2. Review the values before generating the estimate.
        3. Click **Predict Sale Price**.
        4. The application will display the estimated property value.

        Only the most influential features are entered manually. Features that
        are not included in the form are automatically assigned representative
        values from the training dataset.
        """
    )

    st.subheader("Property Characteristics")

    st.write(
        """
        Enter the known characteristics of the property below. These variables
        were selected because they have a strong influence on house sale prices
        and are used by the prediction model to estimate the property's value.
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

        st.subheader("Prediction Result")

        st.success(
            f"Estimated Sale Price: **${prediction:,.0f}**"
        )

        st.write(
            f"""
            The estimated sale price was generated using the trained
            **{model_name}** regression model and the property characteristics
            entered above.
            """
        )

        st.info(
            """
            Features that were not entered manually were
            assigned representative
            median values from the training dataset.

            As a result, the prediction should be
            interpreted as an estimate rather
            than an exact market valuation.
            """
        )

        st.caption(
            """
            This prediction is based on historical housing
            data from Ames, Iowa.
            It should not be used as a substitute for a professional property
            valuation.
            """
        )
