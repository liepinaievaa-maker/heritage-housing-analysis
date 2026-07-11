# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# Heritage Housing Analysis

- A machine learning project that analyzes residential property data and predicts house sale prices using Linear Regression. The project follows a complete data science workflow, including business understanding, data exploration, data cleaning, feature engineering, model training, and evaluation.

## Project Overview

- This project investigates the factors that influence residential property prices in Ames, Iowa. Using historical housing data, the project explores the relationships between house attributes and sale prices before developing a machine learning model capable of predicting the sale price of a property.

- This project was developed in response to a business scenario in which a client inherited four residential properties in Ames, Iowa. The objective is to understand which property characteristics influence sale price and to build a machine learning model capable of estimating property values.

##  Business Problem

- Buying and selling houses involves many factors that influence property value. Real estate companies and homeowners benefit from understanding which features have the greatest impact on sale prices.

-  The objective of this project is to build a machine learning model capable of predicting house sale prices based on property characteristics while identifying the most influential features affecting price.

##  Project Objectives

- Explore the housing dataset.
- Clean and preprocess the data.
- Handle missing values.
- Engineer new predictive features.
- Train a Linear Regression model.
- Evaluate model performance.
- Predict house sale prices accurately.

##  Project Hypotheses

Before analysing the dataset, the following hypotheses were established:

### Hypothesis 1
Properties with higher overall quality tend to achieve higher sale prices.

**Validation**

Confirmed.

The correlation analysis showed that `OverallQual` has the strongest positive correlation with `SalePrice` (0.79).

---

### Hypothesis 2

Larger living areas contribute to higher sale prices.

**Validation**

Confirmed.

`GrLivArea` was identified as one of the strongest predictors with a correlation coefficient of approximately 0.71.

---

### Hypothesis 3

Feature engineering improves predictive performance.

**Validation**

Partially confirmed.

The engineered features improved the dataset used for modelling and contributed to an R² score of 0.8477.


## Exploratory Data Analysis

The exploratory analysis identified several important insights.

- OverallQual showed the strongest positive correlation with SalePrice.
- GrLivArea was the second strongest predictor.
- SalePrice follows a positively skewed distribution.
- Several garage and basement variables required missing-value treatment before modelling.

## Hypothesis and how to validate?

### Hypothesis Validation

The hypotheses defined at the beginning of the project were evaluated using exploratory data analysis and model performance.

| Hypothesis | Result |
|------------|--------|
| Higher overall quality leads to higher sale prices. | Confirmed |
| Larger living areas are associated with higher sale prices. | Confirmed |
| Feature engineering improves the predictive model. | Partially confirmed |

## Dataset Content


- This project uses the Ames Housing Dataset, publicly available on from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data).

- The dataset contains 1,460 residential properties from Ames, Iowa, with 29 selected features after data cleaning and feature engineering

- The target variable is `SalePrice`.

Key variables include:

- OverallQual
- GrLivArea
- TotalBsmtSF
- GarageArea
- YearBuilt
- LotArea
 
 | Variable | Description |
|----------|-------------|
| SalePrice | Target variable representing the final sale price of the property. |
| OverallQual | Overall material and finish quality of the house. |
| GrLivArea | Above-ground living area in square feet. |
| TotalBsmtSF | Total basement area in square feet. |
| GarageArea | Garage size in square feet. |
| YearBuilt | Original construction year of the property. |
| LotArea | Lot size in square feet. |
| KitchenQual | Kitchen quality rating. |
| BedroomAbvGr | Number of bedrooms above ground. |
| YearRemodAdd | Year of the most recent renovation or remodel. |

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Mapping Business Requirements to the Solution

### Business Requirement 1

Identify which house characteristics have the strongest relationship with sale price.

**Approach**

- Performed exploratory data analysis.
- Generated correlation heatmaps.
- Analysed feature correlations with SalePrice.
- Visualised the distribution of sale prices.

### Business Requirement 2

Predict the sale price of inherited houses and other properties.

**Approach**

- Cleaned and prepared the dataset.
- Engineered additional predictive features.
- Trained and evaluated a Linear Regression model.

## ML Business Case

- The business objective of this project is to support the client in estimating realistic sale prices for residential properties located in Ames, Iowa.

- This problem is formulated as a **supervised machine learning regression task**, where the model learns the relationship between house characteristics and the corresponding sale price from historical data.

### Inputs

The model uses property characteristics such as:

* Overall quality
* Ground living area
* Basement area
* Garage size
* Lot size
* Year built
* Porch and deck areas
* Kitchen quality
* Additional engineered features created during data preparation

### Target

The target variable is **SalePrice**, representing the final selling price of each property.

### Success Criteria

The project was considered successful if the predictive model achieved an R² score greater than 0.75 on unseen data.

The final Linear Regression model achieved an **R² score of 0.8477**, exceeding the agreed business requirement and demonstrating that the selected features provide a reliable basis for predicting residential property prices.


## Dashboard Design


## Unfixed Bugs

At the time of submission, no critical bugs affecting the functionality of the notebooks were identified.

Future improvements include:

- Comparing additional machine learning models.
- Expanding the Streamlit dashboard with more interactive visualisations.
- Improving feature engineering with additional domain-specific variables.

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the .python-version Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

### Dataset

- Code Institute Heritage Housing dataset (hosted on Kaggle)

### Learning Resources

- Code Institute Predictive Analytics walkthrough materials
- Scikit-learn documentation
- Pandas documentation
- Matplotlib documentation
- Seaborn documentation

### Acknowledgements

Thanks to the Code Institute Predictive Analytics course for providing the project brief and supporting learning materials.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)


* In case you would like to thank the people that provided support through this project.

