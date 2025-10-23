🏠 House Price Prediction using Linear Regression
📘 Overview

This project is part of my Machine Learning & Artificial Intelligence Internship at Cloudcredits Technologies.
The objective of this task is to predict house prices based on various features such as area, number of rooms, crime rate, and other housing characteristics.
We use Linear Regression, one of the most fundamental algorithms in machine learning, to model the relationship between these features and house prices.

🎯 Objective

To build a regression model that can accurately predict the median value of owner-occupied homes (MEDV) based on features from the Boston Housing Dataset.

📊 Dataset

Name: Boston Housing Dataset

Source: UCI Machine Learning Repository / Kaggle

Features:

CRIM — per capita crime rate by town

ZN — proportion of residential land zoned for lots over 25,000 sq.ft.

INDUS — proportion of non-retail business acres per town

CHAS — Charles River dummy variable (1 if tract bounds river; 0 otherwise)

NOX — nitric oxides concentration (parts per 10 million)

RM — average number of rooms per dwelling

AGE — proportion of owner-occupied units built prior to 1940

DIS — weighted distances to employment centers

RAD — index of accessibility to radial highways

TAX — full-value property-tax rate per $10,000

PTRATIO — pupil-teacher ratio by town

B — 1000(Bk - 0.63)² where Bk is the proportion of Black residents by town

LSTAT — percentage of lower-status population

MEDV — median value of owner-occupied homes (target variable)

⚙️ Steps Involved
1️⃣ Data Preprocessing

Handled missing values (if any)

Encoded categorical variables

Standardized features using StandardScaler

2️⃣ Model Building

Split data into 80% training and 20% testing sets

Trained a Linear Regression model using sklearn.linear_model.LinearRegression

3️⃣ Model Evaluation

Metrics used:

Mean Squared Error (MSE): Measures average squared difference between predictions and actual values

R² Score: Indicates how well the model explains the variance in the target variable

4️⃣ Visualization

Plotted Actual vs Predicted Prices using matplotlib
