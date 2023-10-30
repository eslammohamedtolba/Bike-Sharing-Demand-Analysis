# Bike-Sharing-Demand-Analysis

## Overview
This project is focused on building a bike sharing prediction model using various machine learning algorithms. 
The goal is to predict the number of bike rentals based on features like weather conditions, time of day, and more. The dataset used for this project is 'hour.csv'. 

## Prerequisites
Before running the code, make sure you have the following prerequisites installed:
- Python 3.x
- Jupyter Notebook (recommended for interactive development)
- Required Python libraries:
- pandas
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- numpy

## Project Details
Here's an overview of the steps and code used in this project:

1-Importing Required Libraries:
- Import necessary Python libraries such as pandas, matplotlib, seaborn, statsmodels, scikit-learn, and numpy.

2-Loading the Dataset:
- Load the dataset 'hour.csv' using pandas and display its shape, the first five samples, and some statistical information.

3-Data Preprocessing:
- Check for missing values in the dataset and rename some columns for better readability.

4-Data Exploration:
- Explore unique values and visualize distributions of various columns in the dataset.

5-Data Transformation:
- Modify the 'count' column's distribution by taking the natural logarithm to improve model performance.

6-Bivariate Analysis:
- Analyze the relationship between features and the 'count' column, considering factors like weekday, holiday, season, and more.

7-Feature Selection:
- Remove unnecessary columns such as 'instant', 'dteday', and 'year' for modeling.

8-Correlation Analysis:
- Find and visualize the correlation between all dataset features.

9-Data Encoding:
- Convert integer columns into categorical columns and perform one-hot encoding.

10-Data Splitting:
- Split the data into input (X) and label (Y) datasets, and further split them into training and testing sets.

11-Model Building and Evaluation:
- Create a list of machine learning models and train each model on the training data.
- Evaluate the models' performance by measuring the R-squared (R2) score on both the training and testing datasets.

12-Model Comparison:
- Compare the accuracy of different models on both the training and testing datasets.


## Accuracy 
- The RandomForestRegressor and ExtraTreesRegressor models have achieved an accuracy of approximately 83%.

## Contribution
If you wish to contribute to this project, feel free to:
- Improve the model's accuracy by exploring different machine learning algorithms or hyperparameter tuning.
- Enhance data preprocessing and feature engineering techniques.
- Create visualizations for deeper insights into the dataset.
- Add new features to improve prediction accuracy.
- Provide documentation improvements or additional code comments for clarity.
- Your contributions are welcome to make this project even better!

