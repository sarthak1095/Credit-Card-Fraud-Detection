# Credit Card Fraud Detection

This project aims to build a machine learning model to detect credit card fraud using the Credit Card Fraud dataset from Kaggle. The dataset contains anonymized features and labeled transactions as either fraudulent or non-fraudulent.

## Dataset

The Credit Card Fraud dataset used in this project is sourced from Kaggle:

- Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Data Preprocessing

1. Importing Libraries and Dataset
   - Importing the necessary libraries: numpy, pandas, matplotlib, seaborn
   - Loading the Credit Card Fraud dataset

2. Data Exploration
   - Displaying the first few rows of the dataset
   - Checking the shape and columns of the dataset
   - Examining information about the dataset (data types, non-null values)
   - Generating descriptive statistics of the dataset

3. Dealing with Missing Values
   - Checking for missing values in the dataset
   - Calculating the total number of missing values

4. Encoding Categorical Data
   - Identifying columns with object data type
   - Calculating the number of categorical columns

5. Visualizing Class Distribution
   - Creating a countplot to show the distribution of classes (fraudulent vs. non-fraudulent transactions)

6. Correlation Analysis
   - Creating a correlation matrix and heatmap to visualize feature-target correlations

7. Splitting the Dataset
   - Splitting the data into training and testing sets using train_test_split()

8. Feature Scaling
   - Standardizing the features using the StandardScaler from sklearn.preprocessing

## Model Building

1. Logistic Regression
   - Training a logistic regression classifier
   - Evaluating the model using accuracy, F1 score, precision, and recall
   - Displaying the confusion matrix

2. Random Forest
   - Training a random forest classifier
   - Evaluating the model using accuracy, F1 score, precision, and recall
   - Displaying the confusion matrix

3. XGBoost Classifier
   - Training an XGBoost classifier
   - Evaluating the model using accuracy, F1 score, precision, and recall
   - Displaying the confusion matrix

4. Selecting the Best Model
   - Determining the model with the maximum accuracy
   - Printing the model name and maximum accuracy

## Final Model (XGBoost)

1. Importing the XGBClassifier
2. Training the XGBoost classifier using the entire dataset
3. Making predictions on new data

## Code

The code for this project is implemented in a Jupyter Notebook file named "Credit_card_fraud_detection.ipynb".

## Credits

I would like to give credit to the "Build 10 Real World Machine Learning Projects" by "Vijay Gadhave". The code and project structure in this repository were developed based on the lessons and instructions provided in the course. I highly recommend checking out the course for detailed explanations and in-depth learning.
