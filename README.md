# 240-674 Machine Learning
## Dataset
- From Prince of Songkla University Cooperative Credit And Saving, Limited
## Author Information
- Name: Wathunyu Phetpaya
- Student Code: 6710120039
- Institution: Prince of Songkla University
- Department: Computer Engineering
- Degree: Master
## Presentation
- Canva: https://www.canva.com/design/DAGT1j1-C84/Bgz2iX7lCsmSbHWmt4AItg/view?utm_content=DAGT1j1-C84&utm_campaign=designshare&utm_medium=link&utm_source=editor
## Environment
- Python Version: 3.12.0
- Library: numpy, pandas, scikit-learn, matplotlib, time, wraps
- pip install -r requirements.txt
## Active
### Windows
#### In cmd.exe
- .venv\bin\activate.bat
#### In PowerShell
- .venv\bin\Activate.ps1
### MacOS / Linux
- source .venv/bin/activate
## DeActive
- deactive
## Data Transformation
The data transformation section is responsible for cleaning, structuring, and preparing the raw data for analysis. This involves converting data types, creating new features, and handling missing values to ensure data consistency and accuracy.
```python
combined_data = pd.concat([data2564, data2565, data2566, data2567])
combined_data['Tran_Date'] = pd.to_datetime(combined_data['Tran_Date'], format='%Y%m%d')
combined_data['Birth_date'] = pd.to_datetime(combined_data['Birth_date'], format='%Y%m%d')
combined_data['Age'] = (combined_data['Tran_Date'] - combined_data['Birth_date']).dt.days // 365

# Define age ranges, including 51-60 and 61+
bins = [0, 18, 30, 40, 50, 60, 100]
labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']
combined_data['AgeRange'] = pd.cut(combined_data['Age'], bins=bins, labels=labels, right=False)

# Extract year and month for grouping
combined_data['YearMonth'] = combined_data['Tran_Date'].dt.to_period('M')
transcode_counts_by_age = combined_data.groupby(['YearMonth', 'AgeRange', 'TransCode'], observed=False).size().unstack(fill_value=0)
all_age_ranges = pd.Categorical(labels, categories=labels, ordered=True)
transcode_counts_by_age = transcode_counts_by_age.fillna(0)
```
#### Key Steps:
1. Data Concatenation: The code combines multiple datasets (data2564, data2565, data2566, data2567) into a single DataFrame named combined_data. This allows for unified analysis across different data sources.
2. Date Conversion: The Tran_Date and Birth_date columns are converted to datetime format using pd.to_datetime. This enables date-based calculations and analysis.
3. Age Calculation: The Age column is calculated by subtracting the birthdate from the transaction date and dividing by 365.25 (to account for leap years). This provides the age of the individual at the time of the transaction.
4. Age Range Categorization: Age ranges are defined using pd.cut to categorize individuals into groups based on their age. This creates a new column AgeRange with categories such as '0-18', '19-30', '31-40', etc.
5. Year-Month Extraction: The YearMonth column is extracted from the Tran_Date to group data by month and year. This allows for time-series analysis and trend identification.
6. Grouping and Aggregation: The data is grouped by YearMonth, AgeRange, and TransCode using groupby. This creates a hierarchical grouping structure. The size() function is used to count the number of occurrences within each group, providing transaction counts for different age ranges, transaction codes, and time periods.
7. Missing Value Handling: The fillna(0) method is used to replace missing values in the grouped DataFrame with 0. This ensures that all groups have a count, even if there are no transactions for a particular combination of YearMonth, AgeRange, and TransCode.
#### Conclusion
> [!NOTE]
> The data transformation process ensures that the data is in a suitable format for subsequent analysis and modeling. By cleaning, structuring, and creating relevant features, the data becomes more informative and valuable for answering research questions or making data-driven decisions.
## Data Preparation
The data preparation section focuses on transforming the data into a suitable format for machine learning modeling. This involves encoding categorical variables, creating a feature matrix and target vector, and splitting the data into training and testing sets.
```python
model_rf, model_svm, model_nn= None, None, None
X_train, X_test, y_train, y_test = None, None, None, None

data = transcode_counts_by_age.reset_index()
data['YearMonth'] = data['YearMonth'].astype(str)

# Convert categorical variables to indicator variables
X = pd.get_dummies(data[['AgeRange']], drop_first=True)  # Only AgeRange for now
X = pd.concat([X, data[transcode_counts_by_age.columns]], axis=1) # Add TransCode columns

target_columns = transcode_counts_by_age.columns  # All TransCodes as targets
y = X[target_columns]

X.columns = X.columns.astype(str)
y.columns = y.columns.astype(str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
#### Key Steps:
1. Data Resetting: The reset_index() method is applied to the transcode_counts_by_age DataFrame to ensure that the index is reset to a simple integer index, which is often required for subsequent operations.
2. Data Type Conversion: The YearMonth column is converted to a string data type using astype(str). This is necessary for certain machine learning algorithms that require categorical features to be represented as strings.
3. One-Hot Encoding: Categorical variables, such as AgeRange, are converted into numerical representations using one-hot encoding. This creates new binary columns for each category, indicating the presence or absence of that category. The pd.get_dummies function is used for this purpose, with the drop_first=True parameter to avoid redundant columns.
4. Feature Matrix Creation: The encoded categorical variables and the original transaction count columns (from transcode_counts_by_age) are combined into a single feature matrix X. This matrix represents the input features that will be used to predict the target variables.
5. Target Vector Creation: The target variables are extracted from the transcode_counts_by_age DataFrame and stored in the y variable. This represents the output that the model will learn to predict.
6. Data Splitting: The data is divided into training and testing sets using train_test_split. This allows for evaluating the model's performance on unseen data and preventing overfitting. The test_size parameter specifies the proportion of data allocated to the testing set (20% in this case).
#### Conclusion
> [!NOTE]
> The data preparation process ensures that the data is in a format that is compatible with machine learning algorithms. By encoding categorical variables, creating features and targets, and splitting the data, the dataset becomes ready for training and evaluation of predictive models.
## Data Training
### Random Forest (RF)
The data training section focuses on building and training a Random Forest model to predict the target variables using the prepared features.
```python
predictions_by_age_rf, model_rf = random_forest(
    transcode_counts_by_age=transcode_counts_by_age,
    labels=labels,
    target_columns=target_columns,
    model_rf=model_rf,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
predictions_by_age_rf = apply_ceiling_and_convert(predictions_df=predictions_by_age_rf)
```
#### Key Steps:
1. Function Call: The random_forest function is called to create and train the Random Forest model. This function takes several inputs:
    * transcode_counts_by_age: The DataFrame containing the transaction counts for different age ranges, transaction codes, and time periods.
    * labels: The labels or categories associated with the target variables.
    * target_columns: The columns representing the target variables that the model will predict.
    * model_rf: The Random Forest model object itself.
    * X_train: The training set feature matrix containing the input features.
    * X_test: The testing set feature matrix containing the input features.
    * y_train: The training set target vector containing the true values of the target variables.
    * y_test: The testing set target vector containing the true values of the target variables.
2. Model Training: The random_forest function trains the Random Forest model using the provided training data (X_train and y_train). This involves constructing multiple decision trees and combining their predictions to improve accuracy and reduce overfitting.
3. Predictions: Once the model is trained, it is used to make predictions on the testing set (X_test). The predicted values are stored in the predictions_by_age_rf variable.
4. Post-Processing: The apply_ceiling_and_convert function is applied to the predicted values. This step may involve rounding the predictions up to the nearest integer or converting them to a specific data type, depending on the nature of the target variables.
#### Conclusion
> [!NOTE]
> The data training section demonstrates the process of building and training a Random Forest model for predictive modeling. By using the prepared features and target variables, the model learns to identify patterns and relationships in the data, enabling it to make accurate predictions on new, unseen data.
### Support Vector Machine (SVM)
The data training section focuses on building and training a Support Vector Machine (SVM) model to predict the target variables using the prepared features.
```python
predictions_by_age_svm, model_svm = svm_regression(
    transcode_counts_by_age=transcode_counts_by_age,
    labels=labels,
    target_columns=target_columns,
    model_svm=model_svm,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)
predictions_by_age_svm = apply_ceiling_and_convert(predictions_df=predictions_by_age_svm)
```
#### Key Steps:
1. Function Call: The svm_regression function is called to create and train the SVM model. This function takes similar inputs as the random_forest function, including:
    * transcode_counts_by_age: The DataFrame containing the transaction counts for different age ranges, transaction codes, and time periods.
    * labels: The labels or categories associated with the target variables.
    * target_columns: The columns representing the target variables that the model will predict.
    * model_svm: The SVM model object itself.
    * X_train: The training set feature matrix containing the input features.
    * X_test: The testing set feature matrix containing the input features.
    * y_train: The training set target vector containing the true values of the target variables.
    * y_test: The testing set target vector containing the true values of the target variables.
2. Model Training: The svm_regression function trains the SVM model using the provided training data (X_train and y_train). SVM models find a hyperplane that separates the data into different classes or predicts a continuous value based on the input features.
3. Predictions: Once the model is trained, it is used to make predictions on the testing set (X_test). The predicted values are stored in the predictions_by_age_svm variable.
4. Post-Processing: The apply_ceiling_and_convert function is applied to the predicted values. This step may involve rounding the predictions up to the nearest integer or converting them to a specific data type, depending on the nature of the target variables.
#### Conclusion
> [!NOTE]
> The data training section demonstrates the process of building and training an SVM model for predictive modeling. By using the prepared features and target variables, the SVM model learns to identify patterns and relationships in the data, enabling it to make accurate predictions on new, unseen data.
### Neural Network (NN)
The data training section focuses on building and training a Neural Network (NN) model to predict the target variables using the prepared features.
```python
predictions_by_age_nn, model_nn = neural_net(
    transcode_counts_by_age=transcode_counts_by_age,
    labels=labels,
    target_columns=target_columns,
    model_nn=model_nn,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    epochs=10000,
)
predictions_by_age_nn = apply_ceiling_and_convert(predictions_df=predictions_by_age_nn)
```
#### Key Steps:
1. Function Call: The neural_net function is called to create and train the Neural Network model. This function takes similar inputs as the previous models, including:
    * transcode_counts_by_age: The DataFrame containing the transaction counts for different age ranges, transaction codes, and time periods.
    * labels: The labels or categories associated with the target variables.
    * target_columns: The columns representing the target variables that the model will predict.
    * model_nn: The Neural Network model object itself.
    * X_train: The training set feature matrix containing the input features.
    * X_test: The testing set feature matrix containing the input features.
    * y_train: The training set target vector containing the true values of the target variables.
    * y_test: The testing set target vector containing the true values of the target variables.
    * epochs: The number of training iterations or epochs to be performed.
2. Model Training: The neural_net function trains the Neural Network model using the provided training data (X_train and y_train). Neural Networks are composed of interconnected layers of neurons that learn complex patterns and relationships in the data. The training process involves adjusting the weights and biases of these neurons to minimize the error between the predicted and actual values.
3. Predictions: Once the model is trained, it is used to make predictions on the testing set (X_test). The predicted values are stored in the predictions_by_age_nn variable.
4. Post-Processing: The apply_ceiling_and_convert function is applied to the predicted values. This step may involve rounding the predictions up to the nearest integer or converting them to a specific data type, depending on the nature of the target variables.
#### Conclusion
> [!NOTE]
> The data training section demonstrates the process of building and training a Neural Network model for predictive modeling. By using the prepared features and target variables, the Neural Network learns to identify complex patterns and relationships in the data, enabling it to make accurate predictions on new, unseen data.
## Error Comparison
```python
# Make predictions on the test set for Random Forest
y_pred_rf = model_rf.predict(X_test)

# Calculate error metrics for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Make predictions on the test set for SVM
y_pred_svm = model_svm.predict(X_test)

# Calculate error metrics for Neural Network
mse_svm = mean_squared_error(y_test, y_pred_svm)
rmse_svm = mean_squared_error(y_test, y_pred_svm)
mae_svm = mean_absolute_error(y_test, y_pred_svm)

# Make predictions on the test set for Neural Network
y_pred_nn = model_nn.predict(X_test.astype(float))

# Calculate error metrics for Neural Network
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = mean_squared_error(y_test, y_pred_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
```
