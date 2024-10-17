import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from modules.time_measure import time_measure


@time_measure
def svm_regression(transcode_counts_by_age, labels, target_columns, model_svm, X_train, X_test, y_train, y_test):
  # Prepare the data for the model
  data = transcode_counts_by_age.reset_index()
  data['YearMonth'] = data['YearMonth'].astype(str)

  # Convert categorical variables to dummy/indicator variables
  X = pd.get_dummies(data[['AgeRange']], drop_first=True)  # Only AgeRange for now
  X = pd.concat([X, data[transcode_counts_by_age.columns]], axis=1) # Add TransCode columns

  # Create the target variable (e.g., predicting counts for a specific TransCode)
  target_columns = transcode_counts_by_age.columns  # All TransCodes as targets
  y = X[target_columns]

  # Ensure column names are strings
  X.columns = X.columns.astype(str)  # Convert all column names to strings
  y.columns = y.columns.astype(str)  # Convert all column names to strings

  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and train the SVM model (using SVR for regression)
  model_svm = MultiOutputRegressor(SVR(kernel='rbf'))  # You can try different kernels like 'linear', 'poly'
  model_svm.fit(X_train, y_train)

  # Prepare data for prediction in 2024 OCT
  future_year_months = pd.date_range(start='2024-10-01', end='2024-10-31', freq='ME').to_period('M')
  age_ranges = labels

  # Create DataFrame for future predictions
  future_data = []
  for month in future_year_months:
      for age_range in age_ranges:
          future_data.append({'YearMonth': month, 'AgeRange': age_range})
  future_data = pd.DataFrame(future_data)

  # Convert categorical variables to dummy/indicator variables
  future_X = pd.get_dummies(future_data[['AgeRange']], drop_first=True)
  # Get the dummy column names from training data
  age_range_dummy_cols = X_train.columns[X_train.columns.str.startswith('AgeRange_')]
  # Align with training data columns (excluding TransCode columns which are targets)
  future_X = future_X.reindex(columns=age_range_dummy_cols, fill_value=0)

  # Add TransCode columns to future_X, initializing with 0 and ensuring string column names
  for transcode in transcode_counts_by_age.columns:
      future_X[str(transcode)] = 0  # Convert transcode to string for column name

  # Make predictions for 2024 OCT
  future_predictions = model_svm.predict(future_X)

  # Create DataFrame for predictions
  predictions_df = pd.DataFrame(future_predictions, columns=target_columns)
  predictions_df = pd.concat([future_data, predictions_df], axis=1)

  # Reshape to match transcode_counts_by_age format
  predictions_by_age = predictions_df.groupby(['YearMonth', 'AgeRange'])[target_columns].sum().unstack(fill_value=0)

  # Display the predictions
  return predictions_by_age, model_svm