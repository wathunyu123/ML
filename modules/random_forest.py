import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from modules.time_measure import time_measure


@time_measure
def random_forest(transcode_counts_by_age, labels, target_columns, model_rf, X_train, X_test, y_train, y_test):

  # Create and train the Random Forest model
  model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
  model_rf.fit(X_train, y_train)

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
  # Align with training data columns (excluding TransCode columns which are targets)
  future_X = future_X.reindex(columns=X_train.columns[X_train.columns.str.startswith('AgeRange_')], fill_value=0)

  # Add TransCode columns to future_X, initializing with 0 and ensuring string column names
  for transcode in transcode_counts_by_age.columns:
      future_X[str(transcode)] = 0  # Convert transcode to string for column name

  # Make predictions for 2024 OCT
  future_predictions = model_rf.predict(future_X)

  # Create DataFrame for predictions
  predictions_df = pd.DataFrame(future_predictions, columns=target_columns)
  predictions_df = pd.concat([future_data, predictions_df], axis=1)

  # Reshape to match transcode_counts_by_age format
  predictions_by_age = predictions_df.groupby(['YearMonth', 'AgeRange'])[target_columns].sum().unstack(fill_value=0)

  # Display the predictions
  return predictions_by_age, model_rf