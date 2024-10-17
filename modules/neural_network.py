import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf

from modules.time_measure import time_measure


@time_measure
def neural_net(transcode_counts_by_age, labels, target_columns, model_nn, X_train, X_test, y_train, y_test, epochs=100):

  # Get the columns before scaling
  X_train_columns = X_train.columns

  # Scale the data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # Create the neural network model
  model_nn = tf.keras.models.Sequential([
      tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(len(target_columns))  # Output layer with number of TransCodes
  ])

  # Compile the model
  model_nn.compile(optimizer='adam', loss='mse')  # Mean squared error for regression

  # Train the model
  model_nn.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

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

  # Use X_train_columns instead of X_train.columns
  future_X = future_X.reindex(columns=X_train_columns[X_train_columns.str.startswith('AgeRange_')], fill_value=0)

  # Add TransCode columns to future_X, initializing with 0 and ensuring string column names
  for transcode in transcode_counts_by_age.columns:
      future_X[str(transcode)] = 0  # Convert transcode to string for column name

  # Scale the future data
  future_X_scaled = scaler.transform(future_X) # Scale the future_X DataFrame using the same scaler

  # Make predictions for 2024 using the scaled future data
  future_predictions = model_nn.predict(future_X_scaled)

  # Create DataFrame for predictions
  predictions_df = pd.DataFrame(future_predictions, columns=target_columns)
  predictions_df = pd.concat([future_data, predictions_df], axis=1)

  # Reshape to match transcode_counts_by_age format
  predictions_by_age = predictions_df.groupby(['YearMonth', 'AgeRange'])[target_columns].sum().unstack(fill_value=0)

  # Display the predictions
  return predictions_by_age, model_nn