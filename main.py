import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from modules.bar import plot_transcode_counts_by_age
from modules.boxplot import plot_boxplot
from modules.dataframe import save_dataframe_as_image
from modules.hist import plot_histograms
from modules.import_data import import_data
from modules.neural_network import neural_net
from modules.random_forest import random_forest
from modules.show_detail import show_detail
from modules.apply import apply_ceiling_and_convert
from sklearn.model_selection import train_test_split

from modules.svm import svm_regression


if __name__ == "__main__":
    # Declare Path
    path = 'data/'
    image_path = 'images/'
    # Declare Type File
    xlsx = '.xlsx'
    png = '.png'
    # Declare File Name
    files = 'DataDemo' + xlsx
    # -----
    data2567, data2566, data2565, data2564 = import_data(path, files, '2567'), import_data(path, files, '2566'), import_data(path, files, '2565'), import_data(path, files, '2564')
    datasets = [
        ("data2567", data2567),
        ("data2566", data2566),
        ("data2565", data2565),
        ("data2564", data2564),
    ]
    show_detail(
        datasets,
        shape=True,
        column=True,
        info=True,
        describe=True,
        is_null=True,
        dtype=True,
    )
    
    # Columns to be plotted
    columns = ['TransCode', 'TransType', 'TransAmount', 'TransFee']
    labels = ['TransCode', 'TransType', 'TransAmount', 'TransFee']
    # Iterate through each dataset
    for label, df in datasets:
        # Save the boxplot
        plot_boxplot(df, columns, labels, filename=f'{image_path}boxplot_{label}.png')

    plot_histograms(
        datasets=datasets,
        column='TransAmount',
        filename=f'{image_path}hist_{datasets[0][0]}_{datasets[1][0]}_{datasets[2][0]}_{datasets[3][0]}{png}'
    )

    # Iterate through each dataset
    for label, df in datasets:
        # Convert Tran_Date to datetime and extract the month and year
        df['Tran_Date'] = pd.to_datetime(df['Tran_Date'], format='%Y%m%d')
        df['YearMonth'] = df['Tran_Date'].dt.to_period('M')

        # Get the unique TransCode counts by month
        transcode_counts = df.groupby(['YearMonth', 'TransCode']).size().unstack(fill_value=0)

        # Save the DataFrame as an image
        save_dataframe_as_image(df=transcode_counts, filename=f'{image_path}transcode_counts_{label}{png}')
    
    # Combine all the data into one DataFrame
    combined_data = pd.concat([data2564, data2565, data2566, data2567])

    # Convert Tran_Date to datetime and extract the month and year
    combined_data['Tran_Date'] = pd.to_datetime(combined_data['Tran_Date'], format='%Y%m%d')
    combined_data['Birth_date'] = pd.to_datetime(combined_data['Birth_date'], format='%Y%m%d')

    # Calculate age at the time of the transaction
    combined_data['Age'] = (combined_data['Tran_Date'] - combined_data['Birth_date']).dt.days // 365

    # Define age ranges, including 51-60 and 61+
    bins = [0, 18, 30, 40, 50, 60, 100]
    labels = ['0-18', '19-30', '31-40', '41-50', '51-60', '61+']
    combined_data['AgeRange'] = pd.cut(combined_data['Age'], bins=bins, labels=labels, right=False)

    # Extract year and month for grouping
    combined_data['YearMonth'] = combined_data['Tran_Date'].dt.to_period('M')

    # Count TransCode occurrences by month and age range, setting observed=False for compatibility
    transcode_counts_by_age = combined_data.groupby(['YearMonth', 'AgeRange', 'TransCode'], observed=False).size().unstack(fill_value=0)

    # Reindex to ensure all age ranges are included, filling with 0 where necessary
    all_age_ranges = pd.Categorical(labels, categories=labels, ordered=True)

    # Fill missing values with 0 to ensure complete data for all age ranges
    transcode_counts_by_age = transcode_counts_by_age.fillna(0)

    # Save the DataFrame as an image
    save_dataframe_as_image(df=transcode_counts_by_age, filename=f'{image_path}transcode_counts_by_age_df{png}', size=[30, 55])
    print(transcode_counts_by_age)

    plot_transcode_counts_by_age(transcode_counts_by_age, filename=f'{image_path}transcode_counts_by_age_bar{png}')

    # -----

    model_rf, model_svm, model_nn= None, None, None
    X_train, X_test, y_train, y_test = None, None, None, None

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

    # Get the dummy column names from training data
    age_range_dummy_cols = X_train.columns[X_train.columns.str.startswith('AgeRange_')]

    # Random Forest
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
    print(predictions_by_age_rf)

    # SVM
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
    print(predictions_by_age_svm)

    # Neural Network
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
    print(predictions_by_age_nn)

    # -----
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

    # Create a table to compare errors
    error_comparison = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE'],
        'Random Forest': [mse_rf, rmse_rf, mae_rf],
        'SVM': [mse_svm, rmse_svm, mae_svm],
        'Neural Network': [mse_nn, rmse_nn, mae_nn],
    })

    # Save the DataFrame as an image
    save_dataframe_as_image(df=error_comparison, filename=f'{image_path}error_comparison{png}', size=[35, 16])

    print("\nError Comparison Table:")
    print(error_comparison)