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
## Data Preparation
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
