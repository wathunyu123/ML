import pandas as pd

from modules.time_measure import time_measure

# Import Data
@time_measure
def import_data(path, files, sheet_name):
  return pd.read_excel(path + files, sheet_name)