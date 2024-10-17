import numpy as np
import pandas as pd

def apply_ceiling_and_convert(predictions_df):
    """
    Applies ceiling to each value in the DataFrame and converts them to integers.
    """
    for column in predictions_df.columns:
        # Apply ceiling and convert to integer
        predictions_df.loc[predictions_df[column] < 0, column] = 0
        predictions_df.loc[:, column] = predictions_df[column].apply(np.ceil)
        predictions_df[column] = predictions_df[column].astype(int)

    return predictions_df