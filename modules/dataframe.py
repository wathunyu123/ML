# Function to save a DataFrame as an image using matplotlib
from matplotlib import pyplot as plt

def save_dataframe_as_image(df, filename, size=None):
    if size is None:
        plt.figure(figsize=(10, 6))
    else:
        plt.figure(figsize=(size[0], size[1]))
    plt.axis('off')  # Turn off the axis
    plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    print(f"DataFrame saved as {filename}")
    plt.close()