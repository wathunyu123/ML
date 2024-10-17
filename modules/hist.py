import matplotlib.pyplot as plt

def plot_histograms(datasets, column, bins=20, alpha=0.5, xlabel='TransAmount', ylabel='Frequency', title='Distribution of TransAmount across Years', filename=None):
    """
    Plots histograms for the specified column from multiple DataFrames and optionally saves the plot as a PNG file.
    """
    plt.figure(figsize=(12, 6))
    
    for label, df in datasets:
        plt.hist(df[column], bins=bins, alpha=alpha, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if filename:
        plt.savefig(filename, format='png', dpi=300)
        print(f"Histogram saved as {filename}")

    plt.close()
    # plt.show()
