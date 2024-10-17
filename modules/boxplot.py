import matplotlib.pyplot as plt

def plot_boxplot(df, columns, labels, filename=None):
    """
    Plots a boxplot for the specified columns of a DataFrame.
    """
    plt.boxplot([df[col] for col in columns], labels=labels)
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Boxplot of Selected Features')

    if filename:
        plt.savefig(filename, format='png', dpi=300)
        print(f"Boxplot saved as {filename}")

    plt.close()
    # plt.show()