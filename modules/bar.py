import matplotlib.pyplot as plt

def plot_transcode_counts_by_age(transcode_counts_by_age, filename=None):
    # Plotting
    plt.figure(figsize=(84, 10))
    for transcode in transcode_counts_by_age.columns:
        transcode_counts_by_age[transcode].plot(kind='bar', label=f'TransCode {transcode}', alpha=0.7)
        transcode_counts_by_age[transcode].plot(marker='o', label=f'TransCode {transcode} - Line')

    # Adding labels and legend
    plt.title('TransCode Counts by Age Range and Month')
    plt.xlabel('YearMonth, AgeRange')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.grid(True)

    # Show plot or save it to a file
    if filename:
        plt.savefig(filename, format='png', dpi=300)
        print(f"Plot saved as {filename}")
    
    plt.close()
    # plt.show()