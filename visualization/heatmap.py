import matplotlib.pyplot as plt


def plot_heatmap(input_mat, filename=None):
    plt.imshow(input_mat, cmap='viridis')

    plt.colorbar()

    plt.title("Sample Matrix Heatmap with Matplotlib")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()