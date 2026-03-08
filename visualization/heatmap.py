import matplotlib.pyplot as plt


def plot_heatmap(input_mat, filename=None):
    # 2. Plot the heatmap using imshow
    plt.imshow(input_mat, cmap='viridis')

    # 3. Add a color bar
    plt.colorbar()

    # 4. Add title and display the plot
    plt.title("Sample Matrix Heatmap with Matplotlib")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()