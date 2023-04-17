"""Visualize the distribution of different samples."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

def plot_histogram(sample, title, bins=16, **kwargs):
    """Plot the histogram of a given sample of random values.

    Parameters
    ----------
    sample : pandas.Series
        raw values to build histogram
    title : str
        plot title/header
    bins : int
        number of bins in the histogram
    kwargs : dict 
        any other keyword arguments for plotting (optional)
    """
    # TODO: Plot histogram
    #print(sample)
    
    # TODO: show the plot
    plt.hist(x=sample, bins=bins)
    plt.title(title)
    plt.show()


    return

#Sample
# 0      0.027198
# 1      0.563823
# 2      0.239730
# 3      0.332149
# 4      0.001104
# 5      0.261392
# 6      0.843236
# 7      0.462738
sample = [np.random.uniform(0,1) for i in range(100)]
plot_histogram(sample, "MyTitle")

# def test_run():
#     """Test run plot_histogram() with different samples."""
#     # Load and plot histograms of each sample
#     # Note: Try plotting them one by one if it's taking too long
#     A = pd.read_csv("A.csv", header=None, squeeze=True)
#     plot_histogram(A, title="Sample A")
    
#     B = pd.read_csv("B.csv", header=None, squeeze=True)
#     plot_histogram(B, title="Sample B")
    
#     C = pd.read_csv("C.csv", header=None, squeeze=True)
#     plot_histogram(C, title="Sample C")
    
#     D = pd.read_csv("D.csv", header=None, squeeze=True)
#     plot_histogram(D, title="Sample D")


# if __name__ == '__main__':
#     test_run()
