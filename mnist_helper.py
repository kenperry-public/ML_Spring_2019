# %load mnist_1.py
import time
import matplotlib.pyplot as plt
import numpy as np

import os

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sklearn import datasets, svm, metrics

import os


class MNIST_Helper():
    def __init__(self, **params):
        self.X, self.y = None, None
        return

    def fetch_mnist_784(self):
        # The fetch from the remote site is SLOW b/c the data is so big
        # Try getting it from a local cache
        cache_dir = "cache/mnist_784"
        (X_file, y_file) = [ "{c}/{f}.npy".format(c=cache_dir, f=fn) for fn in ["X", "y"] ]

        if os.path.isfile(X_file) and os.path.isfile(y_file):
            print("Retrieving MNIST_784 from cache")
            X = np.load(X_file)
            y = np.load(y_file)
        else:
            print("Retrieving MNIST_784 from remote")
            # Load data from hiittps://www.openml.org/d/554
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

            # Cache it !
            os.makedirs(cache_dr, exist_ok=True)
            np.save(X_file, X)
            np.save(y_file, y)

        self.X, self.y = X, y
        return X,y

    def setup(self):
        # Derived from: https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html

        # Author: Arthur Mensch <arthur.mensch@m4x.org>
        # License: BSD 3 clause

        # Turn down for faster convergence
        train_samples = 5000

        # Fetch the data
        if self.X is not None and self.y is not None:
            X, y = self.X, self.y

        else:
            X, y = self.fetch_mnist_784()

        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X = X.reshape((X.shape[0], -1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_samples, test_size=10000)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train, self.y_train, self.X_test, self.y_test, self.scaler = \
        X_train, y_train, X_test, y_test, scaler

    def visualize(self, X=None, y=None):
        if X is None:
            X = self.X_train

        # %load mnist_vis_train.py
        fig = plt.figure(figsize=(10,10))
        (num_rows, num_cols) = (5, 5)
        for i in range(0, num_rows * num_cols):
            img = X[i].reshape(28, 28)

            ax  = fig.add_subplot(num_rows, num_cols, i+1)
            _ = ax.set_axis_off()

            _ = plt.imshow(img, cmap="gray")

    def fit(self):
        X_train, y_train = self.X_train, self.y_train

        train_samples = X_train.shape[0]

        # Turn up tolerance for faster convergence
        clf = LogisticRegression(C=50. / train_samples,  # n.b. C is 1/(regularization penalty)
                                 multi_class='multinomial',
                                 # penalty='l1',   # n.b., "l1" loss: sparsity (number of non-zero) >> "l2" loss (dafault)
                                 solver='saga', tol=0.1)

        t0 = time.time()

        # Fit the model
        clf.fit(X_train, y_train)

        run_time = time.time() - t0
        print('Example run in %.3f s' % run_time)

        self.clf = clf
        return clf

    def plot_coeff(self):
        clf = self.clf

        fig = plt.figure(figsize=(10, 8))
        coef = clf.coef_.copy()

        (num_rows, num_cols) = (2,5)

        scale = np.abs(coef).max()
        for i in range(10):
            ax = fig.add_subplot(num_rows, num_cols, i+1)

            # Show the coefficients for digit i
            # Reshape it from (784,) to (28, 28) so can interpret it
            _ = ax.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                           cmap="gray", #plt.cm.RdBu, 
                           vmin=-scale, vmax=scale)

            _ = ax.set_xticks(())
            _ = ax.set_yticks(())
            _ = ax.set_xlabel('Class %i' % i)

        _ =fig.suptitle('Classification vector for...')


        _ = fig.show()
        return fig, ax



        
