# Plot the coefficients (length 784) for each of the 10 digits
print("Coefficients shape {s}, range from {mn:.2f} to {mx:.2f}".format(s=mnist_lr_clf.coef_.shape, mn= mnist_lr_clf.coef_.min(), mx=mnist_lr_clf.coef_.max()) )

def plot_coeff():
    fig = plt.figure(figsize=(10, 5))
    coef = mnist_lr_clf.coef_.copy()


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

    fig.suptitle('Classification vector for...')

    
    _ = fig.show()
