fig = plt.figure(figsize=(10,10))
(num_rows, num_cols) = (5, 5)
for i in range(0, num_rows * num_cols):
    img = X_train[i].reshape(28, 28)
  
    ax  = fig.add_subplot(num_rows, num_cols, i+1)
    _ = ax.set_axis_off()
    
    _ = plt.imshow(img, cmap="gray")
