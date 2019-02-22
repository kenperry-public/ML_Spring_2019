X_train.shape, y_train.shape

# Turn up tolerance for faster convergence
mnist_lr_clf = LogisticRegression(C=50. / train_samples,  # n.b. C is 1/(regularization penalty)
                         multi_class='multinomial',
                         # penalty='l1',   # n.b., "l1" loss: sparsity (number of non-zero) >> "l2" loss (dafault)
                         solver='saga', tol=0.1)

t0 = time.time()

# Fit the model
mnist_lr_clf.fit(X_train, y_train)

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
