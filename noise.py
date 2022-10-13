import numpy as np


def salt_and_pepper(X, prop):
    X_clone = X.clone().view(-1, 1)
    num_feature = X_clone.size(0)
    mn = X_clone.min()
    mx = X_clone.max()
    indices = np.random.randint(0, num_feature, int(num_feature * prop))
    for elem in indices:
        if np.random.random() < 0.5:
            X_clone[elem] = mn
        else:
            X_clone[elem] = mx
    return X_clone.view(X.size())