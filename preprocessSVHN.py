import tensorflow as tf

def preprocess(x_train, x_test):
    x_train = np.rollaxis(x_train)
    x_test = np.rollaxis(x_test)

    y_train[y_train==10]=0
    y_test[y_test==10]=0

    x_train = x_train[:,:,7:24,:]
    x_test = x_test[:,:,7:24,:]

    x_train = x_train/255
    x_test = x_test/255

    return x_train, x_test


