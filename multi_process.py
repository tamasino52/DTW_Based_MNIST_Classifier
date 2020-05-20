import cv2
import pandas as pd
import numpy as np
from tqdm import *
from dtaidistance import dtw
from math import sqrt
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
import multiprocessing
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


num_core = multiprocessing.cpu_count() - 3


def DTW_predict(testX, trainX, trainY):
    def search_nearest(test, train_x, train_y, predictions, offset):
        for test_idx, test_sample in enumerate(test):
            min_dist = float('inf')
            for train_idx, train_sample in enumerate(train_x):
                dist_x = dtw.distance(test_sample[0], train_sample[0])
                dist_y = dtw.distance(test_sample[1], train_sample[1])
                dist = sqrt(dist_x ** 2 + dist_y ** 2)
                if dist < min_dist:
                    min_dist = dist
                    predictions[offset + test_idx] = train_y[train_idx]
            pbar.update()

    def chunks(l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]
    total = len(testX)
    chunk_size = int(total / num_core)
    slice = chunks(testX, chunk_size)
    predictions = multiprocessing.Array('i', [0]*len(testX))
    pool = ThreadPool(num_core)

    with tqdm(desc='Predict', total=len(testX)) as pbar:
        for idx, s in enumerate(slice):
            offset = idx * chunk_size
            pool.apply_async(search_nearest, (s, trainX, trainY, predictions, offset))
        pool.close()
        pool.join()
    return predictions


def getContour(mat):
    # otsu threshold
    _, mat = cv2.threshold(mat, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # closing morphology operation
    mat = cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    # find contour
    contours, hierarchy = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x_step, y_step = [], []
    for point in contours[np.argmax(len(contours))]:
        x_step.append(point.flatten()[0])
        y_step.append(point.flatten()[1])

    x_step = [100 * float(i) / max(x_step) for i in x_step]
    y_step = [100 * float(i) / max(y_step) for i in y_step]
    return x_step, y_step


def getSteps(dataset):
    step = []
    for idx, mat in enumerate(tqdm(dataset)):
        step_x, step_y = getContour(mat)
        step.append([step_x, step_y])
    return step


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def main():
    num_train = 1000
    num_test = 50
    # load train data
    np.random.seed(1)  # seed
    df_train = pd.read_csv("./data/train.csv")  # Loading Dataset
    df_train = df_train.iloc[np.random.permutation(len(df_train))]  # Random permutaion
    print('<--- Train Data Set --->')
    print(df_train.head())

    # train_x and train_y
    train_x = np.asarray(df_train.iloc[:num_train, 1:]).reshape([-1, 28, 28]).astype(np.uint8)
    train_y = np.asarray(df_train.iloc[:num_train, 0]).reshape([-1])

    print('\n<--- Train Data Shape --->')
    print('train X : {}\ntrain Y : {}'.format(train_x.shape, train_y.shape))

    # load test data
    df_test = pd.read_csv("./data/test.csv")
    test_x = np.asarray(df_test.iloc[:num_test, 1:]).reshape([-1, 28, 28]).astype(np.uint8)
    test_y = np.asarray(df_test.iloc[:num_test, 0]).reshape([-1])

    print('\n<--- Test Data Set --->')
    print(df_test.head())

    print('\n<--- Test Data Set Shape --->')
    print('test X : {}'.format(test_x.shape))

    print('\n<--- Extract Contours from Train Data and Test Data --->')
    step_train = getSteps(train_x)
    step_test = getSteps(test_x)

    # Get Prediction
    print('\n<--- Search Nearest Data --->')
    predictions = DTW_predict(step_test, step_train, train_y)
    predictions = list(predictions)

    # Accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == test_y[idx]:
            correct += 1

    print("\nCorrect/Total : {} / {}\nAccuracy : {}".format(correct, len(predictions), correct/len(predictions)))

    np.set_printoptions(precision=2)
    class_names = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(test_y, predictions, classes=class_names, title='Confusion matrix, without normalization')
    plt.show()


if __name__ == '__main__':
    main()