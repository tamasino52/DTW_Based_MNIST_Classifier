import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from dtaidistance import dtw
from math import sqrt


def DTW_predict(testX, testY, trainX, trainY):
    predictions = np.zeros(len(testY))
    print('\n<--- KNN Matching --->')
    for testSampleIdx, testSample in enumerate(tqdm(testX, desc='Match')):
        minDistance = float('inf')
        for trainSampleIdx, trainSample in enumerate(tqdm(trainX, desc='Search', leave=False)):
            distanceScanX = dtw.distance(testSample[0], trainSample[0])
            distanceScanY = dtw.distance(testSample[1], trainSample[1])
            distanceScan = sqrt(distanceScanX**2 + distanceScanY**2)
            if distanceScan < minDistance:
                minDistance = distanceScan
                predictions[testSampleIdx] = trainY[trainSampleIdx]
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


def main():
    # load train data
    np.random.seed(1)  # seed
    df_train = pd.read_csv("./data/train.csv")  # Loading Dataset
    df_train = df_train.iloc[np.random.permutation(len(df_train))]  # Random permutaion
    print('<--- Train Data Set --->')
    print(df_train.head())

    sample_size = df_train.shape[0]  # Training set size
    validation_size = int(df_train.shape[0]*0.1)  # Validation set size
    print('\n<--- Validation --->')
    print('Total sampple : {}\nValidation sample : {}'.format(sample_size, validation_size))

    # train_x and train_y
    train_x = np.asarray(df_train.iloc[:1000, 1:]).reshape([-1, 28, 28]).astype(np.uint8)
    train_y = np.asarray(df_train.iloc[:1000, 0]).reshape([-1])

    print('\n<--- Train Data Shape --->')
    print('train X : {}\ntrain Y : {}'.format(train_x.shape, train_y.shape))

    # load test data
    df_test = pd.read_csv("./data/test.csv")
    test_x = np.asarray(df_test.iloc[:20, 1:]).reshape([-1, 28, 28]).astype(np.uint8)
    test_y = np.asarray(df_test.iloc[:20, 0]).reshape([-1])

    print('\n<--- Test Data Set --->')
    print(df_test.head())

    print('\n<--- Test Data Set Shape --->')
    print('test X : {}'.format(test_x.shape))

    print('\n<--- Extract Contours from Train Data --->')
    step_train = getSteps(train_x)

    print('\n<--- Extract Contours from Validation Data --->')
    step_validation = getSteps(test_x)

    # Get Validation Prediction
    predictions = DTW_predict(step_validation, test_y, step_train, train_y)

    # Accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == test_y[idx]:
            correct += 1
    print("\nCorrect/Total : {} / {}\nAccuracy : {}".format(correct, len(predictions), correct/len(predictions)))


if __name__ == '__main__':
    main()