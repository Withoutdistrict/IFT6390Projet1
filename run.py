import os
os.environ['KAGGLE_USERNAME'] = "eugenesanscartier"
os.environ['KAGGLE_KEY'] = "e11d36e43e98fc6fcfe8f9f29a3f41a1"
# {"username":"eugenesanscartier","key":"e11d36e43e98fc6fcfe8f9f29a3f41a1"}

import sys
import time
import numpy
import pandas as pd
numpy.random.seed(1)

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

import matplotlib
# matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
oversample = SMOTE()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC

import subprocess
import joblib


def confusion_matrix(true_labels, pred_labels, n_classes = 2):
    # n_classes = numpy.unique(true_labels).shape[0]
    matrix = numpy.zeros((n_classes, n_classes))

    for (true, pred) in zip(true_labels, pred_labels):
        matrix[int(true), int(pred)] += 1

    return matrix


def compute_accuracy(estimator, x_features, y_labels):
    predict_labels = estimator.predict(x_features)
    c_matrix = confusion_matrix(y_labels, predict_labels, n_classes = 2)
    FP = c_matrix[0, 1] / (c_matrix[0, 1] + c_matrix[1, 1])
    FN = c_matrix[1, 0] / (c_matrix[1, 0] + c_matrix[0, 0])
    FD = c_matrix[0, 1] / (c_matrix[0, 1] + c_matrix[0, 0])
    FO = c_matrix[1, 0] / (c_matrix[1, 0] + c_matrix[1, 1])
    return numpy.sum(numpy.diag(c_matrix)) / numpy.sum(c_matrix), [FP, FN, FD, FO]


def fetch():
    print("fetching dataset!")  # replace this with code to fetch the dataset
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('rikdifos/credit-card-approval-prediction', unzip=True)

    print("Begin: Loading data")
    application = numpy.array(pd.read_csv("application_record.csv"), dtype=str)
    credit = numpy.array(pd.read_csv("credit_record.csv"), dtype=str)
    print("End: Loading data")

    print("Begin: Encoding features label")
    le = LabelEncoder()
    for idx in [1, 2, 3, 6, 7, 8, 9, 16]:
        application[:, idx] = le.fit_transform(application[:, idx])
    idx_18 = numpy.where(application[:, 16] == '18')[0]
    application[idx_18, 16] = numpy.array(numpy.random.randint(0, 18, len(idx_18)), dtype=str)

    credit[numpy.where(credit[:, 2] == '5')[0], 2] = 7
    credit[numpy.where(credit[:, 2] == '4')[0], 2] = 6
    credit[numpy.where(credit[:, 2] == '3')[0], 2] = 5
    credit[numpy.where(credit[:, 2] == '2')[0], 2] = 4
    credit[numpy.where(credit[:, 2] == '1')[0], 2] = 3
    credit[numpy.where(credit[:, 2] == '0')[0], 2] = 2
    credit[numpy.where(credit[:, 2] == 'X')[0], 2] = 1
    credit[numpy.where(credit[:, 2] == 'C')[0], 2] = 0
    credit = numpy.array(credit, dtype=int)
    print("End: Encoding features label")

    print("Begin: Encoding labels")
    credit_label = numpy.zeros((numpy.unique(credit[:, 0]).shape[0], 2), dtype=int)
    for i, l_unique in enumerate(numpy.unique(credit[:, 0])):
        if numpy.any(credit[numpy.where(credit[:, 0] == l_unique)[0], 2] > 3):
            credit_label[i] = [int(l_unique), 1]
        else:
            credit_label[i] = [int(l_unique), 0]

    unique_ID_int = numpy.intersect1d(application[:, 0], credit_label[:, 0], assume_unique=False, return_indices=True)
    application = application[unique_ID_int[1], :]
    credit_label = credit_label[unique_ID_int[2], :]
    print("End: Ecoding labels")

    print("Begin: Data spliting, data sampling, data randomisation")
    x_features = numpy.array(application[:, 1:], dtype=float)
    y_labels = numpy.array(credit_label[:, 1], dtype=int)


    x_features = x_features[:int(y_labels.shape[0]*100/100), :]
    y_labels = y_labels[:int(y_labels.shape[0]*100/100)]


    joblib.dump([x_features, y_labels], "model_data.joblib")


# Train your model on the dataset
def train():
    print("training model!")
    x_features, y_labels = joblib.load("model_data.joblib")
    x_features_train, x_features_test, y_labels_train, y_labels_test = train_test_split(x_features,y_labels, random_state=1, test_size=0.20)
    x_features_train, x_features_valid, y_labels_train, y_labels_valid = train_test_split(x_features_train, y_labels_train, random_state=1, test_size=0.20)

    x_features_train, y_labels_train = oversample.fit_resample(x_features_train, y_labels_train)
    x_features_train = ss.fit_transform(x_features_train)
    rnd_idx = numpy.random.permutation(len(y_labels_train))
    x_features_train, y_labels_train = x_features_train[rnd_idx], y_labels_train[rnd_idx]

    x_features_valid, y_labels_valid = oversample.fit_resample(x_features_valid, y_labels_valid)
    x_features_valid = ss.fit_transform(x_features_valid)
    rnd_idx = numpy.random.permutation(len(y_labels_valid))
    x_features_valid, y_labels_valid = x_features_valid[rnd_idx], y_labels_valid[rnd_idx]

    print("End: Data spliting, data sampling, data randomisation")

    # linear, polynomial, rbf, sigmoid
    print("Bengin: Model initialisation")

    kernel_time = []
    model_time_gamma = []
    model_time_coef0 = []
    model_time_degree = []
    model_time_C = []
    time0 = time.time()

    cv, n_jobs = 5, 5
    std = 0
    max_iter = 5000000
    valid_param = []
    kernel_list = ["linear", "poly", "rbf", "sigmoid"]
    # poly_param_list = [[1, 2, 6], [-0.5, 0, 0.5], ['scale', 'auto']]
    # rbf_param_list = [0.001, 0.01, 0.1, 0.5, 1, 2, 'scale', 'auto']
    # sigmoid_param_list = [[-0.1, -0.01, 0, 0.01, 0.1, 0.5], ['scale', 'auto']]
    C_list = [0.001, 1, 2, 4, 9, 13, 20]
    for k, kernel in enumerate(kernel_list):
        if kernel == "linear":
            print("Cross validation on linear kernel")
            svc = SVC(kernel = kernel)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            kernel_time.append(t1 - t0)
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, kernel])
            print(kernel, acc, std)
        elif kernel == "poly":
            print("Cross validation on polynomial kernel")
            svc = SVC(kernel = kernel)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            kernel_time.append(t1 - t0)
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std])
            print(kernel, acc, std)
        elif kernel == "rbf":
            print("Cross validation on rbf kernel")
            svc = SVC(kernel = kernel)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            kernel_time.append(t1 - t0)
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, kernel])
            print(kernel, acc, std)
        elif kernel == "sigmoid":
            print("Cross validation on sigmoid kernel")
            svc = SVC(kernel = kernel)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            kernel_time.append(t1 - t0)
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, kernel])
            print(kernel, acc, std)
    best_kernel = valid_param[numpy.argmax(valid_param)][2]
    valid_param = []

#########################################################################################################################################################################################
    best_kernel = "sigmoid"
#########################################################################################################################################################################################
    if best_kernel == "rbf":
        for gamma in [0.001, 0.01, 0.1, 0.5, 1, 2, 'scale', 'auto']:
            print("Cross validation on rbf kernel")
            svc = SVC(kernel = best_kernel, gamma = gamma)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_gamma.append([t1 - t0, gamma])
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, gamma])
            print(best_kernel, acc, std, gamma)
        best_gamma = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        valid_param = []
        for C in C_list:
            print("Cross validation on rbf kernel")
            svc = SVC(kernel = best_kernel, gamma = best_gamma, C = C)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_C.append([t1 - t0, C])
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, C])
            print(best_kernel, acc, std, best_gamma, C)
        best_C = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        # valid_param = []
        svc = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C)
        model = svc.fit(x_features_train, y_labels_train)
        acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
        print(best_kernel, acc, std, best_gamma, best_C)
        model = svc.fit(x_features_train, y_labels_train)
        joblib.dump([model, svc], "model_ml.joblib")
#############################################################################################################################
    valid_param = []
    if best_kernel == "sigmoid":
        for gamma in ['scale', 'auto']:
            print("Cross validation on sigmoid kernel")
            svc = SVC(kernel = best_kernel, gamma = gamma)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_gamma.append([t1 - t0, gamma])

            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, gamma])
            print(best_kernel, acc, std, gamma)
        best_gamma = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        valid_param = []
        for C in C_list:
            print("Cross validation on sigmoid kernel")
            svc = SVC(kernel = best_kernel, gamma = best_gamma, C = C)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_C.append([t1 - t0, C])
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, C])
            print(best_kernel, acc, std, best_gamma, C)
        best_C = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        valid_param = []
        for coef0 in [-0.1, -0.01, 0, 0.01, 0.1, 0.5]:
            print("Cross validation on sigmoid kernel")
            svc = SVC(kernel = best_kernel, gamma = best_gamma, C = best_C, coef0=coef0)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_coef0.append([t1 - t0, coef0])

            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, coef0])
            print(kernel, acc, std, best_gamma, best_C, coef0)
        best_coef0 = valid_param[numpy.argmax(valid_param)][2]
        svc = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C, coef0=best_coef0)
        model = svc.fit(x_features_train, y_labels_train)
        acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
        print(best_kernel, acc, std, best_gamma, best_C, best_coef0)
        model = svc.fit(x_features_train, y_labels_train)
        joblib.dump([model, svc], "model_ml.joblib")
#############################################################################################################################
    valid_param = []
    if best_kernel == "poly":
        poly_param_list = [[1, 2, 6], [-0.5, 0, 0.5], ['scale', 'auto']]
        for gamma in ['scale', 'auto']:
            print("Cross validation on poly kernel")
            svc = SVC(kernel=best_kernel, gamma=gamma)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_gamma.append([t1 - t0, gamma])

            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, gamma])
            print(best_kernel, acc, std, gamma)
        best_gamma = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        valid_param = []
        for C in C_list:
            print("Cross validation on poly kernel")
            svc = SVC(kernel=best_kernel, gamma=best_gamma, C=C)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_C.append([t1 - t0, C])
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, C])
            print(best_kernel, acc, std, best_gamma, C)
        best_C = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        valid_param = []
        for coef0 in [-0.5, 0, 0.5]:
            print("Cross validation on poly kernel")
            svc = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C, coef0=coef0)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_coef0.append([t1 - t0, coef0])
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, coef0])
            print(best_kernel, acc, std, best_gamma, best_C, coef0)
        best_coef0 = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        valid_param = []
        for degree in [2, 6, 12]:
            print("Cross validation on poly kernel")
            svc = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C, coef0=best_coef0, degree=degree)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_degree.append([t1 - t0, degree])
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, degree])
            print(best_kernel, acc, std, best_gamma, best_C, best_coef0)
        best_degree = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        svc = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C, coef0=best_coef0, degree=best_degree)
        model = svc.fit(x_features_train, y_labels_train)
        acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
        print(best_kernel, acc, std, best_gamma, best_C)
        model = svc.fit(x_features_train, y_labels_train)
        joblib.dump([model, svc], "model_ml.joblib")
#############################################################################################################################
    valid_param = []
    if best_kernel == "linear":
        for C in C_list:
            print("Cross validation on linear kernel")
            svc = SVC(kernel = kernel, C = C)
            t0 = time.time()
            model = svc.fit(x_features_train, y_labels_train)
            t1 = time.time()
            model_time_C.append([t1 - t0, C])
            acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
            valid_param.append([acc, std, C])
            print(kernel, acc, std, C)
        best_C = valid_param[numpy.argmax(numpy.asarray(valid_param)[:, 0])][2]
        svc = SVC(kernel=kernel, C=best_C)
        model = svc.fit(x_features_train, y_labels_train)
        acc, _ = compute_accuracy(svc, x_features_valid, y_labels_valid)
        print(kernel, acc, std, best_C)
        model = svc.fit(x_features_train, y_labels_train)
        joblib.dump([model, svc], "model_ml.joblib")

    time1 = time.time()
    print(time0 - time1)
    # model = svc_rbf.fit(x_features_train, y_labels_train)

    time_model = []
    s = numpy.arange(1/100, 100/100, 5/100)
    s_tot = y_labels_train.shape[0]
    for s_i in s:
        print(s_i)
        f = int(s_tot * s_i)
        t0 = time.time()
        svc = SVC(kernel="rbf")
        svc.fit(x_features_train[0: f], y_labels_train[0: f])
        t1 = time.time()
        time_model.append([t1 - t0, f])


    joblib.dump([kernel_time, time_model], "model_time.joblib")


# Compute the evaluation metrics and figures
def evaluate():
    print("evaluating model!")
    x_features, y_labels = joblib.load("model_data.joblib")

    x_features_train, x_features_test, y_labels_train, y_labels_test = train_test_split(x_features,y_labels, random_state=1, test_size=0.20)

    x_features_train, y_labels_train = oversample.fit_resample(x_features_train, y_labels_train)
    x_features_test, y_labels_test = oversample.fit_resample(x_features_test, y_labels_test)

    idx_wom = numpy.where(x_features_test[:, 0] == 0)[0]
    idx_men = numpy.where(x_features_test[:, 0] == 1)[0]

    x_features_train = ss.fit_transform(x_features_train)
    x_features_test = ss.transform(x_features_test)

    rnd_idx = numpy.random.permutation(len(y_labels_train))
    rnd_idx_test = numpy.random.permutation(len(y_labels_test))

    x_features_train, y_labels_train = x_features_train[rnd_idx], y_labels_train[rnd_idx]
    x_features_test, y_labels_test = x_features_test[rnd_idx_test], y_labels_test[rnd_idx_test]


    model, svc3 = joblib.load("model_ml.joblib")

    prediction = svc3.predict(x_features_test)
    c_matrix = confusion_matrix(y_labels_test, prediction)

    accuracy, c_matrix = compute_accuracy(svc3, x_features_test, y_labels_test)

    idx_safe = numpy.where(y_labels_test == 0)[0]
    safe_accuracy, safe_c_matrix = compute_accuracy(svc3, x_features_test[idx_safe, :], y_labels_test[idx_safe])

    idx_unsa = numpy.where(y_labels_test == 1)[0]
    unsa_accuracy, unsa_c_matrix = compute_accuracy(svc3, x_features_test[idx_unsa, :], y_labels_test[idx_unsa])

    men_accuracy, men_c_matrix = compute_accuracy(svc3, x_features_test[idx_men, :], y_labels_test[idx_men])
    idx_men_safe = numpy.intersect1d(idx_men, idx_safe)
    idx_men_unsa = numpy.intersect1d(idx_men, idx_unsa)
    men_accuracy_safe, men_safe_c_matrix = compute_accuracy(svc3, x_features_test[idx_men_safe, :], y_labels_test[idx_men_safe])
    men_accuracy_unsa, men_unsa_c_matrix = compute_accuracy(svc3, x_features_test[idx_men_unsa, :], y_labels_test[idx_men_unsa])

    idx_wom_safe = numpy.intersect1d(idx_wom, idx_safe)
    idx_wom_unsa = numpy.intersect1d(idx_wom, idx_unsa)
    wom_accuracy, wom_c_matrix = compute_accuracy(svc3, x_features_test[idx_wom, :], y_labels_test[idx_wom])
    wom_accuracy_safe, wom_safe_c_matrix = compute_accuracy(svc3, x_features_test[idx_wom_safe, :], y_labels_test[idx_wom_safe])
    wom_accuracy_unsa, wom_unsa_c_matrix = compute_accuracy(svc3, x_features_test[idx_wom_unsa, :], y_labels_test[idx_wom_unsa])

    print("End: Model initialisation")

    print("Begin: Reporting")

    # Validation Set Size Bar Chart
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    width = 0.25
    rects0 = ax1.bar(0, len(y_labels_train), width, label='Overall')
    rects1 = ax1.bar(1 - width*1/2, len(y_labels_train[idx_safe]), width, label='Safe')
    rects2 = ax1.bar(1 + width*1/2, len(y_labels_train[idx_unsa]), width, label='Unsafe')
    rects3 = ax1.bar(2 - width, len(y_labels_train[idx_men]), width, label='Men')
    rects4 = ax1.bar(2, len(y_labels_train[idx_men_safe]), width)
    rects5 = ax1.bar(2 + width, len(y_labels_train[idx_men_unsa]), width)
    rects6 = ax1.bar(3 - width, len(y_labels_train[idx_wom]), width, label='Women')
    rects7 = ax1.bar(3, len(y_labels_train[idx_wom_safe]), width)
    rects8 = ax1.bar(3 + width, len(y_labels_train[idx_wom_unsa]), width)
    ax1.set_xticks(numpy.arange(4))
    ax1.set_ylabel('Traning Set Size')
    ax1.set_title('Set Size for Slices')
    ax1.legend(loc="upper right")

    rects0 = ax2.bar(0, len(y_labels_test), width)
    rects1 = ax2.bar(1 - width*1/2, len(y_labels_test[idx_safe]), width)
    rects2 = ax2.bar(1 + width*1/2, len(y_labels_test[idx_unsa]), width)
    rects3 = ax2.bar(2 - width, len(y_labels_test[idx_men]), width)
    rects4 = ax2.bar(2, len(y_labels_test[idx_men_safe]), width, label='Men Safe')
    rects5 = ax2.bar(2 + width, len(y_labels_test[idx_men_unsa]), width, label='Men Unsafe')
    rects6 = ax2.bar(3 - width, len(y_labels_test[idx_wom]), width)
    rects7 = ax2.bar(3, len(y_labels_test[idx_wom_safe]), width, label='Women Safe')
    rects8 = ax2.bar(3 + width, len(y_labels_test[idx_wom_unsa]), width, label='Women Unsafe')
    ax2.set_xticks(numpy.arange(4))
    ax2.set_xticklabels(['Overall', 'Safety', 'Men', 'Women'])
    ax2.set_ylabel('Validation Set Size')
    ax2.set_xlabel('Slices')
    ax2.legend(loc="upper right")
    plt.savefig("valid_size.pdf")

    # Acuracy Bar Chart
    fig, ax = plt.subplots()
    width = 0.25
    rects0 = ax.bar(0, accuracy, width, label='Overall')
    rects1 = ax.bar(1 - width*1/2, safe_accuracy, width, label='Safe')
    rects2 = ax.bar(1 + width*1/2, unsa_accuracy, width, label='Unsafe')
    rects3 = ax.bar(2 - width, men_accuracy, width, label='Men')
    rects4 = ax.bar(2, men_accuracy_safe, width, label='Men Safe')
    rects5 = ax.bar(2 + width, men_accuracy_unsa, width, label='Men Unsafe')
    rects6 = ax.bar(3 - width, wom_accuracy, width, label='Women')
    rects7 = ax.bar(3, wom_accuracy_safe, width, label='Women Safe')
    rects8 = ax.bar(3 + width, wom_accuracy_unsa, width, label='Women Unsafe')
    ax.set_xticks(numpy.arange(4))
    ax.set_xticklabels(['Overall', 'Safety', 'Men', 'Women'])
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Slices')
    ax.set_title('Accuracy on Slices')
    ax.legend()
    plt.savefig("valid_acuracy.pdf")

    # Acuracy Bar Chart
    fig, ax = plt.subplots()
    width = 0.25
    rects0 = ax.bar(0 - width, c_matrix[0], width, label='Overall FP') ##OverFP
    rects1 = ax.bar(0 , men_c_matrix[0], width, label='Men FP') ##MenFP
    rects2 = ax.bar(0 + width, wom_c_matrix[0], width, label='Wom FP') ##WomFP

    rects3 = ax.bar(1 - width, c_matrix[1], width, label='Overall FN') ##OverFN
    rects4 = ax.bar(1, men_c_matrix[1], width, label='Men FN') ##MenFN
    rects5 = ax.bar(1 + width, wom_c_matrix[1], width, label='Wom FN') ##WomFN

    rects6 = ax.bar(2 - width, c_matrix[2], width, label='Overall FD') ##OverFD
    rects7 = ax.bar(2, men_c_matrix[2], width, label='Men FD') ##MenFD
    rects8 = ax.bar(2 + width, wom_c_matrix[2], width, label='Wom FD') ##WomFD

    rects9 = ax.bar(3 - width, c_matrix[3], width, label='Overall FO') ##OverFO
    rects10 = ax.bar(3, men_c_matrix[3], width, label='Men FO') ##MenFO
    rects11 = ax.bar(3 + width, wom_c_matrix[3], width, label='Wom FO') ##WomFO
    ax.set_xticks(numpy.arange(4))
    ax.set_xticklabels(['FP', 'FN', 'FD', 'FO'])
    ax.set_ylabel('Error rate')
    ax.set_xlabel('Slices')
    ax.set_title('Fairness on Slices')
    ax.legend()
    plt.savefig("valid_fairness.pdf")

    print("End: Reporting")

    model_time = joblib.load("model_time.joblib")
    kernel_time = model_time[0]
    time_model = numpy.asarray(model_time[1])

    plt.figure()
    plt.plot(["linear", "poly", "rbf", "sigmoid"], kernel_time)
    plt.xlabel("kernel type")
    plt.ylabel("time")
    plt.savefig("kernel_time.pdf")

    plt.figure()
    plt.semilogy(time_model[:, 1], time_model[:, 0])
    plt.xlabel("data set size")
    plt.ylabel("time")
    plt.savefig("traning_time.pdf")


    plt.show()

    print(0)



# Compile the PDF documents
def build_paper():
    print("building papers!")  # replace this with code to make the papers
    subprocess.run(["pdflatex", "card.tex"])
    subprocess.run(["pdflatex", "paper.tex"])


###############################
# No need to modify past here #
###############################

supported_functions = {'fetch': fetch,
                       'train': train,
                       'evaluate': evaluate,
                       'build_paper': build_paper}

# If there is no command-line argument, return an error
if len(sys.argv) < 2:
    print("""
        You need to pass in a command-line argument.
        Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
    """)
    sys.exit(1)

# Extract the first command-line argument, ignoring any others
arg = sys.argv[1]

# Run the corresponding function
if arg in supported_functions:
    supported_functions[arg]()
else:
    raise ValueError("""
        '{}' not among the allowed functions.
        Choose among 'fetch', 'train', 'evaluate' and 'build_paper'.
        """.format(arg))















