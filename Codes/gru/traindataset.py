# encoding:utf-8
import numpy as np
# import matplotlib.pyplot as plt
# import math
import random
from PreprocessTrain import pre_traindata

def dataset(path_train,path_test,train_k,train_sample,test_k,test_sample):

    ## load data
    data1 = pre_traindata(path_train)
    data2 = pre_traindata(path_test)

    train_data = []
    train_label = []

    for n in range(10):
        start_list = []
        label = [n]
        for i in range(train_sample):
            start = random.randint(0, len(data1[n]) - train_k)
            start_list.append(start)
        start_list.sort()
        for i in range(train_sample):
            start_p = start_list[i]
            RHS = []
            for j in range(start_p, start_p + train_k):
                RHS.append(data1[n][j])
            train_data.append(RHS)
            train_label.append(label)

    train_x = train_data
    train_y = train_label

    test_data = []
    test_label = []

    for n in range(10):
        start_list = []
        label = [n]
        for i in range(test_sample):
            start = random.randint(0, len(data2[n]) - test_k)
            start_list.append(start)
        start_list.sort()
        for i in range(test_sample):
            start_p = start_list[i]
            RHS = []
            for j in range(start_p, start_p + test_k):
                RHS.append(data2[n][j])
            test_data.append(RHS)
            test_label.append(label)

    test_x = test_data
    test_y = test_label

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label