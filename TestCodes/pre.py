#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
# import matplotlib.pyplot as plt
import random


# In[26]:



def preprocessing(test_data, train_sample, train_k):
    # remove the stroke that have only one point

    student_n=[]
    for i in range(len(test_data)):
        word_n = []
        for j in range(len(test_data[i])):
            if len(test_data[i][j])==1:
                continue
            else:
                n_stroke = []
                for k in range(len(test_data[i][j])):
                    n_stroke.append(test_data[i][j][k])
            word_n.append(n_stroke)
        student_n.append(word_n)
    all_data=student_n

    
   
    # remove the point far from mu+4*sigma
    student_n=[]
    for i in range(len(all_data)):
        word_n=[]
        for j in range(len(all_data[i])):
            mu_x = np.mean(all_data[i][j],0)[0]
            mu_y = np.mean(all_data[i][j],0)[1]
            std_x = np.std(all_data[i][j],0)[0]
            std_y = np.std(all_data[i][j],0)[1]
            stroke = []
            for k in range(len(all_data[i][j])):
                if all_data[i][j][k][0]>mu_x + 4*std_x or all_data[i][j][k][1] > mu_y + 4*std_y:
                    continue
                else:
                    stroke.append(all_data[i][j][k])
            word_n.append(stroke)
        student_n.append(word_n)
    all_data2=student_n

    # calculate delta x y
    student=[]
    for i in range(len(all_data2)):
        word_n=[]
        for j in range(len(all_data2[i])):
            stroke=[]
            for k in range(1, len(all_data2[i][j])):
                di=[]
                delta_x=all_data2[i][j][k][0]-all_data2[i][j][k-1][0]
                delta_y=all_data2[i][j][k][1]-all_data2[i][j][k-1][1]
                di.append(delta_x)
                di.append(delta_y)
                stroke.append(di)
            word_n.append(stroke)
        student.append(word_n)
    all_delta=student

    
    # add 1 and -1
    for i in range(len(all_delta)):
        for j in range(len(all_delta[i])):
            for k in range(len(all_delta[i][j])):
                if k==0:
                    all_delta[i][j][k].append(-1)
                elif k==len(all_delta[i][j])-1:
                    all_delta[i][j][k].append(-1)
                else:
                    all_delta[i][j][k].append(1)
    
    # seperate the stroke, w[i] = one word
    w=[]
    for i in range(len(all_delta)):
        word=[]
        for j in range(len(all_delta[i])):
            for k in range(len(all_delta[i][j])):
                word.append(all_delta[i][j][k])
        w.append(word)
    
    # seperate the word
    s=[]
    for i in range(len(w)):
        for j in range(len(w[i])):
            s.append(w[i][j])
    
    
    train_data=[]
    start_list=[]
    for i in range(train_sample):
        start=random.randint(0,len(s)-train_k)
        start_list.append(start)
    start_list.sort()
    for i in range(train_sample):
        start_p = start_list[i]
        RHS=[]
        for j in range(start_p,start_p+train_k):
            RHS.append(s[j])
        train_data.append(RHS)
    train_x=np.array(train_data)
    return train_x

