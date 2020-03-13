#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Preprocessing for the training data
# input the location of train data
# ex: 
# path = 'C:\\Users\\user\\Documents\\北京清華\\碩一下\\模式識別\\final_project\\data\\WriterID\\Data10\\Train'
# pre_train_data = pre_traindata(path)

def pre_traindata(path):
    
    import numpy as np
    import os
    name=[]
    name.append(os.listdir(path))
    
    all_train=[]
    for i in range(len(name[0])):
        a = np.load(path+'//'+name[0][i], allow_pickle=True)
        all_train.append(a)
    
    # remove the stroke that only have one point
    all_student=[]
    for n in range(len(all_train)):
        student_n=[]
        for i in range(len(all_train[n])):
            word_n=[]
            for j in range(len(all_train[n][i])):
                if len(all_train[n][i][j])==1:
                    continue
                else:
                    n_stroke=[]
                    for k in range(len(all_train[n][i][j])):
                        n_stroke.append(all_train[n][i][j][k])
                word_n.append(n_stroke)
            student_n.append(word_n)
        all_student.append(student_n)
    
    # sigma and mu
    new_student=[]
    for n in range(len(all_student)):
        n_student=[]
        for i in range(len(all_student[n])):
            word=[]
            for j in range(len(all_student[n][i])):
                mu_x = np.mean(all_student[n][i][j],0)[0]
                mu_y = np.mean(all_student[n][i][j],0)[1]
                std_x = np.std(all_student[n][i][j],0)[0]
                std_y = np.std(all_student[n][i][j],0)[1]
                stroke=[]
                for k in range(len(all_student[n][i][j])):
                    if all_student[n][i][j][k][0] > mu_x + 4*std_x or all_student[n][i][j][k][1] > mu_y + 4*std_y:
                        continue
                    else:
                        stroke.append(all_student[n][i][j][k])
                word.append(stroke)
            n_student.append(word)
        new_student.append(n_student)
    
    # delta
    alll=[]
    for n in range(len(new_student)):
        student=[]
        for i in range(len(new_student[n])):
            word=[]
            for j in range(len(new_student[n][i])):
                stroke=[]
                for k in range(1,len(new_student[n][i][j])):
                    di=[]
                    delta_x=new_student[n][i][j][k][0]-new_student[n][i][j][k-1][0]
                    delta_y=new_student[n][i][j][k][1]-new_student[n][i][j][k-1][1]
                    di.append(delta_x)
                    di.append(delta_y)
                    stroke.append(di)
                word.append(stroke)
            student.append(word)
        alll.append(student)
    
    # add 1 and -1
    for n in range(len(alll)):
        for i in range(len(alll[n])):
            for j in range(len(alll[n][i])):
                for k in range(len(alll[n][i][j])):
                    if k==0:
                        alll[n][i][j][k].append(-1)
                    elif k==len(alll[n][i][j])-1:
                        alll[n][i][j][k].append(-1)
                    else:
                        alll[n][i][j][k].append(1)
    
    # seperate the stroke
    all_data=[]
    for n in range(len(alll)):
        student_id=[]
        for i in range(len(alll[n])):
            word=[]
            for j in range(len(alll[n][i])):
                for k in range(len(alll[n][i][j])):
                    word.append(alll[n][i][j][k])
            student_id.append(word)
        all_data.append(student_id)
    
    # saperate the word as delta s
    al=[]
    for n in range(len(all_data)):
        delta_s=[]
        for i in range(len(all_data[n])):
            for j in range(len(all_data[n][i])):
                delta_s.append(all_data[n][i][j])
        al.append(delta_s)
    
    return al


# In[ ]:




