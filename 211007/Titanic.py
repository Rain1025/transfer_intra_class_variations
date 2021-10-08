# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 20:47:47 2021

@author: Rain L
"""
import pandas as pd
import numpy as np
import csv
import math

def load_data():
    data = pd.read_csv('train.csv')
    data_sex = data.iloc[:,4]
    data_sex[data_sex == 'female'] = 0
    data_sex[data_sex == 'male']=1
    x_train=np.empty([700,3])
    x_train[:,0] = data.iloc[0:700,2]
    x_train[:,1] = data_sex[0:700]
    x_train[:,2] = data.iloc[0:700,1]
    x_test=np.empty([191,3])
    x_test[:,0] = data.iloc[700:891,2]
    x_test[:,1] = data_sex[700:891]
    x_test[:,2] = data.iloc[700:891,1]
    return x_train, x_test

def train_data(x_train, x_test):
    survive_rate=np.mean(x_train[:,2])
    death_rate = 1-survive_rate
    survive_num=len(x_train[x_train[:,2]==1])
    death_num=len(x_train[x_train[:,2]==0])
    sd_rate=[death_rate,survive_rate]
    sd_pclass=np.empty([2,3])
    sd_sex=np.empty([2,2])
    count_class=0
    count_sex=0
    for i in range(2):
        for j in range(3):
            for k in range(700):
                if x_train[k,0]==j+1 and x_train[k,2]==i:
                    count_class=count_class+1
            sd_pclass[i,j]=count_class/len(x_train[x_train[:,2]==i])
            count_class=0
            
    for i in range(2):
        for j in range(2):
            for k in range(700):
                if x_train[k,1]==j and x_train[k,2]==i:
                    count_sex=count_sex+1
            sd_sex[i,j]=count_sex/len(x_train[x_train[:,2]==i])
            count_sex=0
    y_yuc=np.empty([191,1])
    for i in range(191):
        if ((sd_rate[0]*sd_pclass[0,int(x_test[i,0])-1]*sd_sex[0,int(x_test[i,1])]) > (sd_rate[1]*sd_pclass[1,int(x_test[i,0])-1]*sd_sex[1,int(x_test[i,1])])):
            y_yuc[i,0]=0
        else:
            y_yuc[i,0]=1
    num=0
    for i in range(191):
        if y_yuc[i,0] == x_test[i,2]:
            num=num+1
    print('x_test的预测表现为'+str(num/191))
    return sd_pclass, sd_sex ,sd_rate

def test_data(sd_pclass, sd_sex, sd_rate):
    data = pd.read_csv('test.csv')
    data.iloc[:,3][data.iloc[:,3]=='female']=0
    data.iloc[:,3][data.iloc[:,3]=='male']=1
    f = open('submission.csv','w',encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["PassengerId","Survived"])
    data_test=np.empty([len(data.iloc[:,0]),2])
    data_test[:,0]=data.iloc[:,1]
    data_test[:,1]=data.iloc[:,3]
    for i in range(len(data.iloc[:,0])):
        if ((sd_rate[0]*sd_pclass[0,int(data_test[i,0])-1]*sd_sex[0,int(data_test[i,1])]) > (sd_rate[1]*sd_pclass[1,int(data_test[i,0])-1]*sd_sex[1,int(data_test[i,1])])):
            csv_writer.writerow([i+1,0])
        else:
            csv_writer.writerow([i+1,1])
    f.close()
    
    
if __name__ == '__main__':
    x_train, x_test= load_data()
    sd_pclass, sd_sex, sd_rate = train_data(x_train, x_test)
    test_data(sd_pclass, sd_sex, sd_rate)

            
    
    
    
    
    