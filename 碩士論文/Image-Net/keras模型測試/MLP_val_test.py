# 7.此為使用ISLVRC測試資料集來測試keras模型的準確率
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tkinter import X
from imblearn.over_sampling import SMOTE
import numpy as np
import multiprocessing
import numpy as np
import pandas as pd
import math
#from pandas.core.construction import array
from concurrent.futures import ThreadPoolExecutor
from PIL import Image  
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar
#keras-------
import keras
from keras.layers import Input,Conv2D,MaxPooling2D,Dense,Flatten,concatenate,Dropout,BatchNormalization,LSTM
from keras.models import Sequential,Model
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from openpyxl import Workbook
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import mean_squared_error
from keras import regularizers
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import threading
import numba
import gc
savepath = "D:/F111156114_Aliline/imgtoword/wordtovector/MLP_temp/"
testpath = "D:/F111156114_Aliline/imgtoword/wordtovector/MLP_temp/"
# savepath = "D:/F111156114_Aliline/imgtoword/wordtovector/MLP_model/"
# testpath = "D:/F111156114_Aliline/imgtoword/wordtovector/MLP_renew/"
xvector = np.load('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_renew/valx.npy',allow_pickle=True) #來源於createVALvector
yvector = np.load('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_renew/val_290_y_2.npy')    #來源於createVALvector
testtxt = "D:/F111156114_Aliline/imgtoword/wordtovector/vector_txt/vectors_all_ver_.txt"
# testvecotr = np.load("D:/F111156114_Aliline/imgtoword/wordtovector/vector_txt/vectors_all_ver_.npy")
actiname = 'tanh'
lr=0.09
beta_1=0.9
beta_2=0.999
epsilon=1e-08
decay=0.0
ver = 0
ver_num = 0
peroid = 500
# @numba.jit
def cul(testvecotr,y,classes):
    vtotal = 0
    # for v in range(classes):
    #     vc = (y[v] - testvecotr[v])**2 #距離
    #     vtotal = vtotal + vc
        # temp = (testvecotr[v] - y[v])**2
    # vc = np.mean((y - testvecotr)**2)  #MSE

    temp = (y - testvecotr)**2
    temp = np.sum(temp)
    temp = np.sqrt(temp)
    # vtotal = math.sqrt(vtotal)
    # temp2 = mean_squared_error(testvecotr , y)

    # return vtotal
    return temp



# @numba.jit
def result(X,Y,savepath,model_name,classes,testvecotr):
    label = []
    
    # with open(testtxt, 'r', encoding='utf-8') as labels:
    #     for temp in labels:
    #         if "：" in temp:
    #             te = temp.split(".")
    #             te = te[1].split("：")
    #             label.append(te[0])
    total = len(Y)
    model=tf.keras.models.load_model(f'{savepath}{model_name}')
    y_prob=model.predict(X) #預測機率
    y_prob_L = []
    y_top_5 = []
    y_top_min = []
    y_dis = []
    y_class = []
    total = len(y_prob)
    count = 0
    #老師用的是直接排序所有距離後直接抓前5 這隻程式只看前5 所以不知道第6以後的距離
    with alive_bar(total) as bar:
        for y in y_prob:
            min = 0
            min_lab = -1
            y_top_5.append([-1,-1,-1,-1,-1])
            y_top_min.append([-1,-1,-1,-1,-1])
            temp_dist = []
            temp_class = []
            # if count == 65 :
            #     break
            for k in range(1,1001):
                vtotal = 0
                vtotal = cul(testvecotr[k-1],y,classes)
                temp_dist.append(vtotal)
                temp_class.append(k)
                    # if k == 361 :
                    #     k = k
                    #改成MSE的計算公式
                    #Top5正確率試試看(距離前5短)
                if min_lab == -1 :
                    min = vtotal
                    min_lab = k
                elif min > vtotal:
                    min = vtotal
                    min_lab = k
                #以下是if式Top-5準確率                    
                if y_top_5[count][0] == -1 :
                    y_top_min[count][0] = vtotal
                    y_top_5[count][0] = k
#========================================================
                elif y_top_5[count][1] == -1:
                    y_top_min[count][1] = vtotal
                    y_top_5[count][1] = k
#========================================================                    
                elif y_top_5[count][2] == -1:
                    y_top_min[count][2] = vtotal
                    y_top_5[count][2] = k
#========================================================                    
                elif y_top_5[count][3] == -1:
                    y_top_min[count][3] = vtotal
                    y_top_5[count][3] = k
#========================================================                   
                elif y_top_5[count][4] == -1:
                    y_top_min[count][4] = vtotal
                    y_top_5[count][4] = k  
#========================================================                                                                              
                elif y_top_min[count][0] > vtotal:
                    y_top_min[count][0] = vtotal
                    y_top_5[count][0] = k
#========================================================                                                                              
                elif y_top_min[count][1] > vtotal:
                    y_top_min[count][1] = vtotal
                    y_top_5[count][1] = k
#========================================================                                                                              
                elif y_top_min[count][2] > vtotal:
                    y_top_min[count][2] = vtotal
                    y_top_5[count][2] = k
#========================================================                                                                              
                elif y_top_min[count][3] > vtotal:
                    y_top_min[count][3] = vtotal
                    y_top_5[count][3] = k
#========================================================                                                                              
                elif y_top_min[count][4] > vtotal:
                    y_top_min[count][4] = vtotal
                    y_top_5[count][4] = k                                                                                
                bar.text("Sub_Progessing...{}%".format(k/10))
            y_prob_L.append(min_lab)
            y_dis.append(temp_dist)
            y_class.append(temp_class)
            bar()
            count = count +1
            #這裡是每5000張圖片輸出所有的類別距離 用於檢查哪邊出問題
            if count % 5000 == 0 :
                y_dis_d = pd.DataFrame({'class' : y_class , 'dist' : y_dis})
                name_out = model_name.replace(".h5","")
                y_dis_d = y_dis_d.sort_values('dist'  , ignore_index = True , axis = 0 , ascending = True)
                y_dis_d.to_excel("D:/F111156114_Aliline/imgtoword/wordtovector/temp_excel2/{}_dist_{}.xlsx".format(name_out,count))
                y_dis = []
                y_class = []


    y_top_min_d = pd.DataFrame(y_top_min)
    y_top_5_d  = pd.DataFrame(y_top_5)
    y_dis_d = pd.DataFrame(y_dis)
    name_out = model_name.replace(".h5","")
    #用於輸出前5類別編號(top_5)與其距離(top_min) 同樣用於檢查
    y_top_5_d.to_excel("D:/F111156114_Aliline/imgtoword/wordtovector/temp_excel/{}_top5_temp.xlsx".format(name_out))
    y_top_min_d.to_excel("D:/F111156114_Aliline/imgtoword/wordtovector/temp_excel/{}_topmin_temp.xlsx".format(name_out))
    # y_dis_d.to_excel("D:/F111156114_Aliline/imgtoword/wordtovector/temp_excel/{}_dist_temp.xlsx".format(name_out))
    Accuracy=accuracy_score(Y,y_prob_L)    

    
    top_5_acc = 0
    correct = 0
    count = 0
    for ans in Y :
        for temp in range(5):
            if ans == y_top_5[count][temp] :
                correct = correct +1
                break
        count = count +1
    top_5_acc = correct/total
    return Accuracy,top_5_acc


model_name = []
Acc_set = []
top_5 = []
#開始測試 架構同MLP_train的測試 只多了個top-5輸出
for model in os.listdir(testpath) :
    classes = int(model.split("_")[1])
    testvecotr = np.load("D:/F111156114_Aliline/imgtoword/wordtovector/vectors_{}_ver2.npy".format(classes))
    test = xvector.shape
    if len(test) == 3:
        xvector = xvector.reshape(test[0],test[2])
    acc,top_5_acc = result(xvector,yvector,testpath,model,classes,testvecotr)
    model_name.append(model)
    Acc_set.append(acc)
    top_5.append(top_5_acc)
    print("model:{},Acc={},top_5={}".format(model,acc,top_5_acc))
for k in range(len(model_name)):
    print("model:{},Acc={},top5 = {}".format(model_name[k],Acc_set[k],top_5[k]))