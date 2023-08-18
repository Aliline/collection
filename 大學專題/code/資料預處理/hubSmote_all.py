from imblearn.over_sampling import SMOTE, ADASYN
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt
import showtrain
import numpy as np
import pandas as pd
import os
import urllib.request
import modin.pandas as modinpd
import ray
import random
import tqdm
random.seed(1)


# from distributed import Client
# client = Client(n_workers=6)
# os.environ["MODIN_CPUS"] = "10"
os.environ["MODIN_ENGINE"] = "ray"
# original_path = "D:/tempdata/"
S_path = "D:/hubdata/S/"
D_path = "D:/hubdata/D/"
useCore = 6
# filter_day = 3  # 天數篩選標準_3天
# filter_biomarker_num = 2  # 一個biomarker測量的次數
outpath_csv = "D:/smotedata/"
# outpath_pic = "D:/picdata/"

os.system("cls")  # 清除螢幕
fail = []


def Main():

    # 讀取原始資料
    # args = []
    ready_run = 0
    file_runed = 0
    file_set = []
    fname_set = []
    dnum = 0
    for path, folders, files in os.walk(S_path):  # walk遞迴印出資料夾中所有目錄及檔名
        file_id = 0
        # pic_list = multiprocessing.Manager().list()
        for file in files:
            # os.path.join()：將多個路徑組合後返回
            file_set.append(os.path.join(path, file))
            fname_set.append(file)
    for path, folders, files in os.walk(D_path):  # walk遞迴印出資料夾中所有目錄及檔名
        file_id = 0
        # pic_list = multiprocessing.Manager().list()
        for file in files:
            # os.path.join()：將多個路徑組合後返回
            file_set.append(os.path.join(path, file))
            fname_set.append(file)
    process(file_set, fname_set)
    print("{}處理完成{}".format("="*10, "="*10))


def process(filePaths, fileNames):

    try:
        x_train = []
        x_label = []
        duid = []
        usecols = ["BT", "HR", "NBP D", "NBP S", "RR", "age", "sex"]
        dl = 72
        dc = 7
        D_files = os.listdir(D_path)
        print("importing Data....")
        for f in tqdm.tqdm(filePaths):
            df = modinpd.read_csv(f,
                                  low_memory=False)
            # itnm 正規化

            # df[(df["itnm"] == "BT")]
            # df[(df["itnm"] == "HR")] = 1
            # df[(df["itnm"] == "NBP D")] = 2
            # df[(df["itnm"] == "NBP S")] = 3
            # df[(df["itnm"] == "RR")] = 4
            # if (count == 44):
            #     print(f)
            # 為了讓資料長度一致,填充0
            # if (dl - len(df) > 0):
            #     tl = []
            #     for t in range(dl - len(df)):
            #         tl.append(0)

            #     tpdf = pd.DataFrame(tl, columns=['val'])
            #     # tpdf = pd.concat([pd.DataFrame(np.array([[0]]),
            #     #                                columns=['val']) for i in range(dl - len(df))],
            #     #                  ignore_index=True)
            #     df = modinpd.concat([df, tpdf], ignore_index=True)

            tdf = df.values
            # minmaxscaler
            # minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # # 轉成float32讓smote可以吃
            # tdf = minmax.fit_transform(tdf).astype('float32')
            x_train.append(np.array(tdf))
        # label 正規化
        for f in fileNames:
            ftype = f[0:1]
            if ftype == "S":
                x_label.append(1)
            else:
                x_label.append(0)

        bdf = np.array(x_train).astype('float32')
        bl = np.array(x_label).astype('float32')
        bdf = bdf.reshape(len(bdf), dl*dc)
        train_feature_rs, train_label_rs = SMOTE(
            random_state=42).fit_resample(bdf, bl)
        train_feature_rs = train_feature_rs.reshape(
            len(train_feature_rs), dl, dc)
        # train_feature_ad, train_label_ad = ADASYN(
        #     random_state=42).fit_resample(bdf, bl)
        # train_feature_ad = train_feature_ad.reshape(
        #     len(train_feature_ad), dl, dc)
        bdf = bdf.reshape(len(bdf), dl, dc)
        # string_tad = []
        string_bdf = []
        for k in range(len(bdf)):
            string_bdf.append(bdf[k].tostring())
        for k in range(len(train_feature_rs)):
            temps = train_feature_rs[k].tostring()
            if temps in string_bdf:
                duid.append(k)

        train_feature_rs = np.delete(train_feature_rs, duid, 0)
        # df = modinpd.read_csv(filePaths[len(bdf)-1],
        #                       low_memory=False)
        # smote 成功 接下來寫輸出excel
        print("writing SMOTE......")
        csv_outpath = "{}/{}".format(outpath_csv, "ver1")
        adas_outpath = "{}/{}".format(outpath_csv, "ver2")
        if(not os.path.isdir(csv_outpath)):
            os.makedirs(csv_outpath)
        if(not os.path.isdir(adas_outpath)):
            os.makedirs(adas_outpath)
        for k in tqdm.tqdm(range(len(train_feature_rs))):
            random_D = random.choice(D_files)
            fname = "Smote_{}_{}".format(random_D, k)
            sdf = pd.DataFrame(train_feature_rs[k], columns=usecols)
            random_Df = modinpd.read_csv(os.path.join(D_path, random_D), dtype={"BT": float, "HR": float, "NBP": float, "NBP": float, "RR": float, "age": float, "sex": int},
                                         low_memory=False)
            sdf["sex"] = random_Df["sex"]

            sdf.to_csv("{}/{}.csv".format(csv_outpath,
                                          fname), index=False)

        # print("writing ADASYN......")
        # for k in tqdm.tqdm(range(len(bdf), len(train_feature_ad))):
        #     random_D = random.choice(D_files)
        #     fname = "Smote_{}_{}".format(random_D, k)
        #     sdf = pd.DataFrame(train_feature_ad[k], columns=usecols)
        #     random_Df = modinpd.read_csv(os.path.join(D_path, random_D), dtype={"BT": float, "HR": float, "NBP": float, "NBP": float, "RR": float, "age": float, "sex": int},
        #                                  low_memory=False)
        #     sdf["sex"] = random_Df["sex"]

        #     sdf.to_csv("{}/{}.csv".format(adas_outpath,
        #                                   fname), index=False)

        print(fileNames[0])
    except Exception as e:
        # print(testdf)
        # traceback.print_exception(e)
        print("處理失敗：{}=>{}".format(fileNames[0], e))
        fail.append("{}=>{}".format(fileNames[0], e))


if __name__ == "__main__":
    # import multiprocessing
    Main()
