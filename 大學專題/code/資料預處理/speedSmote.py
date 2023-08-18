from imblearn.over_sampling import SMOTE
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


# from distributed import Client
# client = Client(n_workers=6)
# os.environ["MODIN_CPUS"] = "10"
os.environ["MODIN_ENGINE"] = "ray"
original_path = "D:/tempdata/"
S_path = "D:/tempdata/S/"
D_path = "D:/tempdata/D/"
useCore = 6
# filter_day = 3  # 天數篩選標準_3天
# filter_biomarker_num = 2  # 一個biomarker測量的次數
outpath_csv = "D:/smotedata/"
outpath_pic = "D:/picdata/"
filter_day = 3
run_file = 60  # 設定六個D要讓SMOTE產生幾個DF
os.system("cls")  # 清除螢幕
fail = []
file_limit = 132000  # 設定檔案上限


def Main():

    # 讀取原始資料
    args = []
    ready_run = 0
    file_runed = 0
    file_set = []
    fname_set = []
    dnum = 0
    D_files = os.listdir(D_path)
    for path, folders, files in os.walk(S_path):  # walk遞迴印出資料夾中所有目錄及檔名
        file_id = 0
        # pic_list = multiprocessing.Manager().list()
        for file in files:
            # os.path.join()：將多個路徑組合後返回
            file_set.append(os.path.join(path, file))
            fname_set.append(file)
            file_id = file_id + 1
            ready_run = ready_run + 1
            if ((ready_run == run_file) or (file_id == len(files))):
                for tem in range(6):
                    file_set.append(os.path.join(D_path, D_files[dnum]))
                    fname_set.append(D_files[dnum])
                    dnum = dnum + 1
                # args.append([file_set, fname_set, pic_list])
                # with multiprocessing.Pool(processes=useCore, maxtasksperchild=6000) as pool:
                #     pool.starmap(process, args)
                process(file_set, fname_set)
                print("{}處理完成{}".format("="*10, "="*10))
                ready_run = 0
                args = []
                file_set = []
                fname_set = []

                # print(len(pic_list))
                # for x in pic_list:
                #     file_runed = file_runed + x
                # pic_list[:] = []
                # if file_runed >= file_limit:
                #     break
                if dnum >= len(D_files):
                    break
        if file_runed >= file_limit:
            break
        if dnum >= len(D_files):
            break
    if len(fail) >= 1:
        for k in fail:
            print(k)


def chk_itnm(itnm):
    if itnm == "BT":
        return 0
    elif itnm == "HR":
        return 1
    elif itnm == "NBP D":
        return 2
    elif itnm == "NBP S":
        return 3
    else:
        return 4


def process(filePaths, fileNames):

    try:
        x_train = []
        x_label = []
        usecols = ["val"]
        dl = 0
        mfilepath = ""
        for f in filePaths:
            tf = len(modinpd.read_csv(f, usecols=usecols,
                                      low_memory=False))
            if dl <= tf:
                dl = tf
                mfilepath = f

        for f in filePaths:
            df = modinpd.read_csv(f, usecols=usecols,
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
            if (dl - len(df) > 0):
                tl = []
                for t in range(dl - len(df)):
                    tl.append(0)

                tpdf = pd.DataFrame(tl, columns=['val'])
                # tpdf = pd.concat([pd.DataFrame(np.array([[0]]),
                #                                columns=['val']) for i in range(dl - len(df))],
                #                  ignore_index=True)
                df = modinpd.concat([df, tpdf], ignore_index=True)

            tdf = df.values
            # minmaxscaler
            minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # 轉成float32讓smote可以吃
            tdf = minmax.fit_transform(tdf).astype('float32')
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
        bdf = bdf.reshape(len(bdf), dl)
        train_feature_rs, train_label_rs = SMOTE(
            random_state=42).fit_resample(bdf, bl)
        train_feature_rs = train_feature_rs.reshape(
            len(train_feature_rs), dl, 1)
        # smote 成功 接下來寫輸出excel
        df = modinpd.read_csv(mfilepath,
                              low_memory=False)
        csv_outpath = "{}/{}".format(outpath_csv, "ver1")
        if(not os.path.isdir(csv_outpath)):
            os.makedirs(csv_outpath)
        for k in range(len(bdf), len(train_feature_rs)):
            tf = df.copy()
            sdf = pd.DataFrame(train_feature_rs[k], columns=["val"])
            tf["val"] = sdf["val"]

            tid = tf["opdno"].head(1)[0]
            # ver1 : only val smote
            tf["opdno"] = "{}_{}".format("Smote", tid)
            tf.to_csv("{}/{}/{}_{}_{}.csv.gz".format(outpath_csv, "ver1",
                                                     "Smote", tid, k), index=False, compression='gzip')
        print(fileNames[0])
    except Exception as e:
        # print(testdf)
        # traceback.print_exception(e)
        print("處理失敗：{}=>{}".format(fileNames[run_file], e))
        fail.append("{}=>{}".format(fileNames[run_file], e))


if __name__ == "__main__":
    # import multiprocessing
    Main()
