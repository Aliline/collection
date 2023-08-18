import os
import multiprocessing
import tqdm as tq
import datetime
import json
import pandas as pd
import numpy as np
import traceback


import matplotlib.pyplot as plt
from multiprocessing import current_process
original_path = "D:/comdata/S/"
useCore = 6
# filter_day = 3  # 天數篩選標準_3天
# filter_biomarker_num = 2  # 一個biomarker測量的次數
outpath_csv = "D:/comdata/"
outpath_pic = "D:/picdata/"
filter_day = 3
run_file = 50  # 設定一次要跑幾個檔案
os.system("cls")  # 清除螢幕
fail = []
file_limit = 132000  # 設定檔案上限


def Main():
    # 讀取原始資料
    args = []
    ready_run = 0
    file_runed = 0
    for path, folders, files in os.walk(original_path):  # walk遞迴印出資料夾中所有目錄及檔名
        file_id = 0
        pic_list = multiprocessing.Manager().list()
        for file in files:
            # os.path.join()：將多個路徑組合後返回

            args.append([os.path.join(path, file), file, pic_list])
            file_id = file_id + 1
            ready_run = ready_run + 1
            if ((ready_run == run_file) or (file_id == len(files))):
                with multiprocessing.Pool(processes=useCore, maxtasksperchild=6000) as pool:
                    pool.starmap(process, args)
                    print("{}處理完成{}".format("="*10, "="*10))
                ready_run = 0
                args = []
                # print(len(pic_list))
                for x in pic_list:
                    file_runed = file_runed + x
                pic_list[:] = []
                if file_runed >= file_limit:
                    break
        if file_runed >= file_limit:
            break
    if len(fail) >= 1:
        for k in fail:
            print(k)


def iptmProcess(x):
    if x == 0:
        return ("0000")
    elif len(str(x)) <= 3:
        tstr = ""
        for k in range(4-len(str(x))):
            tstr = tstr + "0"
        return ("{}{}".format(tstr, str(x)))

    else:
        return str(x)


def tetmCheck(x):

    tempk = int(x[0:2])
    tempm = int(x[2:4])


def tetmHour(x):

    return(int(x[0:2]))


def tetmMin(x):

    return(int(x[2:4]))


def creatXtime(tetm, ipday):
    temp = pd.to_datetime(ipday)
    hour = str(tetm)[0:2]
    mins = str(tetm)[2:4]
    xtime = temp + \
        datetime.timedelta(hours=int(hour), minutes=int(mins))
    return xtime


def genPic(df, mode, outpath, ty, fty, id):
    plt.figure(figsize=(2, 1.5))
    if mode == 0:

        plt.plot(df["xtime"], df["val"], c="gray")

    elif mode == 1:
        plt.plot(df["xtime"], df["sex"], c="gray")
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
    elif mode == 2:
        plt.plot(df["xtime"], df["age"], c="gray")
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)

    # plt.show()

    # plt.set_cmap('gray')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # plt.axis('off')
    # plt.show()
    outpath = "{}/{}".format(outpath, ty)
    if(not os.path.isdir(outpath)):
        os.makedirs(outpath)
    plt.savefig("{}/{}_{}_{}.jpg".format(outpath, fty, ty, id),
                box_inches='tight',               # 去除座標軸占用的空間
                pad_inches=0.0, format='jpg')
    plt.close()


def process(filePath, fileName, pic_list):
    # if(not "ED" in filePath):
    #   return
    ftype = fileName[0:1]
    pic_outpath = "{}{}".format(outpath_pic,
                                ftype)

    print("處理{}".format(fileName))

    dch = [".csv"]

    try:
        motor = pd.read_csv(filePath,
                            low_memory=False)
        # print(motor.head(3))    # 顯示前3筆資料
        # print(motor["ipday"].head(3))
        # print('{}處理完成'.format(fileName))
        motor["tetm"] = motor["iptm"].apply(
            lambda x: iptmProcess(x))
        motor["hour"] = motor["tetm"].apply(lambda x: tetmHour(x))
        motor["min"] = motor["tetm"].apply(lambda x: tetmMin(x))
        colNames = ['tetm', 'ipday']
        tedf = pd.DataFrame(motor[["tetm", "ipday"]], columns=colNames)
        motor["xtime"] = tedf.apply(
            lambda x: creatXtime(x[colNames[0]], x[colNames[1]]), axis=1)
        tedf = ""

        # data = []
        # for i, row in motor.iterrows():
        #     temp = pd.to_datetime(row["ipday"])
        #     try:
        #         hour = str(row["tetm"])[0:2]
        #         mins = str(row["tetm"])[2:4]
        #         xtime = temp + \
        #             datetime.timedelta(hours=int(hour), minutes=int(mins))
        #         data.append(xtime)
        #     except Exception as e:
        #         print(row["tetm"])
        # motor["xtime"] = pd.DataFrame(data)
        BTdf = motor[(motor["itnm"] == "BT")]
        HRdf = motor[(motor["itnm"] == "HR")]
        NBPDdf = motor[(motor["itnm"] == "NBP D")]
        NBPSdf = motor[(motor["itnm"] == "NBP S")]
        RRdf = motor[(motor["itnm"] == "RR")]
        if(len(BTdf) <= 1 or len(HRdf) <= 1 or len(NBPDdf) <= 1 or len(NBPSdf) <= 1 or len(RRdf) <= 1):
            pic_outpath = "{}pdata/{}".format(outpath_pic,
                                              ftype)
        # for i, row in BTdf.head(3).iterrows():
        #     # row["itnm"] = "hello"
        #     print(row)
        # plt.plot(BTdf["xtime"], BTdf["val"], c="r")
        # ax = plt.gca()
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # text = []
        # plt.xticks(BTdf["xtime"], text)

        genPic(BTdf, 0, pic_outpath, "BT",
               ftype, motor["opdno"][0])

        genPic(HRdf, 0, pic_outpath, "HR",
               ftype, motor["opdno"][0])

        genPic(NBPDdf, 0, pic_outpath, "NBPD",
               ftype, motor["opdno"][0])

        genPic(NBPSdf, 0, pic_outpath, "NBPS",
               ftype, motor["opdno"][0])

        genPic(RRdf, 0, pic_outpath, "RR",
               ftype, motor["opdno"][0])

        genPic(motor, 1, pic_outpath, "SEX",
               ftype, motor["opdno"][0])

        genPic(motor, 2, pic_outpath, "AGE",
               ftype, motor["opdno"][0])

        # def pic(var):
        #     if var == 0:
        #         genPic(BTdf, 0, pic_outpath, "BT",
        #                ftype, motor["opdno"][0])
        #     elif var == 1:
        #         genPic(HRdf, 0, pic_outpath, "HR",
        #                ftype, motor["opdno"][0])
        #     elif var == 2:
        #         genPic(NBPDdf, 0, pic_outpath, "NBPD",
        #                ftype, motor["opdno"][0])
        #     elif var == 3:
        #         genPic(NBPSdf, 0, pic_outpath, "NBPS",
        #                ftype, motor["opdno"][0])
        #     elif var == 4:
        #         genPic(RRdf, 0, pic_outpath, "RR",
        #                ftype, motor["opdno"][0])
        #     elif var == 5:
        #         genPic(motor, 1, pic_outpath, "SEX",
        #                ftype, motor["opdno"][0])
        #     elif var == 6:
        #         genPic(motor, 2, pic_outpath, "AGE",
        #                ftype, motor["opdno"][0])
        # return{
        #     '0': genPic(BTdf, 0, pic_outpath),
        #     '1': genPic(HRdf, 0, pic_outpath),
        #     '2': genPic(NBPDdf, 0, pic_outpath),
        #     '3': genPic(NBPSdf, 0, pic_outpath),
        #     '4': genPic(RRdf, 0, pic_outpath),
        #     '5': genPic(motor, 1, pic_outpath),
        #     '6': genPic(motor, 2, pic_outpath)
        # }
        # for index in range(7):
        #     pic(index)
        # print(index)
        pic_list.append(1)
    except Exception as e:
        # print(testdf)
        # traceback.print_exception(e)
        print("處理失敗：{}=>{}".format(fileName, e))
        fail.append("{}=>{}".format(fileName, e))

    # for te in pic_list:
    #     print(te)


if __name__ == "__main__":
    Main()
