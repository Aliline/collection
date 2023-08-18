import os
import multiprocessing
import tqdm as tq
import datetime
import json
import pandas as pd
import numpy as np
import traceback
import matplotlib.pyplot as plt
import threading
import swifter
from multiprocessing import current_process

original_path = "C:/Users/USER/Downloads/Compressed/vital sign data/HS/"
useCore = 6
# filter_day = 3  # 天數篩選標準_3天
# filter_biomarker_num = 2  # 一個biomarker測量的次數
outpath_csv = "D:/comdata/"
filter_day = 3
run_file = 2

os.system("cls")  # 清除螢幕
# sex_filtered = 0
# age_filtered = 0
# sex_lock = threading.Lock()
# age_lock = threading.Lock()


def Main():
    # 讀取原始資料

    args = []
    ready_run = 0
    for path, folders, files in os.walk(original_path):  # walk遞迴印出資料夾中所有目錄及檔名
        file_id = 0
        age_list = multiprocessing.Manager().list()
        sex_list = multiprocessing.Manager().list()
        for file in files:
            # os.path.join()：將多個路徑組合後返回
            args.append([os.path.join(path, file), file, age_list, sex_list])
            file_id = file_id + 1
            ready_run = ready_run + 1
            if ((ready_run == run_file) or (file_id == len(files))):
                with multiprocessing.Pool(processes=useCore, maxtasksperchild=6000) as pool:
                    pool.starmap(process, args)
                    print("{}處理完成{}".format("="*10, "="*10))
                ready_run = 0
                args = []
    total_sex = 0
    total_age = 0
    test_age = "各年齡被過濾:"
    test_sex = "各性別被過濾:"
    for ts in sex_list:
        total_sex = total_sex + ts
        test_sex = "{}{},".format(test_sex, ts)
    for ta in age_list:
        total_age = total_age + ta
        test_age = "{}{},".format(test_age, ta)
    txtf = open("{}filter_doc.txt".format(outpath_csv), mode="w")
    txtf.write("性別被過濾:{}".format(total_sex))
    txtf.write("年齡被過濾:{}".format(total_age))
    txtf.writelines(test_age)
    txtf.writelines(test_sex)
    txtf.close()


def filt_sex(opid, df, tl):

    tsdf = df[(df["opdno"] == opid)].copy()
    tsdf = tsdf.drop_duplicates(subset=['sex'], keep='first')
    if (len(tsdf) > 1):
        # print(tsdf)
        tl.append(opid)


def filt_age(opid, df, tl):

    tsdf = df[(df["opdno"] == opid)].copy()
    tsdf = tsdf.drop_duplicates(subset=['age'], keep='first')
    if (len(tsdf) > 1):
        # print(tsdf)
        tl.append(opid)


def fix_sex(slist, x):

    # with global_index.sex_lock:
    #     print("before:{}".format(global_index.sex_filtered))
    #     global_index.sex_filtered = global_index.sex_filtered + x
    #     print("after:{}".format(global_index.sex_filtered))
    print("before:{}".format(len(slist)))
    slist.append(x)
    print("after:{}".format(len(slist)))


def fix_age(alist, x):
    # with global_index.age_lock:
    #     print("before:{}".format(global_index.age_filtered))
    #     global_index.age_filtered = global_index.age_filtered + x
    #     print("after:{}".format(global_index.age_filtered))
    print("before:{}".format(len(alist)))
    alist.append(x)
    print("before:{}".format(len(alist)))


def timeprocess(x):
    # print(x)
    try:

        return (datetime.date(int(str(x)[0:4]),
                              int(str(x)[4:6]), int(str(x)[6:8])))
    except:
        # return "no"
        x = x
        # print("here")
        # print(x)


def valprocess(x):
    try:
        x = float(x)
        return x
    except:
        x = x


def iptmprocess(x):
    try:
        if x == "    ":
            return ""
        return str(int(x))
    except:
        temp = str(x).replace(".", "0")
        temp = str(temp).replace(",", "0")
        temp = str(temp).replace("!", "0")
        temp = str(temp).replace(";", "0")
        temp = str(temp).replace("'", "0")
        temp = str(temp).replace("/", "0")
        temp = str(temp).replace("\\", "0")
        temp = str(temp).replace("[", "0")
        temp = str(temp).replace("]", "0")
        temp = str(temp).replace("|", "0")
        temp = str(temp).replace("-", "0")
        temp = str(temp).replace("=", "0")
        temp = str(temp).replace("+", "0")
        temp = str(temp).replace("*", "0")
        temp = str(temp).replace("-", "0")
        temp = str(temp).replace("_", "0")
        temp = str(temp).replace("(", "0")
        temp = str(temp).replace(")", "0")
        temp = str(temp).replace("&", "0")
        temp = str(temp).replace("^", "0")
        temp = str(temp).replace("%", "0")
        temp = str(temp).replace("#", "0")
        temp = str(temp).replace("@", "0")
        temp = str(temp).replace("$", "0")
        temp = str(temp).replace("~", "0")
        temp = str(temp).replace("`", "0")
        temp = str(temp).replace("{", "0")
        temp = str(temp).replace("}", "0")
        temp = str(temp).replace(":", "0")
        temp = str(temp).replace("<", "0")
        temp = str(temp).replace(">", "0")
        temp = str(temp).replace(" ", "0")
        # print(temp)
        return str(int(temp))


def process(filePath, fileName, a_list, s_list):
    # if(not "ED" in filePath):
    #   return
    print("處理{}".format(fileName))

    if ("E" in fileName):
        itnmType = "itnm2"
        leaveType = "disdat"
    else:
        itnmType = "itnm3"
        leaveType = "dgdat"
    if ("D" in fileName):
        ftype = "D"
    else:
        ftype = "S"
    csv_outpath = "{}{}".format(outpath_csv, ftype)
    nowdt = ""
    testdf = ""
    try:
        dtypes = {itnmType: "category"}
        usecols = ["idcode", "opdno", "csn",
                   "ipdat", "iptm", "itid", "age", "sex", itnmType, leaveType, "ipday", "val"]  # 使用欄位
        df = pd.read_csv(filePath, usecols=usecols,
                         low_memory=False)

        # print(df.head())
        temp = []
        nowid = ""
        ed = 0
        chunk = df
        if(leaveType == "dgdat"):
            chunk = chunk.rename(columns={"dgdat": "disdat"})
        chunk["disdat"] = chunk["disdat"].astype(str).str[:8]
        chunk = chunk.dropna(subset=["val"])
        chunk = chunk.dropna(subset=["iptm"])
        chunk = chunk.dropna(subset=["ipday"])
        # chunk["iptm"] = chunk["iptm"].fillna("0000")
        # chunk["disdat"] = pd.to_numeric(chunk["disdat"], errors='coerce')
        # chunk = chunk.dropna(subset=["disdat"])

        chunk["disdat"] = chunk["disdat"].apply(
            lambda x: timeprocess(x))
        # chunk["disdat"] = chunk["temp"]
        # chunk = (chunk[chunk['disdat'] != "no"])
        chunk["ipdat"] = chunk["ipdat"].apply(lambda x: timeprocess(x))
        # chunk["ipdat"] = chunk["temp"]
        # chunk = (chunk[chunk['ipdat'] != "no"])
        chunk["ipday"] = chunk["ipday"].apply(
            lambda x: timeprocess(x))
        # chunk = (chunk[chunk['ipday'] != "no"])
        chunk["iptm"] = chunk["iptm"].apply(
            lambda x: iptmprocess(x))
        chunk["val"] = chunk["val"].apply(lambda x: valprocess(x))
        chunk = chunk.dropna(subset=["val"])
        chunk = chunk.dropna(subset=["iptm"])
        chunk = chunk.dropna(subset=["ipday"])
        # chunk["ipday"] = chunk["ipday"].apply(lambda x: datetime.date(
        #     int(str(x)[0:4]), int(str(x)[4:6]), int(str(x)[6:8])))
        # 日期轉換、天數計算
        chunk['day'] = (pd.to_datetime(
            chunk["disdat"])-pd.to_datetime(chunk['ipdat'])).apply(lambda x: x.days)
        # 篩選不滿filter_day
        chunk = (chunk[chunk['day'] >= filter_day])
        chunk = chunk.drop(['day'], axis=1)
        nchunk = chunk.sort_values(by=["opdno"])
        oplist = nchunk['opdno'].copy()
        oplist = oplist.drop_duplicates(keep='first')
        # 篩選性別或年齡大於1種
        # nowsex = ''
        # testid_sex = ''
        # problemopdno_sex = []
        # nowage = ''
        # testid_age = ''
        # problemopdno_age = []
        filtered_age = 0
        filtered_sex = 0
        tl = []
        oplist.swifter.progress_bar(enable=True, desc="filtering sex").apply(
            lambda x: filt_sex(x, nchunk, tl))
        filtered_sex = len(tl)
        for y in tl:
            # print('problemopdno:sex', y)
            nchunk = nchunk[nchunk['opdno'] != y]
            oplist = oplist[oplist != y]

        # nchunk = nchunk[~nchunk['opdno'].isin(tl)]
        tl = []
        oplist.swifter.progress_bar(enable=True, desc="filtering age").apply(
            lambda x: filt_age(x, nchunk, tl))
        filtered_age = len(tl)
        for y in tl:
            # print('problemopdno:age', y)
            nchunk = nchunk[nchunk['opdno'] != y]
            oplist = oplist[oplist != y]
        # nchunk = nchunk[~nchunk['opdno'].isin(tl)]
        # for index, nowrow in nchunk.iterrows():
        #     # print(nowrow['opdno'], nowrow['sex'])
        #     if(testid_sex == ''):
        #         testid_sex = nowrow['opdno']
        #     if(testid_age == ''):
        #         testid_age = nowrow['opdno']
        #     if(testid_sex != nowrow['opdno']):
        #         nowsex = ''
        #         testid_sex = nowrow['opdno']
        #     if(testid_age != nowrow['opdno']):
        #         nowage = ''
        #         testid_age = nowrow['opdno']
        #     if(nowsex == ''):
        #         nowsex = nowrow['sex']
        #         # print('id', testid_sex, 'nowsex', nowsex)
        #     elif(nowrow['sex'] != nowsex):
        #         # print('id', nowrow['opdno'], 'nowsex',
        #         #       nowrow['sex'], 'oldsex', nowsex, fileName, 'wrong data')
        #         if nowrow['opdno'] not in problemopdno_sex:
        #             problemopdno_sex.append(nowrow['opdno'])
        #     if(nowage == ''):
        #         nowage = nowrow['age']
        #         # print('id', testid_age, 'nowage', nowage)
        #     elif(nowrow['age'] != nowage):
        #         # print('id', nowrow['opdno'], 'nowage',
        #         #       nowrow['age'], 'oldage', nowage, fileName, 'wrong data')
        #         if nowrow['opdno'] not in problemopdno_age:
        #             if nowrow['opdno'] not in problemopdno_sex:
        #                 problemopdno_age.append(nowrow['opdno'])

        fix_age(a_list, filtered_age)
        fix_sex(s_list, filtered_sex)
        print(fileName, "progressing...")
        current = current_process()
        pos = current._identity[0]-1
        itnmChecked = 0
        nchunk = nchunk.reset_index(drop=True)
        for i, row in tq.tqdm(nchunk.iterrows(), total=nchunk.shape[0]):
            if itnmChecked == 0:
                tempchunk = nchunk[(nchunk['opdno'] == row['opdno'])]
                if(tempchunk[(tempchunk[itnmType] == "BT")].empty or tempchunk[(tempchunk[itnmType] == "HR")].empty or tempchunk[(tempchunk[itnmType] == "NBP D")].empty or tempchunk[(tempchunk[itnmType] == "NBP S")].empty or tempchunk[(tempchunk[itnmType] == "RR")].empty):
                    csv_outpath = "{}{}{}".format(outpath_csv, "pdata/", ftype)
                    if(not os.path.isdir(csv_outpath)):
                        os.makedirs(csv_outpath)
                itnmChecked = 1
            if nowid == "":
                nowid = row['opdno']

            if nowid == row['opdno']:
                k = row['iptm']
                if k == '083.':
                    print(k)
                nk = str(int(k))
                l = len(nk)
                if l < 4:
                    for te in range(4-l):
                        nk = "0{}".format(nk)
                row['iptm'] = nk
                temp.append(row)
                # temp.append((df[(df['opdno'] == row['opdno'])]))
                # t = (df[(df['opdno'] == row['opdno'])]).reset_index()
                # ed = 1

            elif nowid != row['opdno'] or i == len(nchunk):
                # debug用 判斷如果沒有輸入資料時可以debug
                if(temp == []):
                    print(fileName, ":", nowid)
                nowid = row['opdno']
                if(not os.path.isdir(csv_outpath)):
                    os.makedirs(csv_outpath)
                # 檢查指定路徑存不存在
                if(os.path.isdir(csv_outpath)):
                    data = pd.DataFrame(temp)
                    testdf = data
                    ndata = data.sort_values(by=[itnmType, "ipday", "iptm"])
                    newData = []
                    prvtm = ""
                    prvd = ""
                    nowtm = ""
                    prvitem = ""
                    prvdata = []
                    # 開始填表
                    for k, nr in ndata.iterrows():
                        if ((prvitem != "") & (prvitem != nr[itnmType])):
                            prvtm = ""

                        # 如果現在是第一筆走訪資料
                        if prvtm == "":
                            if nr['iptm'][0:2] == '083.':
                                print(nr['iptm'])
                            prvtm = int(nr['iptm'][0:2])
                            prvd = nr['ipday']
                            prvitem = nr[itnmType]
                            prvdata = nr.copy()

                        else:
                            # print("prvd:", prvd)
                            # print("prvtm:", prvtm)
                            if nr['iptm'][0:2] == '083.':
                                print(nr['iptm'])
                            nowtm = int(nr['iptm'][0:2])
                            nowd = nr['ipday']
                            # print("nowd:", nowd)
                            # print("nowtm:", nowtm)
                            if type(pd.to_datetime(nowd)) == 'NoneType':
                                print(nr)
                            d = (pd.to_datetime(
                                nowd)-pd.to_datetime(prvd)).days
                            h = nowtm - prvtm
                            h = h + (int(d)*24)
                            # 過濾相差時間<=1的情況
                            if h > 1:
                                plusDay = 0  # 天數
                                plusToken = 0  # 現在天數
                                plusHour = 0  # 時間
                                # 從1~相差時間-1
                                for pl in range(1, h):
                                    tr = prvdata.copy()  # 複製
                                    plusHour = prvtm + pl
                                    # 如果填入時間超過24小時

                                    # 如果出現跨天(i > 24)

                                    if plusHour >= 24:
                                        # 判斷是否跨天 ex 48(第二天) - 24(第一天)
                                        if plusHour - plusToken >= 24:
                                            plusToken = plusToken + 24
                                            plusDay = plusDay + 1
                                    while plusHour >= 24:
                                        plusHour = abs(plusHour - 24)
                                    tr['ipday'] = (
                                        pd.to_datetime(prvd) + datetime.timedelta(days=plusDay))
                                    addz = ""
                                    if plusHour <= 9:
                                        addz = "0"
                                    why = "{}{}00".format(addz, plusHour)
                                    # print(why)
                                    tr['iptm'] = why
                                    # print(tr)
                                    newData.append(tr)
                                    # print("prvd:", prvd)
                                    # print("prvtm:", prvtm)
                                    # print("trday:", tr['ipday'])
                                    # print("trtm:", tr['iptm'])
                                    # print("nowd:", nowd)
                                    # print("nowtm:", nowtm)

                        prvtm = int(nr['iptm'][0:2])
                        prvd = nr['ipday']
                        prvdata = nr.copy()
                        # print(nr)
                        newData.append(nr)
                        # print(newd)

                    ndata = pd.DataFrame(newData).reset_index(drop=True)
                    nid = ndata["opdno"].head(1)[0]
                    # print("填表完成")
                    ndata = ndata.rename(columns={itnmType: "itnm"})
                    ndata.to_csv("{}/{}_{}.csv".format(csv_outpath, ftype,
                                                       nid), index=False)

                    # t.to_csv("{}/{}_{}.csv".format(csv_outpath, ftype,
                    #                                row["opdno"]), index=False)
                temp = []
                itnmChecked = 0
                csv_outpath = "{}{}".format(outpath_csv, ftype)
                if itnmChecked == 0:
                    tempchunk = nchunk[(nchunk['opdno'] == row['opdno'])]
                    if(tempchunk[(tempchunk[itnmType] == "BT")].empty or tempchunk[(tempchunk[itnmType] == "HR")].empty or tempchunk[(tempchunk[itnmType] == "NBP D")].empty or tempchunk[(tempchunk[itnmType] == "NBP S")].empty or tempchunk[(tempchunk[itnmType] == "RR")].empty):
                        csv_outpath = "{}{}{}".format(
                            outpath_csv, "pdata/", ftype)
                        if(not os.path.isdir(csv_outpath)):
                            os.makedirs(csv_outpath)
                    itnmChecked = 1
                if nowid == row['opdno']:
                    k = row['iptm']
                    nk = str(int(k))
                    l = len(nk)
                    if l < 4:
                        for te in range(4-l):
                            nk = "0{}".format(nk)
                    row['iptm'] = nk
                    temp.append(row)
                    # temp.append((df[(df['opdno'] == row['opdno'])]))
                    # t = (df[(df['opdno'] == row['opdno'])]).reset_index()
                    # ed = 1

        print('{}處理完成'.format(fileName))
    except Exception as e:
        # print(testdf)
        # traceback.print_exception(e)
        traceback.print_exc()
        print("處理失敗：{}=>{}".format(fileName, e))


if __name__ == "__main__":

    Main()
