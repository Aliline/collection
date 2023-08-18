import numpy as np
import cv2
import os
import random
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Input, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.metrics import Recall
from sklearn.metrics import confusion_matrix
all_file_limit = 800
file_per_run = 200
pathT = 'D:/tempdata/'
if(not os.path.isdir(pathT)):
    os.makedirs(pathT)
pathM = "{}{}".format(pathT, "CNN_v3.h5")
pathT = "{}{}".format(pathT, "file_runed_v3.csv")


file_runed_S = []
file_runed_D = []
pathPD = 'D:/picdata/mixD/'
pathPS = 'D:/picdata/S/'
BT = []
HR = []
NBPD = []
NBPS = []
RR = []
AGE = []
SEX = []
target = []


def chk_itnm(itnm, img):
    if itnm == "BT":
        BT.append(img)
    elif itnm == "HR":
        HR.append(img)
    elif itnm == "NBPD":
        NBPD.append(img)
    elif itnm == "NBPS":
        NBPS.append(img)
    elif itnm == "RR":
        RR.append(img)
    elif itnm == "AGE":
        AGE.append(img)
    elif itnm == "SEX":
        SEX.append(img)


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def pic_read(path, create, df, status):
    done = 0
    final = 0
    for parent, dirnames, filename in os.walk(path):
        if dirnames == []:
            done = 1
            break
        if create == 1:
            sorted_dir = dirnames
            if status == 0:
                df["file_runed_S"].apply(lambda x: sorted_dir.remove(x))
            else:
                df["file_runed_D"].apply(lambda x: sorted_dir.remove(x))
            if sorted_dir == []:
                done = 1
                break
            if len(sorted_dir) <= file_per_run:
                final = 1
        if final == 0:
            while True:
                if create == 0:
                    sorted_dir = dirnames

                dirname = random.choice(sorted_dir)
                if status == 0:
                    if file_runed_S != []:
                        if dirname in file_runed_S:
                            while dirname in file_runed_S:
                                dirname = random.choice(sorted_dir)
                else:
                    if file_runed_D != []:
                        if dirname in file_runed_D:
                            while dirname in file_runed_D:
                                dirname = random.choice(sorted_dir)

                td = os.path.join(path, dirname)

                checked = 0

                if checked == 0:
                    for inpath, folders, files in os.walk(td):
                        for file in files:
                            t = file.rstrip(".jpeg")
                            itnm, l = t.split("_")

                            img = cv2.imread(os.path.join(inpath, file),
                                             cv2.IMREAD_GRAYSCALE)
                            chk_itnm(itnm, img)
                        if l == 'D':
                            target.append(1)
                            file_runed_D.append(dirname)
                        else:
                            target.append(0)
                            file_runed_S.append(dirname)

                        yield 1
                else:
                    yield 0
        else:
            for dirname in sorted_dir:
                td = os.path.join(path, dirname)

                checked = 0

                if checked == 0:
                    for inpath, folders, files in os.walk(td):
                        for file in files:
                            t = file.rstrip(".jpeg")
                            itnm, l = t.split("_")

                            img = cv2.imread(os.path.join(inpath, file),
                                             cv2.IMREAD_GRAYSCALE)
                            chk_itnm(itnm, img)
                        if l == 'D':
                            target.append(1)
                            file_runed_D.append(dirname)
                        else:
                            target.append(0)
                            file_runed_S.append(dirname)

                        yield 1
                else:
                    yield 0

    if done == 1:
        yield "done"


def x_normalize(x_train, x_test):
    x_train_normalize = np.array(x_train).astype('float32')/255
    x_test_normalize = np.array(x_test).astype('float32')/255
    x_train_normalize = tf.expand_dims(x_train_normalize, axis=-1)
    x_test_normalize = tf.expand_dims(x_test_normalize, axis=-1)

    return x_train_normalize, x_test_normalize


tr = 0
while tr < all_file_limit:
    file_runed_S.clear()
    file_runed_D.clear()
    BT.clear()
    HR.clear()
    NBPS.clear()
    NBPD.clear()
    RR.clear()
    AGE.clear()
    SEX.clear()
    target.clear()
    if (os.path.isfile(pathT)):
        create = 1
        df = pd.read_csv(pathT,
                         low_memory=False)
    else:
        create = 0
        df = ""

    g = pic_read(pathPS, create, df, 0)
    runed = 0
    while runed < file_per_run:
        try:
            runed = runed + next(g)
            # if runed == 250:
            #     print(runed)
        except TypeError as e:
            break
    g.close()

    g = pic_read(pathPD, create, df, 1)
    runed = 0
    while runed < file_per_run:
        try:
            runed = runed + next(g)
            # if runed == 250:
            #     print(runed)
        except TypeError as e:
            break
    g.close()

    ndf = pd.DataFrame(
        {"file_runed_S": file_runed_S, "file_runed_D": file_runed_D}).reset_index(drop=True)
    if create == 0:
        df = ndf
    else:

        df = pd.concat([df, ndf], ignore_index=True)

# 續跑功能完成
# 寫load model與save model
    data = [BT, HR, NBPD, NBPS, RR, AGE, SEX]
    BT_train, BT_test, y_label, y_test = train_test_split(
        BT, target, test_size=0.2, random_state=0)
    HR_train, HR_test, y_label, y_test = train_test_split(
        HR, target, test_size=0.2, random_state=0)
    NBPD_train, NBPD_test, y_label, y_test = train_test_split(
        NBPD, target, test_size=0.2, random_state=0)
    NBPS_train, NBPS_test, y_label, y_test = train_test_split(
        NBPS, target, test_size=0.2, random_state=0)
    RR_train, RR_test, y_label, y_test = train_test_split(
        RR, target, test_size=0.2, random_state=0)
    AGE_train, AGE_test, y_label, y_test = train_test_split(
        AGE, target, test_size=0.2, random_state=0)
    SEX_train, SEX_test, y_label, y_test = train_test_split(
        SEX, target, test_size=0.2, random_state=0)
    BT_train_normalize, BT_test_normalize = x_normalize(BT_train, BT_test)
    HR_train_normalize, HR_test_normalize = x_normalize(HR_train, HR_test)
    NBPD_train_normalize, NBPD_test_normalize = x_normalize(
        NBPD_train, NBPD_test)
    NBPS_train_normalize, NBPS_test_normalize = x_normalize(
        NBPS_train, NBPS_test)
    RR_train_normalize, RR_test_normalize = x_normalize(RR_train, RR_test)
    AGE_train_normalize, AGE_test_normalize = x_normalize(AGE_train, AGE_test)
    SEX_train_normalize, SEX_test_normalize = x_normalize(SEX_train, SEX_test)
    y_label_onehot = np.array(y_label)
    y_test_onehot = np.array(y_test)
    if create == 0:
        # v1
        visibleBT = Input(shape=(60, 60, 1))
        convBT = Conv2D(9, kernel_size=(3, 3),
                        padding='same')(visibleBT)
        BNBT = BatchNormalization()(convBT)
        actBT = Activation(activation='relu')(BNBT)
        dropBT = Dropout(0.6)(actBT)
        poolBT = MaxPooling2D(pool_size=(2, 2))(dropBT)
        convBT = Conv2D(18, kernel_size=(3, 3),
                        padding='same', activation='relu')(poolBT)
        BNBT = BatchNormalization()(convBT)
        actBT = Activation(activation='relu')(BNBT)
        dropBT = Dropout(0.6)(actBT)
        poolBT = MaxPooling2D(pool_size=(2, 2))(dropBT)
        # dropedBT = Dropout(0.3)(poolBT)
        flatBT = Flatten()(poolBT)
        # v2
        visibleHR = Input(shape=(60, 60, 1))
        convHR = Conv2D(9, kernel_size=(3, 3),
                        padding='same')(visibleHR)
        BNHR = BatchNormalization()(convHR)
        actHR = Activation(activation='relu')(BNHR)
        dropHR = Dropout(0.6)(actHR)
        poolHR = MaxPooling2D(pool_size=(2, 2))(dropHR)
        convHR = Conv2D(18, kernel_size=(3, 3),
                        padding='same', activation='relu')(poolHR)
        BNHR = BatchNormalization()(convHR)
        actHR = Activation(activation='relu')(BNHR)
        dropHR = Dropout(0.6)(actHR)
        poolHR = MaxPooling2D(pool_size=(2, 2))(dropHR)
        # dropedHR = Dropout(0.3)(poolHR)
        flatHR = Flatten()(poolHR)
        # v3
        visibleNBPD = Input(shape=(60, 60, 1))
        convNBPD = Conv2D(9, kernel_size=(3, 3),
                          padding='same')(visibleNBPD)
        BNNBPD = BatchNormalization()(convNBPD)
        actNBPD = Activation(activation='relu')(BNNBPD)
        dropNBPD = Dropout(0.6)(actNBPD)
        poolNBPD = MaxPooling2D(pool_size=(2, 2))(dropNBPD)
        convNBPD = Conv2D(18, kernel_size=(3, 3),
                          padding='same', activation='relu')(poolNBPD)
        BNNBPD = BatchNormalization()(convNBPD)
        actNBPD = Activation(activation='relu')(BNNBPD)
        dropNBPD = Dropout(0.6)(actNBPD)
        poolNBPD = MaxPooling2D(pool_size=(2, 2))(dropNBPD)
        # dropedNBPD = Dropout(0.3)(poolNBPD)
        flatNBPD = Flatten()(poolNBPD)
        # v4
        visibleNBPS = Input(shape=(60, 60, 1))
        convNBPS = Conv2D(9, kernel_size=(3, 3),
                          padding='same')(visibleNBPS)
        BNNBPS = BatchNormalization()(convNBPS)
        actNBPS = Activation(activation='relu')(BNNBPS)
        dropNBPS = Dropout(0.6)(actNBPS)
        poolNBPS = MaxPooling2D(pool_size=(2, 2))(dropNBPS)
        convNBPS = Conv2D(18, kernel_size=(3, 3),
                          padding='same', activation='relu')(poolNBPS)
        BNNBPS = BatchNormalization()(convNBPS)
        actNBPS = Activation(activation='relu')(BNNBPS)
        dropNBPS = Dropout(0.6)(actNBPS)
        poolNBPS = MaxPooling2D(pool_size=(2, 2))(dropNBPS)
        # dropedNBPS = Dropout(0.3)(poolNBPS)
        flatNBPS = Flatten()(poolNBPS)
        # v5
        visibleRR = Input(shape=(60, 60, 1))
        convRR = Conv2D(9, kernel_size=(3, 3),
                        padding='same')(visibleRR)
        BNRR = BatchNormalization()(convRR)
        actRR = Activation(activation='relu')(BNRR)
        dropRR = Dropout(0.6)(actRR)
        poolRR = MaxPooling2D(pool_size=(2, 2))(dropRR)
        convRR = Conv2D(18, kernel_size=(3, 3),
                        padding='same', activation='relu')(poolRR)
        BNRR = BatchNormalization()(convRR)
        actRR = Activation(activation='relu')(BNRR)
        dropRR = Dropout(0.6)(actRR)
        poolRR = MaxPooling2D(pool_size=(2, 2))(dropRR)
        # dropedRR = Dropout(0.3)(poolRR)
        flatRR = Flatten()(poolRR)
        # v6
        visibleAGE = Input(shape=(60, 60, 1))
        convAGE = Conv2D(9, kernel_size=(3, 3),
                         padding='same')(visibleAGE)
        BNAGE = BatchNormalization()(convAGE)
        actAGE = Activation(activation='relu')(BNAGE)
        dropAGE = Dropout(0.6)(actAGE)
        poolAGE = MaxPooling2D(pool_size=(2, 2))(dropAGE)
        convAGE = Conv2D(18, kernel_size=(3, 3),
                         padding='same', activation='relu')(poolAGE)
        BNAGE = BatchNormalization()(convAGE)
        actAGE = Activation(activation='relu')(BNAGE)
        dropAGE = Dropout(0.6)(actAGE)
        poolAGE = MaxPooling2D(pool_size=(2, 2))(dropAGE)
        # dropedAGE = Dropout(0.3)(poolAGE)
        flatAGE = Flatten()(poolAGE)
        # v7
        visibleSEX = Input(shape=(60, 60, 1))
        convSEX = Conv2D(9, kernel_size=(3, 3),
                         padding='same')(visibleSEX)
        BNSEX = BatchNormalization()(convSEX)
        actSEX = Activation(activation='relu')(BNSEX)
        dropSEX = Dropout(0.6)(actSEX)
        poolSEX = MaxPooling2D(pool_size=(2, 2))(dropSEX)
        convSEX = Conv2D(18, kernel_size=(3, 3),
                         padding='same', activation='relu')(poolSEX)
        BNSEX = BatchNormalization()(convSEX)
        actSEX = Activation(activation='relu')(BNSEX)
        dropSEX = Dropout(0.6)(actSEX)
        poolSEX = MaxPooling2D(pool_size=(2, 2))(dropSEX)
        # dropedSEX = Dropout(0.3)(poolSEX)
        flatSEX = Flatten()(poolSEX)
        # marge
        merge = concatenate([flatBT, flatHR, flatHR, flatNBPD,
                             flatNBPS, flatRR, flatAGE, flatSEX])

        hidden1 = Dense(512, activation='relu')(merge)
        hidden2 = Dense(512, activation='relu')(hidden1)
        hidden3 = Dense(512, activation='relu')(hidden2)
        hidden4 = Dense(512, activation='relu')(hidden3)
        dropMLP = Dropout(0.5)(hidden4)
        output = Dense(1, activation='sigmoid')(dropMLP)
        model = Model(inputs=[visibleBT, visibleHR, visibleNBPD,
                              visibleNBPS, visibleRR, visibleAGE, visibleSEX], outputs=output)

        model.summary()
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    else:
        model = load_model(pathM)
        # x改list[]
    train_history = model.fit(x=[BT_train_normalize, HR_train_normalize, NBPD_train_normalize, NBPS_train_normalize, RR_train_normalize, AGE_train_normalize, SEX_train_normalize],
                              y=y_label_onehot,
                              validation_split=0.4,
                              epochs=10,
                              batch_size=32, verbose=2)
    model.save(pathM)
    df.to_csv(pathT, index=False)
    tr = tr + runed

    prediction = model.predict([BT_test_normalize, HR_test_normalize, NBPD_test_normalize,
                                NBPS_test_normalize, RR_test_normalize, AGE_test_normalize, SEX_test_normalize])
    y_pred = (prediction > 0.5)
    predict_classes = []
    for temk in range(len(y_pred)):
        if y_pred[temk][0]:
            predict_classes.append(1)
        else:
            predict_classes.append(0)
    predict_classes_hot = np.array(predict_classes)
    # predict_classes = np.argmax(y_pred, axis=-1)
    print(pd.crosstab(y_test_onehot, predict_classes_hot,
                      rownames=['label'], colnames=['predict']))
    m = tf.keras.metrics.Recall()
    m.update_state(y_test_onehot, predict_classes_hot)
    print(m.result().numpy())

    # print(confusion_matrix(y_test, predict_classes))
    # print(pd.crosstab(y_test,predict_classes,rownames=['label'],colnames=['predict']))
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')
# scores = model.evaluate(
#     [BT_test_normalize, HR_test_normalize, NBPD_test_normalize, NBPS_test_normalize, RR_test_normalize, AGE_test_normalize, SEX_test_normalize], y_test_onehot, verbose=1)
# print(scores[1])
