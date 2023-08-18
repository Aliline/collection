# 5.此程式用於[結合]訓練keras模型所需X(圖片向量)與Y(詞遷入向量)
import numpy as np
import os
from alive_progress import alive_bar
flax = 28
yvector = np.load('D:/F111156114_Aliline/imgtoword/wordtovector/測試/靶心偏移_待開發/Step11_npy/vectors_260_ver2_flax{}.npy'.format(flax)) #來源於get_vector
length = 1000
limiter = -1
with alive_bar(length) as bar:
        if not os.path.exists('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_crop/vy.npy'):
            x = []
            y = []
            for num in range(1,1001):
                # imgvectors = np.load('D:/F111156114_Aliline/imgtoword/wordtovector/corp_vector_inceptionV3/crop_class{}_feature.npy'.format(num)) #來源於createCNNvector
                imgvectors = np.load('D:/F111156114_Aliline/imgtoword/wordtovector/corp_vector_inceptionV3/crop_class{}_feature.npy'.format(num)) #來源於createCNNvector
                limit = 0
                # imgvectors = np.load('./wordtoimage/vector_inceptionV3/class{}_feature.npy'.format(num))
                muti = imgvectors.shape[0]
                for mu in range(muti):
                    # x.append(imgvectors[mu])
                    y.append(yvector[num-1])
                    limit = limit +1
                    if limit == limiter :
                         break
                bar.text("Sub_Progessing...{}%".format(num/10))
                bar()
            # np.save('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_renew/vx260_full_corp',x) #_corp代表使用已標註框資料
            
            # np.save('./wordtoimage/vector_MLP/vy',y)
            np.save('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_renew/v260y_full_corp_flax{}'.format(flax),y)
            
            

        # print(imgvectors.shape)
# print(yvector.shape)