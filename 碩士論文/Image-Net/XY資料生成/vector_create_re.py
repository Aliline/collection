# 5.此程式用於[結合]訓練keras模型所需X(圖片向量)與Y(詞遷入向量)

import numpy as np

file_path='D:/F111156114_Aliline/imgtoword/wordtovector/組合包/XY資料生成'
#Y:將 vector_y*1300(為了符合x的shape[0]，所以[擴張]大約每1300(有些不一定1300)一個類別
vector_y=np.load('D:/F111156114_Aliline/imgtoword/wordtovector/vectors_290_ver2.npy')

vec_x=[]
vec_y=[]
count=1
for i in range(1,1001):
    imgvectors = np.load('D:/F111156114_Aliline/imgtoword/wordtovector/vector_keras_inceptionV3/{}.npy'.format(i)) #來源於createCNNvector
    vec_x.append(imgvectors)
    #看shape
    if imgvectors.shape !=(1300, 1, 2048):
        print(f"No.{count} shape={imgvectors.shape}")
    count+=1
    #隨著x的shape[0]決定擴張程度
    vec_y_re=np.tile(vector_y[i-1], (imgvectors.shape[0], 1))
    vec_y.append(vec_y_re)
       
#X:將 vector_x_1~1000合在一起=>vector_full_x
vector_full_x_re=np.concatenate((vec_x), axis=0)
np.save(f'{file_path}/x', vector_full_x_re)
#合在一起
vector_full_y_re=np.concatenate((vec_y), axis=0)
np.save(f'{file_path}/y', vector_full_y_re)