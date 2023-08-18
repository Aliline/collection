# 3.此程式用於得到Word2Vec訓練結果(也就是Y:詞遷入向量)
from gensim.models import keyedvectors
import numpy as np
import pandas as pd
df = pd.read_excel("D:/F111156114_Aliline/imgtoword/wordtovector/label_information.xlsx")  
keyword_path = 'D:/F111156114_Aliline/imgtoword/wordtovector/LOC_replaced.txt'#改成替換過的LOC
keyvectors_path = 'D:/F111156114_Aliline/imgtoword/wordtovector/vectors_290_ver2.txt'#輸出成txt可以打開看
vector_path = 'D:/F111156114_Aliline/imgtoword/wordtovector/vectors_290_ver2'#最後以npy儲存，無法打開看
vector_T_path = 'D:/F111156114_Aliline/imgtoword/wordtovector/vectors_290_ver2_T' #T不用管他 甚至輸出直接註解也行
#載入word2vec模型，由training.py訓練而來的
model = keyedvectors.load_word2vec_format(
    'D:/F111156114_Aliline/imgtoword/wordtovector/genshin_model//refreencewiki_youtube_10epochs_v290_mc3_phrases_version1.bin', binary=True)#放重新訓練好的W2V模型

with open(keyword_path, 'r', encoding='utf-8') as f:
    with open(keyvectors_path, 'w', encoding='utf-8') as of:
        id = ''
        lines = ''

        output_vector = []
        length = 1000
        boolean = False
        id = lines[:9]
        lines = lines.replace('\n', '')
        lines = lines[10:].split(', ')
        for k in range(length) :
                try:
                    count = df["ILSVRC2012_ID"][k]
                    keyword = df["class_name"][k]
                    keyword = keyword.split(",")[0]
                    keyword = keyword.lower().strip()
                    keyword = keyword.replace(' ', '_')
                    keyword = keyword.replace('-', '_')
                    keyword = keyword.lower()
                    # if count == 56:
                    #     print('1')
                    vector = model[keyword]
                    output_vector.append(vector)
                    of.write(f'{count}.{keyword}：{vector}\n')

                    
                except:
                    #如果有少目標單詞會自動印出該類別
                    print(keyword)
                    pass
        output_vector = np.array(output_vector)
        output_vector_T = output_vector.T
        np.save(vector_path,output_vector)      
        np.save(vector_T_path,output_vector_T)       
