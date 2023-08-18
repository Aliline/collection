# 4.此為使用keras自帶的InceptionV3模組將圖片轉換成特徵向量
import tensorflow as tf
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image 
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.utils import load_img ,img_to_array
import numpy as np 
import os
import xml.etree.ElementTree as ET
from alive_progress import alive_bar
# print(np.__version__)
# from gensim.test.utils import datapath
# from gensim import utils
# import gensim.models

# class MyCorpus:
#     """An iterator that yields sentences (lists of str)."""

#     def __iter__(self):
#         corpus_path = datapath('lee_background.cor')
#         for line in open(corpus_path):
#             # assume there's one document per line, tokens separated by whitespace
#             yield utils.simple_preprocess(line)
model = InceptionV3(weights='imagenet', include_top=False , pooling = "avg") #avg代表輸出格式為1*1*2048 
# model = VGG16(weights='imagenet', include_top=False) 
# temp = model.output
# model_flat = Flatten()(temp)
# sentences = MyCorpus()
# model = gensim.models.Word2Vec(sentences=sentences)

# 注意: 這裡輸出時沒有按照label_information所標註的順序輸出 要另外寫判定來按照排序輸出
# (LOC_synset_mapping.txt沒有按照label_information的順序輸出，建議使用pandas將欄位抓取)
# 已完成:將LOC_label_information.txt 替換成:LOC_label_information.txt
# image_ids = open('D:/F111156114_Aliline/imgtoword/wordtovector/LOC_label_information.txt').read().strip().splitlines()
image_ids = open('D:/F111156114_Aliline/imgtoword/wordtovector/組合包/LOC_label_information.txt').read().strip().splitlines()
num = 1
test = 0
# runeds = os.listdir("C:/Users/USER/pywork2/images/vgg_answer/")
length = 1000
with alive_bar(length) as bar:
    for image_id in image_ids:
        image_id = image_id.replace(",","")
        ids = image_id.split(" ")
        id = ids[0]
        name = ids[1]
        image_set = os.listdir('C:/ImageNet/ILSVRC/Data/CLS-LOC/train/%s/'%(id))
        images = []
        if not os.path.exists('D:/F111156114_Aliline/imgtoword/wordtovector/vector_keras_inceptionV3/{}.npy'.format(num)):
            test = 0
            for image_file in image_set:
                if ".JPEG" in image_file:
                    # vec_word= model.wv[name]
                    img = load_img('C:/ImageNet/ILSVRC/Data/CLS-LOC/train/{}/{}'.format(id,image_file), target_size=(299, 299))  
                    x = img_to_array(img)  
                    x = np.expand_dims(x, axis=0)  
                    x = preprocess_input(x) 
                    
                    # extract features  
                    features = model.predict(x,verbose=0)
                    # print(features.shape)
                    images.append(features)
                    # kest = str(features)
                    # kest = decode_predictions(features, top=1)[0]               
                    # print(features)
                    bar.text("Sub_Progessing...{}%".format((test/len(image_set))*100))

                    # print(image_file)
                    # if test == 3 :
                    #     break
                    # test = test +1
                test = test +1


            np_images = np.array(images)
            np.save('D:/F111156114_Aliline/imgtoword/wordtovector/vector_keras_inceptionV3/{}'.format(num), np_images)
        num = num +1
        bar()

# load the pretrained model 

# load and preprocess the image 
# img = load_img('images/person.jpg', target_size=(224, 224))  
# x = img_to_array(img)  
# x = np.expand_dims(x, axis=0)  
# x = preprocess_input(x) 
 
# extract features  
# features = model.predict(x)
# print(features)
# print('Predicted:', decode_predictions(features, top=3)[0]