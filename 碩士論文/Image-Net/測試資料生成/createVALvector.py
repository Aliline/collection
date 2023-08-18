#此為生成測試資料 沒意外的話會自動按照label_information進行排序
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
#這裡可以自訂詞遷入向量大小 必須要有對應檔案 來源用 training 與 get_vector生成
vector_size = 290
testtxt = "D:/F111156114_Aliline/imgtoword/wordtovector/vectors_{}_ver2.txt".format(vector_size)
classtxt = "D:/F111156114_Aliline/imgtoword/wordtovector/LOC_replaced.txt"
label = []
classes = []
class_name = []
with open(testtxt, 'r', encoding='utf-8') as labels:
    for temp in labels:
        if "：" in temp:
            te = temp.split(".")
            te = te[1].split("：")
            label.append(te[0])
with open(classtxt, 'r', encoding='utf-8') as classfile:
        for temp in classfile:
                te = temp.split(" ")
                classes.append(te[0])
                rete = te[1].replace("\n","")
                class_name.append(rete)

def getYclass(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    count = 1
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        clen = len(classes)
        for k in range(clen):
            if cls == classes[k] :
                cname = class_name[k]
                cname = cname.lower().strip()
                cname = cname.replace("-","_")
                for r in label :
                    if r == cname :
                         return count
                    count = count +1   

    return -1


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
model = InceptionV3(weights='imagenet', include_top=False , pooling = "avg")
# model = VGG16(weights='imagenet', include_top=False) 
# temp = model.output
# model_flat = Flatten()(temp)
# sentences = MyCorpus()
# model = gensim.models.Word2Vec(sentences=sentences)
# image_ids = open('D:/F111156114_Aliline/imgtoword/wordtovector/LOC_synset_mapping.txt').read().strip().splitlines()
image_ids = open('D:/F111156114_Aliline/imgtoword/wordtovector/組合包/LOC_label_information.txt').read().strip().splitlines()
num = 1
test = 0
temp_set = os.listdir('C:/ImageNet/ILSVRC/Data/CLS-LOC/val/')
# runeds = os.listdir("C:/Users/USER/pywork2/images/vgg_answer/")
length = len(temp_set)
with alive_bar(length) as bar:
        image_set = os.listdir('C:/ImageNet/ILSVRC/Data/CLS-LOC/val/')
        images = []
        ilabel = []
        if not os.path.exists('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_renew/{}.npy'.format(num)):
            test = 0
            for image_file in image_set:
                if ".JPEG" in image_file:
                    # vec_word= model.wv[name]
                    img_path = 'C:/ImageNet/ILSVRC/Data/CLS-LOC/val/{}'.format(image_file)
                    xml_path = img_path.replace(".JPEG",".xml").replace("Data","Annotations")
                    class_num = getYclass(xml_path)
                    if class_num == -1 :
                         class_num = class_num
                    img = load_img(img_path, target_size=(299, 299))  
                    x = img_to_array(img)  
                    x = np.expand_dims(x, axis=0)  
                    x = preprocess_input(x) 
                    
                    # extract features  
                    # features = model.predict(x,verbose=0)
                    # print(features.shape)
                    # images.append(features)
                    ilabel.append(class_num)
                    # kest = str(features)
                    # kest = decode_predictions(features, top=1)[0]               
                    # print(features)
                    bar.text("Sub_Progessing...{}%".format((test/len(image_set))*100))

                    # print(image_file)
                    # if test == 3 :
                    #     break
                    # test = test +1
                test = test +1
                bar()


            np_images = np.array(images)
            # np.save('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_renew/valx', np_images)
            np.save('D:/F111156114_Aliline/imgtoword/wordtovector/vector_MLP_renew/val_{}_y_2'.format(vector_size), ilabel)
        num = num +1
            

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