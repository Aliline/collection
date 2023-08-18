# 2.此為用於訓練Word2Vec模型 來源使用 Clean_data.py的生成資料
import logging
from gensim.models import Word2Vec
import nltk
from gensim.models.word2vec import LineSentence
from gensim.models.phrases import Phrases, Phraser
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
input_text_path = 'D:/F111156114_Aliline/imgtoword/wordtovector/wiki-preprocessed-switched_ver_t.txt'
output_model_path = 'D:/F111156114_Aliline/imgtoword/wordtovector/genshin_model/refreencewiki_youtube_10epochs_v290_mc3_phrases_version1'
# with open(input_text_path, 'r', encoding='utf-8') as f:
# for line in iter(lambda: f.read(1024), ''):
#     sentence = nltk.sent_tokenize(line)
#     if sentences == '':
#         sentences = sentence
#     else:
#         for str in sentence:
#             sentences.append(str)
sentences = LineSentence(input_text_path)  # 將剛剛寫的檔案轉換成 iterable
# phrases = Phrases(sentences)
# bigram = Phraser(phrases)
# new_sentences = bigram[sentences]

#vector_size用於調整詞遷入向量輸出維度
model = Word2Vec(sentences, vector_size=290, window=10, min_count=1,
                 epochs=10, workers=8)  # 可以自行實驗 size = 100, size = 300，再依照你的case來做調整。
# 將 Model 存到 wiki-lemma-100D，他還會一併儲存兩個trainables.syn1neg.npy結尾和wv.vectors.npy結尾的文件
model.save(output_model_path)
model.wv.save_word2vec_format(
    'D:/F111156114_Aliline/imgtoword/wordtovector/genshin_model/refreencewiki_youtube_10epochs_v290_mc3_phrases_version1.bin', binary=True)
