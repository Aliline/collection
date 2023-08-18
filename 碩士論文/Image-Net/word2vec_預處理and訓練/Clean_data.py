# 1.此為專門生成訓練Word2Vec用資料 來源預設使用WikiEN 未來預計使用SRT或google文章
import re
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
# from gensim.models.word2vec import LineSentence
# input_file = 'phrases_test.txt'
# output_file = 'phrases_test_process.txt'
input_file = 'D:/F111156114_Aliline/imgtoword/wordtovector/wiki-preprocessed-raw.txt'
output_file = 'D:/F111156114_Aliline/imgtoword/wordtovector/wiki-preprocessed-raw_cleaned.txt'
output_label = 'D:/F111156114_Aliline/imgtoword/wordtovector/wiki-preprocessed-switched_ver_t.txt'
refreence_file_name = 'D:/F111156114_Aliline/imgtoword/wordtovector/LOC_refreence.txt'
lemmatizer = WordNetLemmatizer()
# 資料清理


def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s_]', '', sentence)
    return re.sub(r'\s{2,}', ' ', sentence)

# 取得詞性


def get_wordnet_pos(tag):
    if tag[1][0] == 'J':
        return wordnet.ADJ
    elif tag[1][0] == 'V':
        return wordnet.VERB
    elif tag[1][0] == 'N':
        return wordnet.NOUN
    elif tag[1][0] == 'R':
        return wordnet.ADV
    else:
        return None

# 取得每行資料


def lemmatizeword(word):
    word = lemmatizer.lemmatize(word, wordnet.ADJ)
    word = lemmatizer.lemmatize(word, wordnet.VERB)
    word = lemmatizer.lemmatize(word, wordnet.NOUN)
    word = lemmatizer.lemmatize(word, wordnet.ADV)
    return word


def get_sentences(input_file_pointer):
    while True:
        line = input_file_pointer.readline()
        if not line:
            break

        yield line

# 去除停用詞


def tokenize(sentence):
    return [token for token in sentence.split() if token not in STOP_WORDS]

# 訓練Phrases模型


# def build_phrases(sentences):
#     phrases = Phrases(sentences,
#                       min_count=5,
#                       threshold=7,
#                       connector_words=ENGLISH_CONNECTOR_WORDS)
#     phrases.save('phrases_model_mc5_t7.model')

# 使用Phrases模型連結短語


def sentence_to_bi_grams(phrases_model, sentence):
    return ''.join(phrases_model[sentence])

# 資料前處理(清理、停用詞、詞性還原)


def sentences_to_bi_grams(phrases_model, input_file_name, output_file_name):
    time = 0
    with open(input_file_name, 'r', encoding='utf-8') as input_file_pointer:
        with open(output_file_name, 'w+', encoding='utf-8') as out_file:
            for sentence in get_sentences(input_file_pointer):
                cleaned_sentence = clean_sentence(sentence)
                tokenized_sentence = tokenize(cleaned_sentence)
                new_word = ''
                # token = nltk.pos_tag(tokenize(sentence))
                count = 0
                for str in tokenized_sentence:
                    # tag = get_wordnet_pos(token[count]) or wordnet.NOUN
                    new_word += f'{lemmatizeword(str)} '
                    count += 1
                # parsed_sentence = sentence_to_bi_grams(
                #     phrases_model, new_word)
                # out_file.write(parsed_sentence.strip() + '\n')
                print(f'執行行數：{time}')
                time += 1
                out_file.write(new_word + '\n')
#主函式!
def switch_label(input_file_name,output_file_name,refreence_file_name):
     with open(input_file_name, 'r', encoding='utf-8') as input_file_pointer:
        with open(refreence_file_name, 'r', encoding='utf-8') as refreence_file_pointer:
            with open(output_file_name, 'w+', encoding='utf-8') as out_file:
                sec_output = ""
                time = 0
                for sentence in get_sentences(input_file_pointer):
                    output = sentence
                    output = output.replace("\n","")
                    
                    for refrences in refreence_file_pointer :
                        ref = refrences.replace("\n","")
                        ref = ref.split(",")
                        labal = ref[0].split(" ")[0]
                        #格式統一
                        first_feture = ref[0].replace("{} ".format(labal),"")
                        first_feture = first_feture.lower().strip()
                        first_feture = first_feture.replace("-","_")
                        #替換同義詞
                        for num in range(1,len(ref)):
                            rep_stentence = ref[num]
                            output = output.replace(rep_stentence,first_feture)
                            output = output.replace(rep_stentence.lower().strip(),first_feture)
                        #重複詞彙(之前為了避免文章不含label單詞,故重複單詞避免未輸出)
                        sec_output = sec_output + "{} {} {} {} {} {} {} {} ".format(first_feture,first_feture,first_feture,first_feture,first_feture,first_feture,first_feture,first_feture)
                        time = time +1
                        if time % 100 == 0:
                            sec_output = sec_output + "\n"
                    out_file.write("{}\n".format(output))         
                # out_file.write("{}\n".format(sec_output))
                

if __name__ == '__main__':
    # sentences = LineSentence(input_file)
    # build_phrases(sentences)
    #phrases_model = Phraser.load('phrases_model_mc5_t7.model')
    # sentences_to_bi_grams("phrases_model", input_file, output_file)
    switch_label(output_file,output_label,refreence_file_name)
