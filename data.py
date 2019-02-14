import numpy as np
from nltk.corpus import stopwords
from re import sub
from scipy import stats


def load_glove(path):
    wordvecs = {}
    glove_voc=[]
    with open(path,encoding='utf-8',mode='r') as file:
        for line in file:
            tokens = line.split(' ')
            vec = np.array(tokens[1:], dtype=np.float32)
            wordvecs[tokens[0]] = vec

            # random
            # vec=np.random.normal(0.0, 0.1, 300)
            # wordvecs[tokens[0]] = vec

            glove_voc.append(tokens[0])
    return wordvecs,glove_voc

def fill_with_gloves(word_to_id, wordvecs):
    emb_size=len(wordvecs['the'])
    n_words = len(word_to_id)
    res = np.zeros([n_words, emb_size], dtype=np.float32)
    n_not_found = 0
    for word, id in word_to_id.items():
        if id==0:
            res[id, :] = np.zeros(emb_size, dtype=np.float32)
            continue;
        if word in wordvecs:
            res[id, :] = wordvecs[word]
        else:
            n_not_found += 1
            res[id, :] = np.random.normal(0.0, 0.1, emb_size)
       #     print(word)
        #    res[id, :] = np.zeros(emb_size,dtype=np.float32)
    print('n words not found in glove word vectors: ' + str(n_not_found))
    return res

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = sub(r"what's", "what is ", text)
    text = sub(r"\'s", " ", text)
    text = sub(r"\'ve", " have ", text)
    text = sub(r"can't", "cannot ", text)
    text = sub(r"n't", " not ", text)
    text = sub(r"i'm", "i am ", text)
    text = sub(r"\'re", " are ", text)
    text = sub(r"\'d", " would ", text)
    text = sub(r"\'ll", " will ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\/", " ", text)
    text = sub(r"\^", " ^ ", text)
    text = sub(r"\+", " + ", text)
    text = sub(r"\-", " - ", text)
    text = sub(r"\=", " = ", text)
    text = sub(r"'", " ", text)
    text = sub(r"(\d+)(k)", r"\g<1>000", text)
    text = sub(r":", " : ", text)
    text = sub(r" e g ", " eg ", text)
    text = sub(r" b g ", " bg ", text)
    text = sub(r" u s ", " american ", text)
    text = sub(r"\0s", "0", text)
    text = sub(r" 9 11 ", "911", text)
    text = sub(r"e - mail", "email", text)
    text = sub(r"j k", "jk", text)
    text = sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


def load_data(file_path):
    data=[]
    with open(file_path,'r',encoding='utf-8') as f:
        f.readline()  #去除其实列标记
        for line in f:
            row=line.split('\t')
            data.append((row[1].strip(),row[2].strip(),float(row[3])))
    return data

def get_vocab(train_data):
    #  stops = set(stopwords.words('english'))
    vocab=set('PAD')
    word_to_id = {'PAD': 0}
    id_to_word = {0: 'PAD'}
    word_to_count = {}
    vocab_size = 1
    for data in train_data:
        for word in text_to_word_list(data[0]+' '+data[1]):
            # Remove unwanted words
            # if word in stops:
            #     continue
            if word not in vocab:
                vocab.add(word)
                word_to_id[word] = vocab_size
                word_to_count[word] = 1
                id_to_word[vocab_size] = word
                vocab_size += 1
            else:
                word_to_count[word] += 1
    return vocab,vocab_size,word_to_id,id_to_word,word_to_count

def convert_to_numeric(train_data,word_to_id):
    train_ndata=[]
    for i in range(len(train_data)):
        sent1=[]
        sent2=[]
        for word in text_to_word_list(train_data[i][0]):
            sent1.append(word_to_id[word])
        for word in text_to_word_list(train_data[i][1]):
            sent2.append(word_to_id[word])
        train_ndata.append((sent1,sent2,train_data[i][2]))
    return train_ndata

def get_train_batch(len,batch_size):
    return zip(range(0, len - batch_size, batch_size), range(batch_size, len, batch_size))

def get_process_train(train_data,max_sent_size):
    x_sent1=[]
    x_sent2=[]
    y_target=[]
    x_sent1_len=[]
    x_sent2_len=[]
    for data in train_data:
        x_sent1.append(data[0]+[0]*(max_sent_size-len(data[0])))
        x_sent2.append(data[1]+[0] * (max_sent_size - len(data[1])) )
        y_target.append((data[2]))
        x_sent1_len.append(len(data[0]))
        x_sent2_len.append(len(data[1]))
    return x_sent1,x_sent2,y_target,x_sent1_len,x_sent2_len

def eval_score(train_pred,train_y_target):
    spearman,_=stats.spearmanr(train_y_target,train_pred)
    pearson,_=stats.pearsonr(train_pred,train_y_target)
    return pearson,spearman
