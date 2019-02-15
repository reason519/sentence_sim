import embedding
import data
import model
import tensorflow as tf
import os
import random
import numpy as np

train_data_file='D:\\nlp\\我的实验\\句子相似度\\sick\\SICK_train.txt'
trial_data_file='D:\\nlp\\我的实验\\句子相似度\\sick\\SICK_trial.txt'
test_data_file='D:\\nlp\\我的实验\\句子相似度\\sick\\SICK_test_annotated.txt'

train_data=data.load_data(train_data_file)
trial_data=data.load_data(trial_data_file)
test_data=data.load_data(test_data_file)

train_data=train_data+trial_data
print('train data size: ',len(train_data))
print('test data size: ',len(test_data))

vocab,vocab_size,word_to_id,id_to_word,word_to_count=data.get_vocab(train_data+test_data)
print('vocab size: ',vocab_size)

word_to_senses_path='D:\\nlp\\我的实验\\多义词\\trained_40w\\word_to_sense.txt'
#word_to_vector_path='D:\\nlp\\我的实验\\多义词\\trained_40w\\word_vectors.txt'
sense_to_vector_path='D:\\nlp\\我的实验\\多义词\\trained_40w\\sense_vectors.txt'

ass_vector=data.get_sense_ass(word_to_senses_path,sense_to_vector_path)
init_emb = data.fill_with_gloves(word_to_id,ass_vector)
#print(init_emb)
print('Embedding Size: %d' % init_emb.shape[1])


train_ndata=data.convert_to_numeric(train_data,word_to_id)
test_ndata=data.convert_to_numeric(test_data,word_to_id)

#model
batch_size=32
max_sent_size = 32

hidden_units=50
#learning_rate=1.0
learning_rate=0.001

epochs=500
log_period = 10

# embd_file='F:\\paper\\pretrain\\GoogleNews-vectors-negative300.bin'
# print('Building Embedding Matrix')
# embedding = embedding.Get_Embedding(embd_file, word_to_id)
# embedding_size = embedding.embedding_matrix.shape[1]
# init_emb=embedding.embedding_matrix
# print('Embedding Size: ',embedding_size)


# embed_file="word_vectors.txt"
# embed_file="D:\\nlp\\我的实验\\多义词\\trained_40w\\word_vectors.txt"
# #load_glove
# wordvecs,glove_voc=data.load_glove(embed_file)
# print('Load glove vectors size:%d  Dimesion:%d' %(len(glove_voc),len(wordvecs['the'])))
# # emb
# init_emb = data.fill_with_gloves(word_to_id,wordvecs)
# print('Embedding Size: %d' % init_emb.shape[1])


print(">> Train start ...")
with tf.variable_scope('model', reuse=None):
    model_train=model.SentSimRNN(True,batch_size,max_sent_size,hidden_units,learning_rate  ,init_emb)
   # embedding.embedding_matrix)
count=0
CKPT_DIR = 'ckpt/'  # 保存训练好的数据
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

for i in range(0,epochs):
    random.shuffle(train_ndata)
    if len(train_ndata)<batch_size:
        print('training data is not enought!')
        exit(0)
    batches = data.get_train_batch(len(train_ndata), batch_size)

    for start,end in batches:
        count+=1
        x_sent1,x_sent2,y_target,x_sent1_len,x_sent2_len=data.get_process_train(train_ndata[start:end],max_sent_size)
        loss,pred=model_train.train(x_sent1,x_sent2,y_target,x_sent1_len,x_sent2_len)

        if count%30==0:
            print("Epoch:", i + 1,  "loss:{0:.4f}.".format(loss))

    if (i + 1) % log_period == 0:
        tf.train.Saver().save(model_train.session, CKPT_DIR + 'wsd.ckpt')
        print("____________eval_______________")

        train_pred=[]
        train_mse=[]
        train_y_target=[]
        batches = data.get_train_batch(len(train_ndata), batch_size)
        for start, end in batches:
            x_sent1, x_sent2, y_target,x_sent1_len,x_sent2_len = data.get_process_train(train_ndata[start:end], max_sent_size)
            train_loss,pred=model_train.train(x_sent1,x_sent2,y_target,x_sent1_len,x_sent2_len,train_cond=False)
            train_mse.append(train_loss)

            train_pred.extend(pred)
            train_y_target=train_y_target+y_target

        train_pearson,train_spearman=data.eval_score(train_pred,train_y_target)
        print("Train Pearson correlation: ",train_pearson," Spearman: ",train_spearman," MSE: ",np.mean(train_mse))

        eval_pred = []
        eval_mse = []
        eval_y_target = []
        batches = data.get_train_batch(len(test_ndata), batch_size)
        for start, end in batches:
            x_sent1, x_sent2, y_target ,x_sent1_len,x_sent2_len= data.get_process_train(test_ndata[start:end], max_sent_size)
            eval_loss, pred = model_train.train(x_sent1, x_sent2, y_target ,x_sent1_len,x_sent2_len, train_cond=False)
            eval_mse.append(eval_loss)
            eval_pred.extend(pred)
            eval_y_target = eval_y_target + y_target

            eval_pearson, eval_spearman = data.eval_score(eval_pred, eval_y_target)
        print("Test Pearson correlation: ", eval_pearson, " Spearman: ", eval_spearman, " MSE: ", np.mean(eval_mse))
