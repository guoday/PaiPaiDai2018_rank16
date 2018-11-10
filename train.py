import numpy as np
import pandas as pd
import tensorflow as tf
import model as model_all
import model_word
import model_char
import os
import utils
import sys
gpu_idx=sys.argv[1]
idx=sys.argv[2]
all_process=sys.argv[3]
sub=sys.argv[4]
model=sys.argv[5]
os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_idx

def create_hparams():
    return tf.contrib.training.HParams(
       hidden_size=300,
       batch_size=128,
       encoder_type='bi',
       op='adam',
       forget_bias=1.0,
       num_layer=4,
        dim=100,
       pay_attention=4,
       decay_cont=3,
       num_units=300,
       dropout=0.2,
       maskdropout=0.0,
        batch_num=10,
       epoch=10,
       learning_rate=0.001,
       num_display_steps=100,
       num_eval_steps=1500, 
       layer_sizes=[300,300],
       all_process=int(all_process),
       batch_norm_decay=0.995,
       idx=int(idx),
       model_name=model+str(idx),
       vocab_threshold=10,
       pretrain=False,
       expend=False,
       k=16,
       model=model,
       sub_name=sub,
       word_single_features=None,
       char_single_features=None, 
       word_num_features=None,
       char_num_features=None,
       word_embedding='data/word_embed.txt',
       char_embedding='data/char_embed.txt',
        )

def build_vocabulary(train_df,hparams):
    print("build vocabulary.....")
    word2index={}
    for s in hparams.word_single_features+hparams.char_single_features:
        groupby_size=train_df.groupby(s).size()
        vals=dict(groupby_size[groupby_size>=hparams.vocab_threshold])
        word2index[s]={}
        for v in vals:
            word2index[s][v]=len(word2index[s])+2

    print("done!")
    return word2index

train_df=pd.read_csv('pre_data/train.csv')
hparams=create_hparams()
utils.print_hparams(hparams)
if hparams.word_num_features is None:
    hparams.word_num_features=[]
if hparams.word_single_features is None:
    hparams.word_single_features=[]
if hparams.char_num_features is None:
    hparams.char_num_features=[]
if hparams.char_single_features is None:
    hparams.char_single_features=[]
hparams.word2index=build_vocabulary(train_df,hparams)



if hparams.model=='model_word':
    print('model_word')
    preds=model_word.train(hparams)
elif hparams.model=='model_char':
    print('model_char')
    preds=model_char.train(hparams)
else:
    preds=model_all.train(hparams)
