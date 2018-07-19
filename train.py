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
       word_num_features=['word_wmd', 'word_norm_wmd', 'word_cosine_distance', 'word_cityblock_distance', 'word_jaccard_distance', 'word_canberra_distance', 'word_euclidean_distance', 'word_minkowski_distance', 'word_braycurtis_distance', 'word_skew_qvec', 'word_skew_avec', 'word_kur_qvec', 'word_kur_avec', 'word_unigram_tfidf_sim', 'word_bigram_tfidf_sim', 'word_node_max_clique', 'word_node_pr', 'word_edge_max_clique_q1', 'word_edge_max_clique_q2', 'word_edge_pr_q1', 'word_edge_pr_q2', 'word_edge_pr_max', 'word_edge_pr_min', 'word_edge_pr_max_ratio_pr_min', 'word_edge_pr_max_dis_pr_min', 'word_edge_pr_max_sum_pr_min', 'word_edge_max_clique_max', 'word_edge_max_clique_min', 'word_edge_max_clique_max_ratio_pr_min', 'word_edge_max_clique_max_dis_pr_min', 'word_edge_max_clique_max_sum_pr_min', 'word_q1_q2_intersect', 'word_q1_freq', 'word_q2_freq', 'word_neigh_0', 'word_neigh_1', 'word_neigh_2', 'word_neigh_3', 'word_neigh_4', 'word_neigh_5', 'word_neigh_6', 'word_neigh_7', 'word_neigh_8', 'word_neigh_9', 'word_neigh_10', 'word_neigh_11', 'word_neigh_12', 'word_neigh_13', 'word_neigh_14', 'word_neigh_15', 'word_neigh_16', 'word_neigh_17', 'word_neigh_18', 'word_neigh_19', 'word_neigh_20', 'word_neigh_21', 'word_neigh_22', 'word_neigh_23', 'word_neigh_24', 'word_neigh_25', 'word_neigh_26', 'word_neigh_27', 'word_neigh_28', 'word_neigh_29', 'word_neigh_30', 'word_neigh_31', 'word_neigh_32', 'word_neigh_33', 'word_neigh_34', 'word_neigh_35', 'word_neigh_36', 'word_neigh_37', 'word_neigh_38', 'word_neigh_39', 'word_neigh_40', 'word_neigh_41', 'word_neigh_42', 'word_neigh_43', 'word_neigh_44',],
       char_num_features=['char_wmd', 'char_norm_wmd', 'char_cosine_distance', 'char_cityblock_distance', 'char_jaccard_distance', 'char_canberra_distance', 'char_euclidean_distance', 'char_minkowski_distance', 'char_braycurtis_distance', 'char_skew_qvec', 'char_skew_avec', 'char_kur_qvec', 'char_kur_avec', 'char_unigram_tfidf_sim', 'char_bigram_tfidf_sim', 'char_node_max_clique', 'char_node_pr', 'char_edge_max_clique_q1', 'char_edge_max_clique_q2', 'char_edge_pr_q1', 'char_edge_pr_q2', 'char_edge_pr_max', 'char_edge_pr_min', 'char_edge_pr_max_ratio_pr_min', 'char_edge_pr_max_dis_pr_min', 'char_edge_pr_max_sum_pr_min', 'char_edge_max_clique_max', 'char_edge_max_clique_min', 'char_edge_max_clique_max_ratio_pr_min', 'char_edge_max_clique_max_dis_pr_min', 'char_edge_max_clique_max_sum_pr_min', 'char_q1_q2_intersect', 'char_q1_freq', 'char_q2_freq', 'char_neigh_0', 'char_neigh_1', 'char_neigh_2', 'char_neigh_3', 'char_neigh_4', 'char_neigh_5', 'char_neigh_6', 'char_neigh_7', 'char_neigh_8', 'char_neigh_9', 'char_neigh_10', 'char_neigh_11', 'char_neigh_12', 'char_neigh_13', 'char_neigh_14', 'char_neigh_15', 'char_neigh_16', 'char_neigh_17', 'char_neigh_18', 'char_neigh_19', 'char_neigh_20', 'char_neigh_21', 'char_neigh_22', 'char_neigh_23', 'char_neigh_24', 'char_neigh_25', 'char_neigh_26', 'char_neigh_27', 'char_neigh_28', 'char_neigh_29', 'char_neigh_30', 'char_neigh_31', 'char_neigh_32', 'char_neigh_33', 'char_neigh_34', 'char_neigh_35', 'char_neigh_36', 'char_neigh_37', 'char_neigh_38', 'char_neigh_39', 'char_neigh_40', 'char_neigh_41', 'char_neigh_42', 'char_neigh_43', 'char_neigh_44',],
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
