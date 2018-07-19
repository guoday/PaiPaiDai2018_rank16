import tensorflow as tf
import data_iterator
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
import time
import numpy as np
import pickle
import utils    
from sklearn.metrics import log_loss
import os
import pandas as pd
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import math
class Model(object):
    def __init__(self,hparams,mode):
        self.mode=mode
        self.hparams=hparams
        params = tf.trainable_variables()
        #define placeholder
        self.vocab_table_word=lookup_ops.index_table_from_file('pre_data/vocab_word.txt', default_value=0) 
        self.vocab_table_char=lookup_ops.index_table_from_file('pre_data/vocab_char.txt', default_value=0)
        self.norm_trainable=tf.placeholder(tf.bool)
        self.q1={}
        self.q2={}
        self.label=tf.placeholder(shape=(None,),dtype=tf.float32)
        
        for q in [self.q1,self.q2]:
            q['words']=tf.placeholder(shape=(None,None), dtype=tf.string)
            q['words_len']=tf.placeholder(shape=(None,), dtype=tf.int32)
            q['chars']=tf.placeholder(shape=(None,None), dtype=tf.string)
            q['chars_len']=tf.placeholder(shape=(None,), dtype=tf.int32)
            q['words_num']=tf.placeholder(shape=(None,len(hparams.word_num_features)), dtype=tf.float32)
            q['chars_num']=tf.placeholder(shape=(None,len(hparams.char_num_features)), dtype=tf.float32)




        
        #build graph
        self.build_graph(hparams) 
        
        #build optimizer
        self.optimizer(hparams)
        params = tf.trainable_variables()
        self.saver = tf.train.Saver(tf.global_variables())
        elmo_param=[]
        for param in tf.global_variables():
            if 'elmo' in param.name and 'elmo/Variable' not in param.name:
                    elmo_param.append(param)
        self.pretrain_saver = tf.train.Saver(elmo_param)         
        utils.print_out("# Trainable variables")
        for param in params:
            if hparams.pretrain is False and 'elmo' in param.name:
                continue
            else:
                utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),param.op.device))
                    

    def build_graph(self, hparams):
        with tf.variable_scope('elmo') as scope:
            self.build_embedding_layer(hparams,trainable=True,scope_name='embedding')
            self.build_bilstm(hparams,scope_name='bilstm')
        if hparams.pretrain:
            self.cost=self.build_elmo_logits(hparams)
            return
            
        self.build_encoder(hparams,scope_name='encoder')
        self.build_interaction(hparams,scope_name='interaction')
        self.build_decoder(hparams,scope_name='decoder')
        logits=self.build_mlp(hparams)
        self.cost=self.compute_loss(hparams,logits)
        

    def build_embedding_layer(self,hparams,trainable,scope_name):
        #create embedding layer
        word_vocab={}
        char_vocab={}
        with open('pre_data/vocab_word.txt','r') as f:
            for line in f:
                word=line.strip()
                word_vocab[word]=len(word_vocab)

        word_embedding=np.random.randn(len(word_vocab), 300)*0.1
        hparams.word_vocab_size=len(word_vocab)
        if hparams.word_embedding:
            with open(hparams.word_embedding, 'r') as f:
                    for line in f:
                        temp=line.split()
                        word=temp[0]
                        vector=temp[1:]
                        if word in word_vocab:
                            word_embedding[word_vocab[word],:]=vector  
        self.word_embedding=tf.Variable(word_embedding,trainable=trainable,dtype=tf.float32) 
        
        with open('pre_data/vocab_char.txt','r') as f:
            for line in f:
                char=line.strip()
                char_vocab[char]=len(char_vocab)

        char_embedding=np.random.randn(len(char_vocab), 300)*0.1
        hparams.char_vocab_size=len(char_vocab)
        if hparams.char_embedding:
            with open(hparams.char_embedding, 'r') as f:
                    for line in f:
                        temp=line.split()
                        char=temp[0]
                        vector=temp[1:]
                        if char in char_vocab:
                            char_embedding[char_vocab[char],:]=vector
                            
        self.char_embedding=tf.Variable(char_embedding,trainable=trainable,dtype=tf.float32)
        
        for q in [self.q1,self.q2]:
            words_id=self.vocab_table_word.lookup(q['words']) 
            q['words_id']=words_id
            if hparams.maskdropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                mask=tf.ones(tf.shape(words_id))
                mask=tf.cast(tf.minimum(tf.nn.dropout(mask,1-hparams.maskdropout),1),tf.int64)
                words_id=tf.cast(words_id*mask,tf.int32) 
            q['words_inp'] = tf.gather(self.word_embedding, words_id[:,1:-1])
            
        for q in [self.q1,self.q2]:
            chars_id=self.vocab_table_char.lookup(q['chars'])
            q['chars_id']=chars_id
            if hparams.maskdropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                mask=tf.ones(tf.shape(chars_id))
                mask=tf.cast(tf.minimum(tf.nn.dropout(mask,1-hparams.maskdropout),1),tf.int64)
                chars_id=tf.cast(chars_id*mask,tf.int32) 
            q['chars_inp'] = tf.gather(self.char_embedding, chars_id[:,1:-1])
            
    def build_bilstm(self,hparams,scope_name):
        with tf.variable_scope(scope_name+'_words') as scope:  
            fw_cell,bw_cell= self._build_encoder_cell(hparams,num_layer=4,num_units=300,encoder_type='bi',dropout=0.5 if hparams.pretrain else 0.0)
            W = layers_core.Dense(512,activation=tf.nn.relu, use_bias=False, name="W")
            for q in [self.q1,self.q2]:
                words_inp = q['words_inp']
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,words_inp,dtype=tf.float32, sequence_length=q['words_len'],time_major=False,swap_memory=True) 
                q['word_elmo_lstm']=bi_outputs
                q['word_elmo_output']=[W(x) for x in bi_outputs]
                q['word_elmo_label']=[q['words_id'][:,2:],q['words_id'][:,:-2]]

        with tf.variable_scope(scope_name+'_chars') as scope:  
            fw_cell,bw_cell= self._build_encoder_cell(hparams,num_layer=4,num_units=300,encoder_type='bi',dropout=0.5 if hparams.pretrain else 0.0)
            W = layers_core.Dense(512,activation=tf.nn.relu, use_bias=False, name="W")
            for q in [self.q1,self.q2]:
                chars_inp = q['chars_inp']
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,chars_inp,dtype=tf.float32, sequence_length=q['chars_len'],time_major=False,swap_memory=True) 
                q['char_elmo_lstm']=bi_outputs
                q['char_elmo_output']=[W(x) for x in bi_outputs]
                q['char_elmo_label']=[q['chars_id'][:,2:],q['chars_id'][:,:-2]]
    def build_elmo_logits(self,hparams):
        costs=[]
        with tf.variable_scope("softmax_words") as scope:
            nce_weights= tf.Variable(\
                    tf.truncated_normal([hparams.word_vocab_size,512],stddev=1.0/math.sqrt(512)))
            nce_biases=tf.Variable(tf.zeros([hparams.word_vocab_size]))
            for q in [self.q1,self.q2]:
                for i in range(2):         
                    mask = tf.sequence_mask(q['words_len'], tf.shape(q['word_elmo_output'][i])[-2], dtype=tf.float32)
                    mask=tf.reshape(mask,[-1])
                    inputs=tf.reshape(q['word_elmo_output'][i],[-1,512])
                    labels=tf.reshape(q['word_elmo_label'][i],[-1,1])
                    cost=tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=labels,inputs=inputs,num_sampled=32,num_classes=hparams.word_vocab_size)
                    cost=tf.reduce_sum(cost*mask)/tf.reduce_sum(mask)
                    costs.append(cost)



        
        loss=tf.reduce_mean(costs)
        return loss                   
                
    def build_encoder(self,hparams,scope_name):
        with tf.variable_scope(scope_name+'_words') as scope:
            #encoding words
            fw_cell,bw_cell= self._build_encoder_cell(hparams)
            for q in [self.q1,self.q2]:    
                inputs=tf.concat(q['word_elmo_output']+[q['words_inp']],-1)
                words_inp = tf.transpose(inputs,[1,0,2])
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,words_inp,dtype=tf.float32, sequence_length=q['words_len'],time_major=True,swap_memory=True)
                bi_outputs=tf.concat(bi_outputs,-1)
                
                q['word_encoder_output']=tf.transpose(tf.concat(bi_outputs,-1),[1,0,2])
                q['word_encoder_hidden']=bi_state


        return 

    def build_decoder(self,hparams,scope_name):
        with tf.variable_scope(scope_name+'_words') as scope: 
            fw_cell,bw_cell= self._build_encoder_cell(hparams)
            for q in [self.q1,self.q2]:   
                decoder_inp=tf.transpose(q['word_interaction'],[1,0,2])
                bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,decoder_inp,dtype=tf.float32, sequence_length=q['words_len'],time_major=True,swap_memory=True)    
                bi_outputs=tf.concat(bi_outputs,-1)
                #bi_outputs=self.HighwayNetwork(bi_outputs)
                q['word_decoder_output']=tf.transpose(bi_outputs,[1,0,2])


                



    def build_interaction(self,hparams,scope_name): 
        with tf.variable_scope(scope_name+'_words') as scope:
            for q in [(self.q1,self.q2),(self.q2,self.q1)]:
                encoder_hidden=q[0]['word_encoder_output']
                weight=tf.reduce_sum(encoder_hidden[:,:,None,:]*q[1]['word_encoder_output'][:,None,:,:],-1)
                mask = tf.sequence_mask(q[1]['words_len'], tf.shape(weight)[-1], dtype=tf.float32)
                weight=tf.nn.softmax(weight)*mask[:,None,:]
                weight=weight/(tf.reduce_sum(weight,-1)[:,:,None]+0.000001)
                word_inter=tf.reduce_sum(q[1]['word_encoder_output'][:,None,:,:]*weight[:,:,:,None],-2)
                q[0]['word_interaction']=tf.concat([encoder_hidden,word_inter,tf.abs(encoder_hidden-word_inter),encoder_hidden*word_inter],-1)

                      
        return     
    
  
    def  build_mlp(self,hparams):
        hidden_word=[]
        with tf.variable_scope("MLP_words") as scope:
            attention_W = layers_core.Dense(hparams.hidden_size,activation=tf.nn.relu, use_bias=False, name="attention_W")
            attention_V = layers_core.Dense(1,use_bias=False, name="attention_V")
            for q in [self.q1,self.q2]:
                weight=tf.nn.softmax(tf.reduce_sum(attention_V(attention_W(q['word_decoder_output'])),-1))
                mask = tf.sequence_mask(q['words_len'], tf.shape(weight)[-1], dtype=tf.float32)
                weight=weight*mask
                weight=weight/(tf.reduce_sum(weight,-1)[:,None]+0.000001)
                context_hidden=tf.reduce_sum(q['word_decoder_output']*weight[:,:,None],1)                
                q['word_rep']=context_hidden
        hidden_word=[self.q1['word_rep'],self.q2['word_rep'],self.q1['word_rep']*self.q2['word_rep']]


        hidden_word.append(self.q1['words_num'])
        
        
 
            
        with tf.variable_scope("MLP_words") as scope:
            layer_W = layers_core.Dense(hparams.hidden_size,activation=tf.nn.tanh, use_bias=False, name="ff_layer")
            hidden_word=tf.concat(hidden_word,-1)
            logits=layer_W(hidden_word)           
            if hparams.dropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                logits = tf.nn.dropout(logits,1-hparams.dropout)          
            layer_W = layers_core.Dense(1, use_bias=False, name="ff_layer_output")
            logits_word=layer_W(logits)[:,0]

        logits=logits_word
        return logits
            
                
    def compute_loss(self,hparams,logits):
        self.prob=tf.nn.sigmoid(logits)
        loss=-tf.reduce_mean(self.label*tf.log(self.prob+0.0001)+(1-self.label)*tf.log(1-self.prob+0.0001),-1)
        return loss
    
    def HighwayNetwork(self,inputs, num_layers=2, function='relu',
                        keep_prob=0.8, scope='HN'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if function == 'relu':
                function = tf.nn.relu
            elif function == 'tanh':
                function = tf.nn.tanh
            else:
                raise NotImplementedError
            hidden_size = inputs.get_shape().as_list()[-1]
            memory = inputs
            for layer in range(num_layers):
                with tf.variable_scope('layer_%d' % (layer)):
                    H = layers_core.Dense(hidden_size,activation=function, use_bias=True, name="h")
                    T = layers_core.Dense(hidden_size,activation=function, use_bias=True, name="t")
                    h = H(memory)
                    t = T(memory)
                    memory = h * t + (1-t) * memory
            if keep_prob > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                outputs = tf.nn.dropout(memory,keep_prob)
            else:
                outputs = memory
            return outputs
        
    def _build_encoder_cell(self,hparams,num_layer=None,num_units=None,encoder_type=None,dropout=None,forget_bias=None):
        num_layer=num_layer or hparams.num_layer
        num_units=num_units or hparams.num_units
        encoder_type=encoder_type or hparams.encoder_type
        dropout=dropout or hparams.dropout
        forget_bias=forget_bias or hparams.forget_bias
        if encoder_type=="uni":
            cell_list = []
            for i in range(num_layer):
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units,forget_bias=hparams.forget_bias)
                # Dropout (= 1 - keep_prob)
                if dropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
                        
                cell_list.append(single_cell)
            if len(cell_list) == 1:  # Single layer.
                return cell_list[0]
            else:  # Multi layers
                return tf.contrib.rnn.MultiRNNCell(cell_list)  
        else:
            num_bi_layers = int(num_layer / 2) 
            fw_cell_list=[]
            bw_cell_list=[]
            for i in range(num_bi_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units,forget_bias=forget_bias)
                if dropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                    single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
                        
                fw_cell_list.append(single_cell)
                single_cell = tf.contrib.rnn.BasicLSTMCell(num_units,forget_bias=forget_bias)
                if dropout > 0.0 and self.mode==tf.contrib.learn.ModeKeys.TRAIN:
                    single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
                        
                bw_cell_list.append(single_cell)

            if num_bi_layers == 1:  # Single layer.
                fw_cell=fw_cell_list[0]
                bw_cell=bw_cell_list[0]
            else:  # Multi layers
                fw_cell=tf.contrib.rnn.MultiRNNCell(fw_cell_list)
                bw_cell=tf.contrib.rnn.MultiRNNCell(bw_cell_list)
            return fw_cell,bw_cell
    def dey_lrate(self,sess,lrate):
        sess.run(tf.assign(self.lrate,lrate))
        
    def optimizer(self,hparams):
        self.lrate=tf.Variable(hparams.learning_rate,trainable=False)
        if hparams.op=='sgd':
            opt = tf.train.GradientDescentOptimizer(self.lrate)
        elif hparams.op=='adam':
            opt = tf.train.AdamOptimizer(self.lrate,beta1=0.9, beta2=0.999,epsilon=1e-8)
        params = tf.trainable_variables()

       
        gradients = tf.gradients(self.cost,params,colocate_gradients_with_ops=True)
        clipped_grads, gradient_norm = tf.clip_by_global_norm(gradients, 5.0)  
        self.grad_norm =gradient_norm 
        self.update = opt.apply_gradients(zip(clipped_grads, params))
        
    def batch_norm_layer(self, x, train_phase, scope_bn):
        z = tf.cond(train_phase, lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=True, reuse=None, trainable=True, scope=scope_bn), lambda: batch_norm(x, decay=self.hparams.batch_norm_decay, center=True, scale=True, updates_collections=None,is_training=False, reuse=True, trainable=True, scope=scope_bn))
        return z
    
    def train(self,sess,iterator):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        q1,q2,label,words_num,chars_num=iterator.next()
        dic={}
        dic[self.q1['words']]=q1[0]
        dic[self.q1['words_len']]=q1[2]
        dic[self.q1['words_num']]=words_num
        dic[self.q2['words']]=q2[0]
        dic[self.q2['words_len']]=q2[2]
        dic[self.label]=label
        dic[self.norm_trainable]=True

            
            
        return sess.run([self.cost,self.update,self.grad_norm],feed_dict=dic)
    
    def pretrain_infer(self,sess,iterator):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        q1,q2,label,words_num,chars_num=iterator.next()
        dic={}
        dic[self.q1['words']]=q1[0]
        dic[self.q1['words_len']]=q1[2]
        dic[self.q2['words']]=q2[0]
        dic[self.q2['words_len']]=q2[2]
        dic[self.label]=label

            
            
        return sess.run(self.cost,feed_dict=dic)
    
    
    def infer(self,sess,iterator):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        q1,q2,label,words_num,chars_num=iterator.next()
        dic={}
        dic[self.q1['words']]=q1[0]
        dic[self.q1['words_len']]=q1[2]
        dic[self.q1['words_num']]=words_num
        dic[self.q2['words']]=q2[0]
        dic[self.q2['words_len']]=q2[2]
        dic[self.norm_trainable]=False
        dic[self.label]=label
        

            
        prob1=sess.run(self.prob,feed_dict=dic)
        dic[self.q2['words']]=q1[0]
        dic[self.q2['words_len']]=q1[2]
        dic[self.q1['words']]=q2[0]
        dic[self.q1['words_len']]=q2[2]
        
        
        dic[self.label]=label
        prob2=sess.run(self.prob,feed_dict=dic)
        return (prob1+prob2)/2.0
 
 
def train(hparams):

    
    if hparams.pretrain:
        hparams.learning_rate=0.001
    config_proto = tf.ConfigProto(log_device_placement=0,allow_soft_placement=0)
    config_proto.gpu_options.allow_growth = True
    train_graph = tf.Graph()
    infer_graph = tf.Graph()    
    
    with train_graph.as_default():
        train_model=Model(hparams,tf.contrib.learn.ModeKeys.TRAIN)
        train_sess=tf.Session(graph=train_graph,config=config_proto)
        train_sess.run(tf.global_variables_initializer())
        train_sess.run(tf.tables_initializer())
        
    with infer_graph.as_default():    
        infer_model=Model(hparams,tf.contrib.learn.ModeKeys.INFER)
        infer_sess=tf.Session(graph=infer_graph,config=config_proto)
        infer_sess.run(tf.global_variables_initializer())
        infer_sess.run(tf.tables_initializer())

    train_model.pretrain_saver.restore(train_sess,'pretrain_model/best_model')
    decay=0  
    pay_attention=0
    global_step=0 
    train_loss=0
    train_norm=0
    best_score=1000
    epoch=0
    flag=False
    if hparams.pretrain:
        train_iterator=data_iterator.TextIterator('train',hparams,32,'pre_data/train.csv')
        dev_iterator=data_iterator.TextIterator('dev',hparams,512,'pre_data/dev.csv')
        test_iterator=data_iterator.TextIterator('test',hparams,512,'pre_data/test.csv')
        while True:
            start_time = time.time()
            try:
                cost,_,norm=train_model.train(train_sess,train_iterator)
                global_step+=1
                train_loss+=cost
                train_norm+=norm
            except StopIteration:
                continue
            if global_step%hparams.num_display_steps==0:
                info={}
                info['learning_rate']=hparams.learning_rate
                info["avg_step_time"]=(time.time()-start_time)/hparams.num_display_steps
                start_time = time.time()
                info["train_ppl"]= train_loss / hparams.num_display_steps
                info["avg_grad_norm"]=train_norm/hparams.num_display_steps
                train_loss=0
                train_norm=0        
                utils.print_step_info("  ", global_step, info)   
            if global_step%hparams.num_eval_steps==0:
                train_model.saver.save(train_sess,'pretrain_model/model')
                with infer_graph.as_default():
                    infer_model.saver.restore(infer_sess,'pretrain_model/model')
                    loss=[]
                    while True:
                        try:
                            cost=infer_model.pretrain_infer(infer_sess,dev_iterator) 
                            loss.append(cost)
                        except StopIteration:
                            break              
                    logloss=round(np.mean(loss),5)
                if logloss<best_score:
                    best_score=logloss
                    pay_attention=0
                    print('logloss',logloss)
                    print('best logloss',best_score)   
                    print('saving best model')
                    train_model.pretrain_saver.save(train_sess,'pretrain_model/best_model')
                else:
                    pay_attention+=1
                    print('logloss',logloss)
                    print('best logloss',best_score)   
                    if pay_attention==hparams.pay_attention:
                        exit()
                        
             
    train_iterator=data_iterator.TextIterator('train',hparams,hparams.batch_size,'pre_data/expend.csv' if hparams.expend else 'pre_data/train.csv')
    dev_iterator=data_iterator.TextIterator('dev',hparams,hparams.batch_size,'pre_data/dev.csv')
    test_iterator=data_iterator.TextIterator('test',hparams,hparams.batch_size,'pre_data/test.csv')                    
    dev_df=pd.read_csv('pre_data/dev.csv')
    test_df=pd.read_csv('pre_data/test.csv')
    while epoch < hparams.epoch:
        start_time = time.time()
        try:
            cost,_,norm=train_model.train(train_sess,train_iterator)
            global_step+=1
            train_loss+=cost
            train_norm+=norm
        except StopIteration:
            epoch+=1
            flag=True
        if global_step%hparams.num_display_steps==0:
            info={}
            info['learning_rate']=hparams.learning_rate
            info["avg_step_time"]=(time.time()-start_time)/hparams.num_display_steps
            start_time = time.time()
            info["train_ppl"]= train_loss / hparams.num_display_steps
            info["avg_grad_norm"]=train_norm/hparams.num_display_steps
            train_loss=0
            train_norm=0        
            utils.print_step_info("  ", global_step, info)
        if flag or global_step%hparams.num_eval_steps==0:
            print(epoch)
            flag=False
            saver = train_model.saver
            saver.save(train_sess,'model_'+hparams.model_name+'/model')
            with infer_graph.as_default():
                infer_model.saver.restore(infer_sess,'model_'+hparams.model_name+'/model')
                dev_iterator.reset()
                probs=[]
                while True:
                    try:
                        prob=infer_model.infer(infer_sess,dev_iterator) 
                        probs+=list(prob)
                    except StopIteration:
                        break

                dev_df['y_pred']=probs
                try:
                    logloss = log_loss(dev_df['label'], dev_df['y_pred'], eps=1e-15)
                except:
                    break
                if logloss<best_score:
                    best_score=logloss
                    pay_attention=0
                    print('saving best model')
                    saver.save(train_sess,'model_'+hparams.model_name+'/best_model')
                else:
                    pay_attention+=1
                    if pay_attention==hparams.pay_attention:
                        train_model.saver.restore(train_sess,'model_'+hparams.model_name+'/best_model') 
                        pay_attention=0
                        decay+=1
                        hparams.learning_rate/=2.0
                        train_model.dey_lrate(train_sess,hparams.learning_rate)
                    if decay==hparams.decay_cont:
                        break
                print('logloss',logloss)
                print('best logloss',best_score)
    with infer_graph.as_default():
        infer_model.saver.restore(infer_sess,'model_'+hparams.model_name+'/best_model')    
        dev_iterator.reset()
        probs=[]
        while True:
            try:
                prob=infer_model.infer(infer_sess,dev_iterator) 
                probs+=list(prob)
            except StopIteration:
                break  
        dev_df['y_pre']=probs 
        print(log_loss(dev_df['label'], dev_df['y_pre'], eps=1e-15))
        dev_df[['y_pre']].to_csv('result/dev_'+hparams.sub_name+'.csv',index=False)
        
    
        test_iterator.reset()
        probs=[]
        while True:
            try:
                prob=infer_model.infer(infer_sess,test_iterator) 
                probs+=list(prob)
            except StopIteration:
                break  
        test_df['y_pre']=probs 
        test_df[['y_pre']].to_csv('result/test_'+hparams.sub_name+'.csv',index=False)
