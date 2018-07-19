import pickle as pkl
import numpy as np
import random
import math
import json
import pandas as pd
random.seed(2018)
class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self,mode,hparams,batch_size,file_name):
        self.mode=mode
        self.hparams=hparams
        self.batch_size=batch_size
        df=pd.read_csv(file_name)
        index=list(range(len(df)))
        if self.mode=='train':
            index=index[int(len(df)*hparams.idx*1.0/hparams.all_process):]+index[:int(len(df)*hparams.idx*1.0/hparams.all_process)]
        self.data=df.iloc[index][['words_1','chars_1','words_2','chars_2','label']].values
        self.word_num=df.iloc[index][hparams.word_num_features].values
        self.char_num=df.iloc[index][hparams.char_num_features].values

            
            
        self.idx=0
        
    def reset(self):
        self.idx=0
        
    def next(self):
        if self.idx>=len(self.data):
            self.reset()
            raise StopIteration
        words1=[]
        chars1=[]
        words_EM1=[]
        chars_EM1=[]
        words1_len=[]
        chars1_len=[]
        words2=[]
        chars2=[]
        words_EM2=[]
        chars_EM2=[]
        words2_len=[]
        chars2_len=[]
        label=[]
        
        word_num=[]
        char_num=[]
            
        while self.idx<len(self.data) and len(label)!=self.batch_size:
            temp=self.data[self.idx]
         
            words1.append(temp[0].split())
            chars1.append(temp[1].split())
            words1_len.append(len(words1[-1]))
            chars1_len.append(len(chars1[-1]))
        
            words2.append(temp[2].split())
            chars2.append(temp[3].split())
            words2_len.append(len(words2[-1]))
            chars2_len.append(len(chars2[-1]))        
            label.append(temp[4])
            word_num.append(self.word_num[self.idx])
            char_num.append(self.char_num[self.idx])

            self.idx+=1
            
        max_word1_len=max(words1_len)
        max_word2_len=max(words2_len)
        max_char1_len=max(chars1_len)
        max_char2_len=max(chars2_len)
        for i in range(len(label)):
            words1[i]=['BOS']+words1[i]+['EOS']+['<PAD>']*(max_word1_len-words1_len[i])
            words2[i]=['BOS']+words2[i]+['EOS']+['<PAD>']*(max_word2_len-words2_len[i])
            chars1[i]=['BOS']+chars1[i]+['EOS']+['<PAD>']*(max_char1_len-chars1_len[i])
            chars2[i]=['BOS']+chars2[i]+['EOS']+['<PAD>']*(max_char2_len-chars2_len[i])
            

            

            
            
        return  (words1,chars1,words1_len,chars1_len,),(words2,chars2,words2_len,chars2_len),label,word_num,char_num   
        
