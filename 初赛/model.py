#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import datetime

warnings.filterwarnings("ignore")


# In[2]:

def run_model():
    #读取数据
    train_data = pd.read_csv('../data/train_data.csv')
    test_data = pd.read_csv('../data/test_data.csv')


# In[3]:

    #数据按照船的编号排序
    train_data = train_data.sort_values('ship').reset_index()
    test_data = test_data.sort_values('ship').reset_index()
    del train_data['index']
    del test_data['index']


# In[4]:


    #确定特征集合
    features = list(set(train_data.columns) - set(['ship','type','from']))


# In[5]:


    #划分特征与目标
    kind=train_data.type
    train_data.drop(['type'],axis=1,inplace=True)
    kind.value_counts(1)


# In[6]:



    #第一次训练，预测testA
    print('第一次训练，预测testA')
    prob_xgb = np.zeros((len(train_data), 3))
    pred_prob_xgb = np.zeros((len(test_data) , 3))
    pred_xgb = np.zeros((len(train_data),))
    xgb_scores = []

    sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=2020)
    for train,test in sk.split(train_data,kind):

        x_train=train_data.iloc[train]
        y_train=kind.iloc[train]
        x_test=train_data.iloc[test]
        y_test=kind.iloc[test]
    
        xlf=xgb.XGBClassifier(max_depth=6
                      ,colsample_bynode=0.1
                      ,learning_rate=0.1
                      ,n_estimators=500
                      ,reg_alpha=0.004
                      ,n_jobs=-1
                      ,reg_lambda=0.002
                      ,importance_type='total_cover'
                      ,objective='multi:softmax')
    
        xlf.fit(x_train[features],y_train)
    
        val_prob_xgb = xlf.predict_proba(x_test[features])
        prob_xgb[test] = val_prob_xgb
        val_pred_xgb = np.argmax(val_prob_xgb, axis=1)
    
        xgb_scores.append(f1_score(y_test, val_pred_xgb, average='macro'))
        print('xgb score :' , xgb_scores[-1])
    
        pred_prob_xgb += xlf.predict_proba(pseudo_label[features])/5


# In[7]:


    pred_xgb = np.argmax(prob_xgb , axis=1)
    print('xgb :' , f1_score(pred_xgb, kind, average='macro') )
    print('average : ' , np.mean(xgb_scores) )


# In[8]:


    #伪标签
    pseudo_kind = pd.Series(np.argmax(pred_prob_xgb,axis=1))
    expand_train_index = np.max(pred_prob_xgb , axis=1) > 0.9


# In[9]:



    #加入伪标签，预测testB
    print('加入伪标签，预测testB')
    prob_xgb = np.zeros((len(train_data), 3))
    pred_prob_xgb = np.zeros((len(test_data) , 3))
    pred_xgb = np.zeros((len(train_data),))
    xgb_scores = []

    sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=2020)
    for train,test in sk.split(train_data,kind):

        x_train=pd.concat([train_data[features].iloc[train],pseudo_label[expand_train_index][features]])
        y_train=pd.concat([kind.iloc[train],pseudo_kind[expand_train_index]])
        x_test=train_data.iloc[test]
        y_test=kind.iloc[test]
    
        xlf=xgb.XGBClassifier(max_depth=7
                      ,colsample_bynode=0.1
                      ,learning_rate=0.1
                      ,n_estimators=600
                      ,reg_alpha=0.004
                      ,n_jobs=-1
                      ,reg_lambda=0.002
                      ,importance_type='total_cover'
                      ,objective='multi:softmax')
    
        xlf.fit(x_train[features],y_train)
    
        val_prob_xgb = xlf.predict_proba(x_test[features])
        prob_xgb[test] = val_prob_xgb
        val_pred_xgb = np.argmax(val_prob_xgb, axis=1)
    
        xgb_scores.append(f1_score(y_test, val_pred_xgb, average='macro'))
        print('xgb score :' , xgb_scores[-1])
    
        pred_prob_xgb += xlf.predict_proba(test_data[features])/5


# In[10]:


    pred_xgb = np.argmax(prob_xgb , axis=1)
    print('xgb :' , f1_score(pred_xgb, kind, average='macro') )
    print('average : ' , np.mean(xgb_scores) )


# In[11]:


    #保存预测结果
    pred = pd.DataFrame(pred_xgb,columns = ['whole']).copy()
    result = pd.DataFrame( np.argmax(pred_prob_xgb,axis=1) ,columns = ['whole'])


# In[12]:


    #对围网预测进行修正
    print('对围网预测进行修正')
    kind_0 = kind.copy()
    kind_0 = kind_0.map({0:0,1:1,2:1})


# In[13]:



    #第一次训练，预测testA
    print('第一次训练，预测testA')
    prob_xgb = np.zeros((len(train_data), 2))
    pred_prob_xgb = np.zeros((len(test_data) , 2))
    pred_xgb = np.zeros((len(train_data),))
    xgb_scores = []

    sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=2020)
    for train,test in sk.split(train_data,kind_0):

        x_train=train_data.iloc[train]
        y_train=kind_0.iloc[train]
        x_test=train_data.iloc[test]
        y_test=kind_0.iloc[test]
    
        xlf=xgb.XGBClassifier(max_depth=6
                      ,colsample_bynode=0.1
                      ,learning_rate=0.1
                      ,n_estimators=500
                      ,reg_alpha=0.004
                      ,n_jobs=-1
                      ,reg_lambda=0.002
                      ,importance_type='total_cover'
                         )
    
        xlf.fit(x_train[features],y_train)
    
        val_prob_xgb = xlf.predict_proba(x_test[features])
        prob_xgb[test] = val_prob_xgb
        val_pred_xgb = np.argmax(val_prob_xgb, axis=1)
    
        xgb_scores.append(f1_score(y_test, val_pred_xgb))
        print('xgb score :' , xgb_scores[-1])
    
        pred_prob_xgb += xlf.predict_proba(pseudo_label[features])/5


# In[14]:


    pred_xgb = np.argmax(prob_xgb , axis=1)
    print('xgb :' , f1_score(pred_xgb, kind_0, average='macro') )
    print('average : ' , np.mean(xgb_scores) )


# In[15]:


    #伪标签
    pseudo_kind_0 = pd.Series(np.argmax(pred_prob_xgb,axis=1))
    expand_train_index_0 = np.max(pred_prob_xgb , axis=1) > 0.9


# In[16]:



    #加入伪标签，预测testB
    print('加入伪标签，预测testB')
    prob_xgb = np.zeros((len(train_data), 2))
    pred_prob_xgb = np.zeros((len(test_data) , 2))
    pred_xgb = np.zeros((len(train_data),))
    xgb_scores = []

    sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=2020)
    for train,test in sk.split(train_data,kind_0):

        x_train=pd.concat([train_data[features].iloc[train],pseudo_label[expand_train_index_0][features]])
        y_train=pd.concat([kind_0.iloc[train],pseudo_kind_0[expand_train_index_0]])
        x_test=train_data.iloc[test]
        y_test=kind_0.iloc[test]
    
        xlf=xgb.XGBClassifier(max_depth=7
                      ,colsample_bynode=0.1
                      ,learning_rate=0.1
                      ,n_estimators=600
                      ,reg_alpha=0.004
                      ,n_jobs=-1
                      ,reg_lambda=0.002
                      ,importance_type='total_cover'
                         )
    
        xlf.fit(x_train[features],y_train)

        val_prob_xgb = xlf.predict_proba(x_test[features])
        prob_xgb[test] = val_prob_xgb
        val_pred_xgb = np.argmax(val_prob_xgb, axis=1)
    
        xgb_scores.append(f1_score(y_test, val_pred_xgb))
        print('xgb score :' , xgb_scores[-1])
    
        pred_prob_xgb += xlf.predict_proba(test_data[features])/5


# In[17]:


    pred_xgb = np.argmax(prob_xgb , axis=1)
    print('xgb :' , f1_score(pred_xgb, kind_0, average='macro') )
    print('average : ' , np.mean(xgb_scores) )


# In[ ]:





# In[18]:


    #保存预测结果，修正之前预测的结果
    result['second'] = pd.Series(np.argmax(pred_prob_xgb,axis=1))
    result['amend_0'] = result['whole'].copy()
    result['amend_0'][ (np.max(pred_prob_xgb , axis=1) > 0.99) & (result['second'] == 0) ] = 0
    result['amend_0'].value_counts(1)


# In[19]:


    pred_xgb = np.argmax(prob_xgb,axis=1)
    pred['second'] = pd.Series(pred_xgb).copy().astype('int')
    pred['amend_0'] = pred['whole'].copy()
    pred['amend_0'][ (np.max(prob_xgb,axis=1) > 0.99) & (pred['second'] == 0) ] = 0


# In[20]:


    print( f1_score(pred['amend_0'] , kind , average='macro') )


    # In[21]:


    #生成结果文件
    print('生成结果文件')
    result['ship'] = test_data['ship'].astype('int')
    result['res'] = result['amend_0'].map({0:'围网',1:'刺网',2:'拖网'})
    result[['ship','res']].to_csv('../submit/result_amend_'+
                              datetime.datetime.now().strftime(format = '%Y%m%d_%H%M%S')+'.csv',
                              header = None , index = None)


# In[ ]:





# In[ ]:




