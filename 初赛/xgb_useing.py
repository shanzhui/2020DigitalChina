#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")


# In[2]:

def run_model(system = 'linux'):
    if system == 'linux':
        train_data = pd.read_csv('/train_data.csv')
        test_data = pd.read_csv('/test_data.csv')
    elif system == 'windows':
        train_data = pd.read_csv('../train_data.csv')
        test_data = pd.read_csv('../test_data.csv')
    

# In[3]:


    train_data = train_data.sort_values('ship').reset_index()
    test_data = test_data.sort_values('ship').reset_index()


# In[4]:


    features = list(set(train_data.columns) - set(['ship','type','from']))


# In[5]:


#划分特征与目标
    kind=train_data.type
    train_data.drop(['type'],axis=1,inplace=True)
    kind.value_counts(1)


# In[6]:

    prob_xgb = np.zeros((len(train_data), 3))
    pred_prob_xgb = np.zeros((len(test_data) , 3))
    pred_xgb = np.zeros((len(train_data),))
    xgb_scores = []

    sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=2020)
    for train,test in sk.split(train_data[features],kind):
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
    
        xlf.fit(x_train[features],y_train,sample_weight=kind.map({0:3,1:5.5,2:2.5}))
    
        val_prob_xgb = xlf.predict_proba(x_test[features])
        prob_xgb[test] = val_prob_xgb
        val_pred_xgb = np.argmax(val_prob_xgb, axis=1)
    
        xgb_scores.append(f1_score(y_test, val_pred_xgb, average='macro'))
        print('xgb score :' , xgb_scores[-1])
    
        pred_prob_xgb += xlf.predict_proba(test_data[features])/5

# In[7]:


    pred_xgb = np.argmax(prob_xgb , axis=1)
    print('xgb :' , f1_score(pred_xgb, kind, average='macro') )
    print('average : ' , np.mean(xgb_scores) )


# In[8]:


    res = pd.DataFrame()
    res['ship'] = test_data['ship']
    res['pred'] = pd.Series(np.argmax(pred_prob_xgb,axis=1))
    res['pred'] = res['pred'].map({0:'围网',1:'刺网',2:'拖网'})#{0:'拖网',1:'围网',2:'刺网'}
    res.to_csv('result_xgb_2.csv',header = None , index = None)


# In[ ]:





# In[9]:


    xlf=xgb.XGBClassifier(max_depth=6
                      ,colsample_bynode=0.1
                      ,learning_rate=0.1
                      ,n_estimators=600
                      ,reg_alpha=0.004
                      ,n_jobs=-1
                      ,reg_lambda=0.002
                      ,importance_type='total_cover'
                      ,objective='multi:softmax')
    
    xlf.fit(train_data[features],kind,sample_weight=kind.map({0:3,1:5.5,2:2.5}))
    test_pred_xgb = xlf.predict_proba(test_data[features])


# In[10]:


    res = pd.DataFrame()
    res['ship'] = test_data['ship'].astype('int')
    res['pred'] = pd.Series(np.argmax(test_pred_xgb,axis=1))
    res['pred'] = res['pred'].map({0:'围网',1:'刺网',2:'拖网'})#{0:'拖网',1:'围网',2:'刺网'}
    if system == 'linux':
        res.to_csv('/result.csv',header = None , index = None)
    elif system == 'windows':
        res.to_csv('../result.csv',header = None , index = None)
    print(os.listdir('/'))

# In[ ]:




