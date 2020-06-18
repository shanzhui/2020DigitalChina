#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

warnings.filterwarnings("ignore")


# In[2]:


features=[]
f = []
def read_information(path,know_type=True , columns = ['ship' , 'x' , 'y' , 'v' , 'd' , 'time','type']):
    global f
    f = []
    df=pd.read_csv(path)
    
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    time = df['time'].agg(lambda x:np.max(x) - np.min(x))
    df.columns = columns
    features.append(time.seconds)
    f.append('diff_seconds')
    features.append(df['ship'][0])
    f.append('ship')
    
    df['d'] = df['d'] / 180.0 * np.pi
    df['v_sin'] = df['v'] * np.sin(df['d'])
    df['v_cos'] = df['v'] * np.cos(df['d'])
    df['dis'] = np.sqrt( df['x']**2 + df['y']**2 )
    
    xmin = df['x'].min()
    ymin = df['y'].min()
    df['x'] = (df['x'] - xmin)/df['x'].median()
    df['y'] = (df['y'] - ymin)/df['y'].median()
    
    for pos in [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,1]:
        for fea in ['dis' ,'x','y','v','d']:
            features.append(df[fea].quantile(pos))
            f.append(f'{fea}_{pos}')
    
    
    
    df['time']=pd.to_datetime(df['time'],format='%m%d %H:%M:%S')
    t_diff=df['time'].diff().iloc[1:].dt.total_seconds()
    x_diff=df['x'].diff().iloc[1:].abs()
    y_diff=df['y'].diff().iloc[1:].abs()
    dis=sum(np.sqrt(x_diff**2+y_diff**2))
    x_a_mean=(x_diff/t_diff).mean()
    y_a_mean=(y_diff/t_diff).mean()
    features.append(np.sqrt(x_a_mean**2+y_a_mean**2))#a
    f.append('a')
 
    if(know_type):
        if(df["type"].iloc[0]=='拖网'):
            features.append(2)
        if(df["type"].iloc[0]=='刺网'):
            features.append(1)
        if(df["type"].iloc[0]=='围网'):
            features.append(0)
        f.append('type')


# In[3]:

def get_train(system = 'linux'):
    path_train="/tcdata/hy_round2_train_20200225/"
    if system == 'windows':
        path_train="../tcdata/hy_round2_train_20200225/"
    print('trainB')    
    print(os.listdir('/'))
    ls = os.listdir(path_train)

    for i in tqdm(ls):
        if '.csv' in i:
            read_information(f'{path_train}{i}',know_type=True)
    
    '''path_train="/tcdata/hy_round1_train_20200102/"
    if system == 'windows':
        path_train="../tcdata/hy_round1_train_20200102/"
    print('trainA')    
    sls = os.listdir(path_train)

    for i in tqdm(sls):
        if '.csv' in i:
            read_information(f'{path_train}{i}',know_type=True)'''
    
    train_data=pd.DataFrame(np.array(features).reshape(len(ls),int(len(features)/( len(ls) ) )))
    train_data.columns=f
    return train_data

# In[4]:

def get_test(system = 'linux'):
    print('testA')
    path_test = "/tcdata/hy_round2_testA_20200225/"
    if system == 'windows':
        path_test = "../tcdata/hy_round2_testA_20200225/"
    ls = os.listdir(path_test)
    for i in tqdm(ls):
        if '.csv' in i:
            read_information(f'{path_test}{i}',know_type=False,columns=['ship' , 'x' , 'y' , 'v' , 'd' , 'time'])
    
    test_data=pd.DataFrame(np.array(features).reshape(len(ls),int(len(features)/len(ls))))
    test_data.columns=f
    return test_data


# In[7]:
def extract(system = 'linux'):
    global features
    features = []
    train_data = get_train(system)
    features = []
    test_data = get_test(system)
    features = []
    if system == 'linux':
        train_data.to_csv('/train_data.csv' , index = None)
        test_data.to_csv('/test_data.csv' , index = None)
    elif system == 'windows' :
        train_data.to_csv('../train_data.csv' , index = None)
        test_data.to_csv('../test_data.csv' , index = None)

# In[ ]:





# In[ ]:




