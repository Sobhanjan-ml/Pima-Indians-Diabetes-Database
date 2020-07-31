#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset = pd.read_csv('Pima Indians Diabetes Dataset.csv')


# In[3]:


dataset


# In[4]:


x = dataset.iloc[:,0:8].values


# In[5]:


x


# In[6]:


y = dataset.iloc[:,8]


# In[8]:


y


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


sc = StandardScaler()


# In[13]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[14]:


x_train


# In[15]:


x_test


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


logr = LogisticRegression()


# In[18]:


logr.fit(x_train,y_train)


# In[19]:


y_pred = logr.predict(x_test)


# In[20]:


y_pred


# In[21]:


from sklearn.metrics import accuracy_score


# In[23]:


accuracy_score(y_pred,y_test)*100


# In[ ]:




