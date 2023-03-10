#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd 
from pandas_profiling import ProfileReport
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import accuracy_score , confusion_matrix , roc_auc_score , roc_curve
import  matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


ingest_data = pd.read_csv("https://raw.githubusercontent.com/Chandrashekhar0106/ml_credit_card_default/main/UCI_Credit_Card.csv")
pd.pandas.set_option('display.max_columns',None)


# In[30]:


ingest_data.head(100)


# In[12]:


x= ingest_data.drop(columns ='default.payment.next.month')


# In[13]:


y= ingest_data['default.payment.next.month']


# In[33]:


pf = ProfileReport(ingest_data)
pf.to_widgets()


# In[34]:


ingest_data.describe()


# In[59]:


x_train , x_test, y_train,y_test = train_test_split(x,y,test_size = .25,random_state = 20)


# In[60]:


x_train


# In[61]:


dt_model = DecisionTreeClassifier()


# In[62]:


dt_model.fit(x_train, y_train)


# In[63]:


dt_model.score(x_test,y_test)


# In[66]:


plt.figure(figsize=(20,20))
tree.plot_tree(dt_model,filled=True)


# In[146]:


df1= ingest_data.head(3000)


# In[147]:


x1=df1.drop(columns= 'default.payment.next.month')


# In[148]:


x1


# In[149]:


y1=df1['default.payment.next.month']


# In[150]:


y1


# In[151]:


[str(i) for i in set(y1)]


# In[152]:


dt_model1= DecisionTreeClassifier()


# In[156]:


dt_model1.predict(x1)


# In[154]:



dt_model1.score(x_test,y_test)


# In[157]:


dt_model.score(x1,y1)


# In[159]:


track=dt_model1.cost_complexity_pruning_path(x1,y1)


# In[161]:


ccp_alpha= track.ccp_alphas


# In[162]:


ccp_alpha


# In[163]:


dt_modle2 = []
for ccp in ccp_alpha:
    dt_m  = DecisionTreeClassifier(ccp_alpha=ccp)
    dt_m.fit(x1,y1)
    dt_modle2.append(dt_m)


# In[166]:


dt_model2 = []
score = []
for i in ccp_alpha:
    dt_m = DecisionTreeClassifier(ccp_alpha = i)
    dt_m.fit(x1,y1)
    dt_model2.append(dt_m)
    score.append(dt_m.score(x_test, y_test))


# In[165]:


plt.plot(score, 'b-+')
plt.xlabel('iteration')
plt.ylabel('score')


# In[167]:


train_score = [i.score(x1,y1) for i in dt_modle2]


# In[168]:


test_score = [i.score(x_test ,y_test) for i in dt_modle2]


# In[169]:


fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alpha, train_score, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alpha, test_score, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[172]:


dt_model_ccp = DecisionTreeClassifier(random_state=0 , ccp_alpha=.030)
dt_model_ccp.fit(x1,y1)


# In[173]:


plt.figure(figsize=(20,20))
tree.plot_tree(dt_model_ccp,filled=True)


# In[174]:


dt_model_ccp.score(x_test,y_test)


# In[175]:


dt_model_ccp.score(x1,y1)


# In[ ]:




