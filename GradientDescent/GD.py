#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt


# In[3]:


#df = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/3_gradient_descent/Exercise/test_scores.csv")
#df.to_csv("GD_testscores.csv", index=False)
df = pd.read_csv("/Users/hamza/DataSc_RM/MachineLearning/GradientDescent/GD_testscores.csv")
df


# In[4]:


x = np.array(df["math"])
y=np.array(df["cs"])
print(x,y)


# In[12]:


n = len(x)
def gradient_descent(x,y):
    m_curr= 0
    b_curr= 0
    iterations = 1000
    learning_rate = 0.05
    for i in range(iterations):
        y_pred = m_curr * x+b_curr
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, iteration {}".format(m_curr,b_curr,i))

gradient_descent(x,y)        


# In[ ]:




