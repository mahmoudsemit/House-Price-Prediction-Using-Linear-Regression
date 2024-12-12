#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv(r'C:\Users\DELL\Desktop\Housing.csv')


# In[3]:


data.head()


# In[4]:


data.tail(10)


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


Categorical_features= [col for col in data.columns if data[col].dtype =='object']
Categorical_features


# In[10]:


Numerical_features= [col for col in data.columns if data[col].dtype != 'object']
Numerical_features


# In[11]:


plt.figure(figsize=(20,25), facecolor='white')
plotnumber=1

for column in Categorical_features:
    if plotnumber<=7:
        ax= plt.subplot(5,5, plotnumber)
        sns.countplot(x=column, data=data)
        plt.xlabel(column,fontsize=20)
    plotnumber+=1
plt.tight_layout()


# In[12]:


data=pd.get_dummies(data,drop_first=True)


# In[13]:


data.head()


# In[14]:


plt.figure(figsize=(15,7))
correlation_matrix=data.corr(numeric_only=True)
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[15]:


X=data.drop(columns=['price'], axis=1)
y=data.price


# In[16]:


X.head()


# In[17]:


y.head()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=5)


# In[20]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lr=LinearRegression()


# In[23]:


lr.fit(X_train, y_train)


# In[24]:


prediction=lr.predict(X_test)


# In[25]:


lr.score(X_train, y_train)


# In[26]:


lr.score(X_test,y_test)


# In[27]:


plt.scatter(y_test, prediction)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title ('Actual Prices vs. Predicted Prices')
plt.show()


# In[28]:


from sklearn import metrics


# In[29]:


print('MAE:' , metrics.mean_absolute_error(y_test,prediction))
print('MSE:' , metrics.mean_squared_error(y_test,prediction))
print('RMSE:' , np.sqrt(metrics.mean_squared_error(y_test,prediction)))
print('R squared:',metrics.r2_score(y_test,prediction))


# In[31]:


lr.predict([[9654,3,4,2,4,True,True,False,False,False,False,True,False]])


# In[ ]:




