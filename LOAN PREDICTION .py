#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv("loan.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.isnull().sum()


# In[6]:


data['Loan_Status'].value_counts()


# In[7]:


data = data.dropna()


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# In[10]:


data['Loan_Status'].value_counts()


# In[11]:


data.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[12]:


data.head()


# In[13]:


data['Dependents'].value_counts()


# In[14]:


data = data.replace(to_replace='3+', value=4)


# In[15]:


data['Dependents'].value_counts()


# In[16]:


data.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[17]:


data.head()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X = data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = data['Loan_Status']


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


new_model=LogisticRegression()


# In[26]:


new_model.fit(X_train,Y_train)


# In[29]:


from sklearn.metrics import accuracy_score


# In[30]:


#accuracy score on training data
X_train_prediction = new_model.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)


# In[31]:


print('Accuracy on training data : ', training_data_accuray)


# In[33]:


# accuracy score on training data
X_test_prediction = new_model.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)


# In[34]:


print('Accuracy on test data : ', test_data_accuray)


# In[ ]:




