#!/usr/bin/env python
# coding: utf-8

# # my-ml-projects
# 
# Use the "Run" button to execute the code.

# # Avoiding Kick Buy
# 
# 
# ![istockphoto-465663510-612x612.jpg](attachment:istockphoto-465663510-612x612.jpg)
# One of the biggest challenges of an auto dealership purchasing a used car at an auto auction is the risk of that the vehicle might have serious issues that prevent it from being sold to customers. The auto community calls these unfortunate purchases "kicks".
# 
# Kicked cars often result when there are tampered odometers, mechanical issues the dealer is not able to address, issues with getting the vehicle title from the seller, or some other unforeseen problem. Kick cars can be very costly to dealers after transportation cost, throw-away repair work, and market losses in reselling the vehicle.
# 
# Modelers who can figure out which cars have a higher risk of being kick can provide real value to dealerships trying to provide the best inventory selection possible to their customers.
# We are trying to predict which of the cars in the datasets is a bad buy

# In[ ]:





# In[1]:


get_ipython().system('pip install jovian plotly.express scikit-learn opendatasets matplotlib xgboost lightgbm --upgrade --quiet')


# In[2]:


import jovian


# In[3]:


# Execute this to save new versions of the notebook
jovian.commit(project="my-ml-projects")


# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import opendatasets as od
import csv
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import jovian


# In[5]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (18, 10)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[6]:


od.download('https://www.kaggle.com/competitions/DontGetKicked')


# In[7]:


import os
os.listdir('DontGetKicked/')


# # READING THE DATA

# In[8]:


raw_df = pd.read_csv('DontGetKicked/training.csv')
test_df = pd.read_csv('DontGetKicked/test.csv')
submissions_df = test_df[['RefId']]


# In[9]:


raw_df


# In[10]:


raw_df.info()


# In[11]:


raw_df.isna().sum()


# In[12]:


#dropping all columns with excessive categorical data and unique id 
columns_to_drop = ['RefId', 'BYRNO',  'VehYear','VNZIP1','PurchDate', 'Make', 'Model', 'SubModel','PRIMEUNIT','AUCGUART', 'Trim', 'VNST', 'Color']


# In[13]:


raw_df.drop(columns_to_drop, axis = 'columns', inplace=True)
test_df.drop(columns_to_drop, axis = 'columns', inplace=True)


# In[14]:


plt.figure(figsize=(21, 10))
sns.heatmap(raw_df.corr(), annot=True)


# In[15]:


#dropping correlations with high correlation 
high_corr = ['MMRCurrentRetailCleanPrice','MMRAcquisitonRetailCleanPrice','MMRCurrentAuctionCleanPrice','MMRAcquisitionAuctionCleanPrice']
raw_df.drop(high_corr, axis='columns', inplace= True)


# In[16]:


raw_df


# The columns dropped above are columns that have high correlation (i.e very close to 1), also columns with excessive categorical data(also unique Id). 

# # EXPLORATORY DATA ANALYSIS
# 

# In[17]:


px.histogram(raw_df, x='Auction', color = 'Transmission')


# In[18]:


plt.title('The age of the Vehicle since manufactured year')
sns.barplot(x='IsBadBuy', y='VehicleAge', data=raw_df);
#px.bar(raw_df, x='IsBadBuy', y ='VehicleAge')


# In[19]:


px.histogram(raw_df, x= 'WheelType')


# The above plots, illustrates the following:
# 1. The Auctioneer and the Type of cars they sell, either being Auto, Manual and it seems from the plot that MANHEIM has the highest auctioned cars and Mostly Auto Cars.
# 2. The age of the Vehicle and whether it is a bad buy, from the plot.
# 3. The most Wheel type, from the plot it shows that Alloy is the most.
# 

# In[20]:


raw_df


# # IMPUTATION

# In[21]:


input_col = list(raw_df)[1:]
target = raw_df['IsBadBuy'].copy()


# In[22]:


train_inputs = raw_df[input_col].copy()
test_inputs = test_df[input_col].copy()


# In[23]:


numeric_cols = train_inputs.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()


# In[24]:


jovian.commit()


# In[25]:


raw_df[numeric_cols].isna().sum()


# In[26]:


from sklearn.impute import SimpleImputer


# In[27]:


Imputer = SimpleImputer(strategy ='mean')


# In[28]:


Imputer.fit(raw_df[numeric_cols])


# In[29]:


train_inputs[numeric_cols] = Imputer.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols]= Imputer.transform(test_inputs[numeric_cols])


# In[30]:


train_inputs[numeric_cols].isna().sum()


# # FEATURE SCALING
# We are scaling the whole numerical data from 0 to 1 using the MinMaxScaler.

# In[31]:


from sklearn.preprocessing import MinMaxScaler


# In[32]:


scaler = MinMaxScaler()


# In[33]:


scaler.fit(raw_df[numeric_cols])


# In[34]:


train_inputs[numeric_cols]= scaler.transform(train_inputs[numeric_cols])
test_inputs[numeric_cols]= scaler.transform(test_inputs[numeric_cols])


# In[35]:


train_inputs.describe()


# # ONE HOT ENCODING

# In[36]:


from sklearn.preprocessing import OneHotEncoder


# In[37]:


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# In[38]:


encoder.fit(raw_df[categorical_cols])


# In[39]:


encoded_cols = list(encoder.get_feature_names(categorical_cols))
print(encoded_cols)


# In[40]:


train_inputs[encoded_cols]= encoder.transform(train_inputs[categorical_cols])
test_inputs[encoded_cols]= encoder.transform(test_inputs[categorical_cols])


# In[41]:


train_df=train_inputs[numeric_cols+encoded_cols]
test_df=test_inputs[numeric_cols+encoded_cols]


# In[42]:


jovian.commit()


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[44]:


X_train, X_val, y_train, y_val = train_test_split(train_df, target, test_size=0.15, random_state=0)


# # XGBOOST

# In[45]:


from xgboost import XGBClassifier


# In[46]:


xgb= XGBClassifier()
xgb.fit(X_train, y_train)


# In[47]:


xgb.predict(X_train)


# In[48]:


xgb.score(X_train, y_train)


# In[49]:


xgb.score(X_val,y_val )


# ## Hyperparameters of XGBOOST and tuning 

# ### n_estimators

# In[50]:


def test_params(**params):
    xgb=XGBClassifier(n_jobs=-1, random_state=42,eval_metric='logloss', use_label_encoder=False,**params)
    xgb.fit(X_train, y_train)
    train_score=xgb.score(X_train,y_train)
    val_score = xgb.score(X_val, y_val)
    return train_score, val_score


# In[51]:


test_params(n_estimators= 20)


# In[52]:


def test_params_and_plot(param_name, param_values):
    train_grade, val_grade =[], []
    for value in param_values:
        params={param_name:value}
        train_score, val_score=test_params(**params)
        train_grade.append(train_score)
        val_grade.append(val_score)

    plt.figure(figsize=(10,6))
    plt.title("Overfitting Curve"+ param_name)
    plt.plot(param_values, train_grade, 'b-o')
    plt.plot(param_values, val_grade, 'r-o')
    plt.ylabel('Score')
    plt.xlabel(param_name)
    plt.legend(['Training', 'Validation'])


# In[53]:


test_params_and_plot('n_estimators', [10,20,30,40,50,60,70,80,90,100])


# ### MAX DEPTH

# In[54]:


def test_params(**params):
    xgb=XGBClassifier(n_jobs=-1, random_state=40,eval_metric='logloss', use_label_encoder=False,**params)
    xgb.fit(X_train, y_train)
    train_score=xgb.score(X_train,y_train)
    val_score = xgb.score(X_val, y_val)
    return train_score, val_score


# In[55]:


test_params(max_depth=5)


# In[56]:


def test_params_and_plot(param_name, param_values):
    train_grade, val_grade =[], []
    for value in param_values:
        params={param_name:value}
        train_score, val_score=test_params(**params)
        train_grade.append(train_score)
        val_grade.append(val_score)

    plt.figure(figsize=(10,6))
    plt.title("Overfitting Curve "+ param_name)
    plt.plot(param_values, train_grade, 'b-o')
    plt.plot(param_values, val_grade, 'r-o')
    plt.ylabel('Score')
    plt.xlabel(param_name)
    plt.legend(['Training', 'Validation'])


# In[57]:


test_params_and_plot('max_depth', [1, 5, 10, 20, 25, 30])


# #### input the parameters and predict

# In[58]:


clf=XGBClassifier(n_jobs=-1, random_state=40,eval_metric='logloss', use_label_encoder=False, max_depth=5, n_estimators =20 )


# In[59]:


clf.fit(X_train,y_train)


# In[60]:


clf.predict(X_train)


# In[61]:


clf.score(X_train, y_train)


# In[62]:


clf.score(X_val, y_val)


# # LGBM
# 

# In[63]:


from lightgbm import LGBMClassifier


# In[64]:


lgb=LGBMClassifier()


# In[65]:


lgb.fit(X_train, y_train)


# In[66]:


lgb.predict(X_train)


# In[67]:


lgb.score(X_train, y_train)


# In[68]:


lgb.score(X_val, y_val)


# ## Hyperparameters and Tuning of LGBMs

# ### n_estimators

# In[69]:


def test_params(**params):
    lgb=LGBMClassifier(n_jobs=-1, random_state=42, **params)
    lgb.fit(X_train, y_train)
    train_score= lgb.score(X_train, y_train)
    val_score= lgb.score(X_val, y_val)
    return train_score, val_score


# In[70]:


test_params(n_estimators = 50)


# In[71]:


def test_params_and_plot(param_name, param_values):
    train_grade, val_grade =[], []
    for value in param_values:
        params={param_name:value}
        train_score, val_score=test_params(**params)
        train_grade.append(train_score)
        val_grade.append(val_score)

    plt.figure(figsize=(10,6))
    plt.title("Overfitting Curve "+ param_name)
    plt.plot(param_values, train_grade, 'b-o')
    plt.plot(param_values, val_grade, 'r-o')
    plt.ylabel('Score')
    plt.xlabel(param_name)
    plt.legend(['Training', 'Validation'])


# In[72]:


test_params_and_plot('n_estimators', [10,20,30,40,50,60,70,80,90,100,150])


# ### learning rate

# In[73]:


def test_params(**params):
    lgb=LGBMClassifier(n_jobs=-1, random_state=42, **params)
    lgb.fit(X_train, y_train)
    train_score= lgb.score(X_train, y_train)
    val_score= lgb.score(X_val, y_val)
    return train_score, val_score


# In[74]:


test_params(learning_rate = 0.5)


# In[75]:


def test_params_and_plot(param_name, param_values):
    train_grade, val_grade =[], []
    for value in param_values:
        params={param_name:value}
        train_score, val_score=test_params(**params)
        train_grade.append(train_score)
        val_grade.append(val_score)

    plt.figure(figsize=(10,6))
    plt.title("Overfitting Curve "+ param_name)
    plt.plot(param_values, train_grade, 'b-o')
    plt.plot(param_values, val_grade, 'r-o')
    plt.ylabel('Score')
    plt.xlabel(param_name)
    plt.legend(['Training', 'Validation'])


# In[76]:


test_params_and_plot('learning_rate', [0.1,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90])


# #### input the parameters

# In[77]:


gbf= LGBMClassifier(n_jobs=-1, random_state=42, n_estimators=50, learning_rate=0.5 )


# In[78]:


gbf.fit(X_train, y_train)


# In[79]:


gbf.predict(X_train)


# In[80]:


gbf.score(X_train, y_train)


# In[81]:


gbf.score(X_val, y_val)


# # CONCLUSION
# 
# After conducting exploratory analysis on the dataset 'Dont Get KicKed Cars', we observe that this best Machine Learning algorithm to use is the Lightgbm.

# In[82]:


lgb = lgb.fit(train_df, target)
predictions = lgb.predict(test_df)
submissions_df['IsBadBuy'] = predictions
submissions_df.to_csv('Submissions_lgb.csv',index=False)
 


# In[83]:


submissions_df


# In the predictions above, 0 means it isnt a badbuy and 1 means it is ... Thank you

# # Reference

# These links and resources were helpful in this analysis:
# 1. https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
# 2. https://lightgbm.readthedocs.io/en/v3.3.2/Parameters-Tuning.html

# # Future Work
# I would like to work on different project to further improve myself in creating efficient models.
# 

# In[ ]:


jovian.commit()


# In[ ]:


jovian.submit(assignment="my-ml-projects")


# In[ ]:




