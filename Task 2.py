#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, mean_squared_error, f1_score, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler



# Installing and calling the necessary libraries. 

# In[2]:


df_test = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")


# Importing the training and testing data. 

# In[3]:


df_train.isnull().sum()


# Checking for missing values.

# In[4]:


df_train.describe()


# The summary statistics of the training dataset shows that the features: HasCrCard, IsActiveMember and Exited are all binary.

# In[5]:


Y = ["Exited"]
X = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
x_train, x_test, y_train, y_test = train_test_split(df_train[X], df_train[Y], test_size = 0.2, random_state = 7)


# Spliting the training set into a seperate testing and training set in order to evaluate the model. 
# The training data was split into 20-80.

# Since our outcome is binary, ML methods such as Decision Trees, Random Forest, Boosted Tree and Logistic Regression would be suitible for building the model. 

# In[6]:


# Logistic Regression
log = LogisticRegression()

log.fit(x_train, np.ravel(y_train))
preds_log = log.predict(x_test)


# In[7]:


log_mse = mean_squared_error(y_test, preds_log)
log_mse


# However, from the summary statistics, we see that features such as CreditScore, Age, Balance, and EstimatedSalary have very wide ranges. 
# 
# Thus, the performance would improve if we scale these features such that their values become more uniform. 

# In[8]:


# Feature Scaling
scale = StandardScaler()

x_train_scale = scale.fit_transform(x_train)
x_test_scale = scale.transform(x_test)


# In[9]:


# Logistic Regression on Scaled Features

log.fit(x_train_scale, np.ravel(y_train)) 
preds_log_scale = log.predict(x_test_scale)


# In[10]:


# MSE of Scaled Logistic Regression
log_scale_mse = mean_squared_error(y_test, preds_log_scale)
log_scale_mse


# In[11]:


# Decision Tree
dc_tree = DecisionTreeClassifier(random_state = 7)

dc_tree.fit(x_train, y_train)
preds_dc_tree = dc_tree.predict(x_test)


# In[12]:


# MSE of Decision Tree Model
dc_tree_mse = mean_squared_error(y_test, preds_dc_tree)
dc_tree_mse


# In[13]:


# Random Forest
rf = RandomForestClassifier(random_state = 7)

rf = rf.fit(x_train, np.ravel(y_train))
preds_rf = rf.predict(x_test)


# In[14]:


rf_mse = mean_squared_error(y_test, preds_rf)
rf_mse


# In[15]:


# Boosted Tree
xgb_tree = xgb.XGBClassifier(objective = "binary:logistic")

xgb_tree.fit(x_train, y_train)
xgb_tree_preds = xgb_tree.predict(x_test)


# In[16]:


# MSE of Boosted Tree
xgb_tree_mse = mean_squared_error(y_test, xgb_tree_preds)
xgb_tree_mse


# From the MSEs, we see that XGBoost and Random Forest Classifiers produces the best results with MSEs of 0.1457 and 0.1512 respectively. 

# In[17]:


accuracy_score(y_test, preds_rf)


# In[18]:


accuracy_score(y_test, xgb_tree_preds)


# In[19]:


print(classification_report(y_test, preds_rf))


# In[20]:


print(classification_report(y_test, xgb_tree_preds))


# In[21]:


con_m_xgb = confusion_matrix(y_test, xgb_tree_preds)


# In[22]:


con_m_rf = confusion_matrix(y_test, preds_rf)


# In[23]:


disp = ConfusionMatrixDisplay(con_m_xgb, display_labels=None)


# In[24]:


disp = ConfusionMatrixDisplay(con_m_rf, display_labels=None)

disp.plot(cmap="Reds")

plt.title("Confusion Matrix RF")
plt.xlabel("Predicted")
plt.ylabel("Testing")
plt.show()


# In[25]:


disp = ConfusionMatrixDisplay(con_m_xgb, display_labels=None)

disp.plot(cmap="Reds")

plt.title("Confusion Matrix RF")
plt.xlabel("Predicted")
plt.ylabel("Testing")
plt.show()


# In[26]:


f1_score(y_test, preds_rf)


# In[27]:


f1_score(y_test, xgb_tree_preds)


# From the accuracy scores of 0.8488 and 0.8543, we can see that XGBoost predicted more accurately compared to Random Forest. Even though it was only be a small margin. The confusion matrix further examplifies this claim, where we can see that the True negatives of the XGBoost model is relatively smaller. One similar flaw of both models is the high False Negative rates. This problem can be fixed by tuning the parameters of each model. Similarily, in terms of the f1 scores of 0.5798 and 0.5951, we can see that XGBoost did relatively better than the Random Forest Classifier. Although a score of 0.5951 is still far from ideal, and can most likely improve with some tuning of parameters, due to the class imbalance of the dataset, tuning would most likely still be limited in its ability to improve the score. Due to time constraints, I did not have the opportunity to attempt to tune the models.
# 
# 

# In[28]:


# Predict test.csv

x_test_actual = df_test[X]

result = xgb_tree.predict(x_test_actual)
results_df = pd.DataFrame({"id": df_test["id"], "CustomerId": df_test["CustomerId"], "Exited": result})


# In[29]:


results_df.to_csv("Test_Exit.csv", index=False)

