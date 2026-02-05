#!/usr/bin/env python
# coding: utf-8

# # Diamond Price Prediction using KNN & Streamlit Deployment

# In[12]:


import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import pickle


# # Load dataset

# In[13]:


df = pd.read_csv("diamonds.csv")
df


# # Separate target and features
# 

# In[14]:


y = df["price"]


# In[15]:


X = df.drop("price", axis=1)


# # Identify categorical and numerical columns

# In[16]:


categorical_cols = X.select_dtypes(include=["object"]).columns


# In[17]:


numerical_cols = X.select_dtypes(exclude=["object"]).columns


# # Split
# 

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # Preprocessor

# In[19]:


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)


# # Full pipeline

# In[20]:


knn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=5))
])


# In[21]:


# Fit pipeline
knn_pipeline.fit(X_train, y_train)


# # Evaluation

# In[25]:


import numpy as np
y_pred = knn_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


# # Save the full pipeline
# 
# 

# In[26]:


with open("diamond_knn_model.pkl", "wb") as f:
    pickle.dump(knn_pipeline, f)

