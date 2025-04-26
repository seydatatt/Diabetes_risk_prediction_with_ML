#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


# In[57]:


df = pd.read_csv("diabetes.csv")


# In[58]:


df


# In[59]:


df_names = df.columns
print(df_names)


# In[60]:


df.info()


# In[61]:


df.describe()


# In[62]:


plt.figure()
sns.pairplot(df, hue="Outcome")
plt.show()


# In[63]:


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True,fmt =".2f", linewidths= 0.5) 
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()

plot_correlation_heatmap(df)


# In[64]:


def detect_outliers_iqr(df):
    outlier_indices = []  # Fix variable name to be consistent
    outlier_df = pd.DataFrame()
    
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1  # Interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR  # Correct the typo here (from 1,5 to 1.5)
   
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)] 
        outlier_indices.extend(outliers_in_col.index)  # Fixed method name (outlier_indices_extend -> extend)
        outlier_df = pd.concat([outlier_df, outliers_in_col], axis=0)

    # Remove duplicate indices  
    outlier_indices = list(set(outlier_indices))   
    # Remove duplicate rows in the outliers dataframe
    outlier_df = outlier_df.drop_duplicates()
    return outlier_df, outlier_indices

outlier_df, outlier_indices = detect_outliers_iqr(df)
#remove outliers from the dataframe 
df_cleaned = df.drop(outlier_indices).reset_index(drop=True)



# In[65]:


#train test split
X = df_cleaned.drop(["Outcome"], axis = 1)
y= df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[110]:


#standartizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[111]:


print(X_train_scaled)


# In[112]:


print(X_test_scaled)


# In[113]:


#Model training and evalution
def getBasedModel():
    basedModels= []
    basedModels.append(("LR", LogisticRegression()))
    basedModels.append(("DT", DecisionTreeClassifier()))
    basedModels.append(("KNN", KNeighborsClassifier()))
    basedModels.append(("NB", GaussianNB()))
    basedModels.append(("SVM", SVC()))
    basedModels.append(("AdaB", AdaBoostClassifier()))
    basedModels.append(("GB",GradientBoostingClassifier()))
    basedModels.append(("RF",RandomForestClassifier()))
    return basedModels

def basedModelsTraning(X_train_scaled, y_train,models):
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits = 10)
        cv_results = cross_val_score(model, X_train_scaled , y_train,cv = kfold, scoring = "accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy {cv_results.mean()}, std: {cv_results.std()}")

    return names, results 
models = getBasedModel()


# In[114]:


display(models)


# In[115]:


def plot_box(names, results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize= (12,8))
    sns.boxplot(data = df)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.show()


# In[116]:


names, results = basedModelsTraning(X_train_scaled, y_train, models)


# In[117]:


plot_box(names, results)


# In[124]:


#hyperparameter tuning
#DT hyperparameter set
param_grid = {
    "criterion": ["gini","entropy"],
    "max_depth": [10,20,30,40,50],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4]
}

dt = DecisionTreeClassifier()


# In[125]:


#grid search cv 
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, scoring = "accuracy")


# In[126]:


#training
grid_search.fit(X_train_scaled, y_train)
print("En iyi parametreleri:", grid_search.best_params_)


# In[129]:


best_dt_model = grid_search.best_estimator_
print(best_dt_model)


# In[131]:


y_pred = best_dt_model.predict(X_test_scaled)
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))


# In[132]:


print("Classification_report")
print(classification_report(y_test, y_pred))


# In[136]:


#Model testing with real data
new_data = np.array([[8,188,32,34,0,44.6,0.787,81]])
new_prediction = best_dt_model.predict(new_data)


# In[137]:


print("New Prediction", new_prediction)


# In[ ]:




