#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


df=pd.read_csv(r"D:\v cube projects\Hottle_Booking_Canclation\dataset.csv")
df
#D:\v cube projects\Hottle_Booking_Canclation\dataset.csv


# In[12]:


# 1. Display first 5 rows
df.head()


# In[13]:


# 2. Check number of rows and columns
df.shape


# In[14]:


# 4. Check data types and missing values
df.info()


# In[15]:


df.isnull().sum()


# In[16]:


df.drop(columns=['company'],inplace=True)


# In[17]:


# Numerical columns: fill missing values with median
df["children"] = df["children"].fillna(df["children"].median())

# Categorical columns: fill missing values with mode
df["country"] = df["country"].fillna(df["country"].mode()[0])

# Agent and company are ID columns, so use mode
df["agent"] = df["agent"].fillna(df["agent"].mode()[0])
#df["company"] = df["company"].fillna(df["company"].mode()[0])

# Check missing values
print(df[["children", "country", "agent"]].isnull().sum())


# In[18]:


# 6. Check duplicate rows
df.duplicated().sum()


# In[19]:


df.isnull().sum()


# In[20]:


print(df["is_canceled"].value_counts())

print(df["is_canceled"].value_counts(normalize=True) * 100)


# In[21]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Select numerical columns
num_cols = df.select_dtypes(include="number").columns.tolist()

# Exclude target and identifier columns from outlier analysis
exclude_cols = ["is_canceled", "agent", "company"]
features = [col for col in num_cols if col not in exclude_cols]

# Create multiple rows of boxplots
n_cols = 3
n_rows = math.ceil(len(features) / n_cols)

plt.figure(figsize=(18, 5 * n_rows))

for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# In[22]:


# 1. Select numerical columns
num_cols = df.select_dtypes(include="number").columns

# 2. Exclude target and identifier columns
exclude_cols = ["is_canceled", "agent", "company"]
features = [col for col in num_cols if col not in exclude_cols]

# 3. Calculate IQR limits and remove outlier rows
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

# 4. Check final dataset shape
print("Dataset shape after removing outliers:", df.shape)


# In[23]:


# Remove columns that could cause data leakage
df = df.drop(
    columns=["reservation_status", "reservation_status_date"],
    errors="ignore"
)

print(df.shape)


# In[24]:


# Convert the date-related columns
df["arrival_date_month"] = df["arrival_date_month"].str.strip()
df


# In[25]:


X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# In[26]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# # pipelining

# In[27]:


import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score, classification_report


# # seperated num cols and cat cols

# In[28]:


num_cols = X.select_dtypes(include="number").columns.tolist()

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)


# In[29]:


# Feature type identification

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

categorical_features = X.select_dtypes(include=['object','category','string']).columns


# In[30]:


# Preprocessing pipelines
from sklearn.preprocessing import StandardScaler
numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])
numeric_transformer 


# In[31]:


categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
categorical_transformer


# In[32]:


# ColumnTransformer

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])
preprocessor


# # PCA

# In[33]:


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# In[34]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)
print("Training data shape:", X_train_scaled.shape)
print("Number of columns:", X_train_scaled.shape[1])


# In[35]:


from sklearn.decomposition import PCA

X_train_dense = X_train_scaled.toarray()
X_test_dense = X_test_scaled.toarray()

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train_dense)
X_test_pca = pca.transform(X_test_dense)

print("Before PCA:", X_train_dense.shape)
print("After PCA:", X_train_pca.shape)
print("Explained variance:", pca.explained_variance_ratio_.sum())


# In[36]:


print("Original columns:", X_train.shape[1])
print("After preprocessing:", X_train_processed.shape[1])


# # filte method

# In[37]:


from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=50)

X_train_selected = selector.fit_transform(
    X_train_processed, y_train
)

X_test_selected = selector.transform(
    X_test_processed
)
print("Before:", X_train_processed.shape)
print("After:", X_train_selected.shape)
print("Before Filter Method:", X_train_processed.shape[1])
print("After Filter Method:", X_train_selected.shape[1])


# # BASE MODEL

# In[38]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)


# In[39]:


from sklearn.linear_model import LogisticRegression

# Create base model
base_model = LogisticRegression(max_iter=1000)

# Train the model
base_model.fit(X_train_scaled, y_train)


# In[40]:


# test the model
y_pred = base_model.predict(X_test_scaled)


# In[41]:


from sklearn.metrics import accuracy_score, classification_report

print("Base Model Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[42]:


from sklearn.metrics import accuracy_score, classification_report

print("Base Model Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[43]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    BaggingClassifier,
    #BoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


# In[44]:


# 1. Bagging Pipeline
bagging_pipeline = Pipeline([
    ("preprocessor", (preprocessor)),
    ("classifier", BaggingClassifier(

        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
bagging_pipeline


# In[45]:


# 2. Boosting Pipeline (AdaBoost)
boosting_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ))
])
boosting_pipeline


# In[46]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create the complete pipeline
random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
random_forest_pipeline


# In[47]:


# 4. AdaBoost Pipeline
adaboost_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ))
])
adaboost_pipeline


# In[48]:


# 5. Gradient Boosting Pipeline
gradient_boosting_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ))
])
gradient_boosting_pipeline


# In[49]:


#6. DECISION TREE PIPELEINE
decision_tree_pipeline=Pipeline([
     ("preprocessor", clone(preprocessor)),
    ("classifier", DecisionTreeClassifier(
        #n_estimators=100,
        random_state=42
    ))
])
decision_tree_pipeline


# In[50]:


boosting_pipeline.fit(X_train, y_train)

random_forest_pipeline.fit(X_train, y_train)

adaboost_pipeline.fit(X_train, y_train)

gradient_boosting_pipeline.fit(X_train, y_train)
decision_tree_pipeline.fit(X_train,y_train)

print("All models trained successfully!")


# # hyperparametertuning

# In[51]:


import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.model_selection import RandomizedSearchCV


# In[52]:


#BOOSTING
param_grid = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0],
    #'classifier__max_depth': [1, 2, 3, 5]
}

random = RandomizedSearchCV(
    boosting_pipeline,
   # estimator=adaboost_pipeline,
    param_distributions=param_grid,
    #n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("Boosting Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_boosting = random_result.best_estimator_


# In[53]:


from sklearn.model_selection import RandomizedSearchCV
import time

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

random = RandomizedSearchCV(
    random_forest_pipeline,
    param_distributions=param_grid,
    #n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("Random Forest Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_random_forest = random_result.best_estimator_


# In[54]:


from sklearn.model_selection import RandomizedSearchCV
import time

param_grid = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0]
}

random = RandomizedSearchCV(
    adaboost_pipeline,
    param_distributions=param_grid,
    #n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("AdaBoost Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_adaboost = random_result.best_estimator_


# In[47]:


from sklearn.model_selection import RandomizedSearchCV
import time

param_grid = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [2, 3, 5],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

random = RandomizedSearchCV(
    gradient_boosting_pipeline,
    param_distributions=param_grid,
    #n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("Gradient Boosting Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_gradient_boosting = random_result.best_estimator_


# In[60]:


best_gradient_boosting = random_result.best_estimator_


# In[55]:


#  decision tree
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features': [None, 'sqrt', 'log2']
}

# RandomizedSearchCV

dt = DecisionTreeClassifier(random_state=42)
random = RandomizedSearchCV(
    estimator=decision_tree_pipeline,
    param_distributions=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("\nDecisionTree Best Accuracy: %.4f" % random_result.best_score_)
print("Best Params:", random_result.best_params_)
print("Execution Time: %.2f seconds" % random_time)
best_decision_tree = random_result.best_estimator_


# In[61]:


best_models = {
    "Boosting": best_boosting,
    "Random Forest": best_random_forest,
    "AdaBoost": best_adaboost,
    "Gradient Boosting": best_gradient_boosting
}
best_models


# # evalution

# In[62]:


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from IPython.display import display
results = []

for name, model in best_models.items():

    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    })

results_df = pd.DataFrame(results)

display(
    results_df.sort_values(by="F1 Score", ascending=False)
)


# # cross validation

# In[63]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    random_forest_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("Cross-validation F1 scores:", cv_scores)
print("Mean F1 Score:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# In[64]:


from sklearn.model_selection import cross_val_score
import pandas as pd

cv_results = []

for name, model in best_models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "Mean F1 Score": scores.mean(),
        "Std": scores.std()
    })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df = cv_results_df.sort_values(
    by="Mean F1 Score",
    ascending=False
).reset_index(drop=True)

print(cv_results_df)


# # comparing best model

# In[66]:


results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)
best_model_name = results_df.loc[0, "Model"]

print("Best Model:", best_model_name)
print("Best F1 Score:", results_df.loc[0, "F1 Score"])

best_model = best_models[best_model_name]


# In[67]:


import matplotlib.pyplot as plt

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 5))

plt.title("Comparison of Tuned Ensemble Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# 
# # Confusion matrix and classification report

# In[68]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Predict test data
best_model = best_random_forest
y_pred = best_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[69]:


# Select your best tuned model
best_model = best_random_forest

# Take one booking from test data
new_booking = X_test.iloc[[0]]

# Get probabilities
probabilities = best_model.predict_proba(new_booking)

print("Not canceled probability:", probabilities[0][0])
print("Canceled probability:", probabilities[0][1])


# # save the model

# In[70]:


import joblib

joblib.dump(best_model, "hotel_cancellation_model.pkl")

print("Model saved successfully!")


# In[71]:


best_model = joblib.load("hotel_cancellation_model.pkl")
print("Model loaded successfully!")


# # Test predictions on new booking data

# In[72]:


# Take one booking from the test dataset
new_booking = X_test.iloc[[0]].copy()

print(new_booking)


# # Predict whether it will be canceled

# In[73]:


prediction = best_model.predict(new_booking)

if prediction[0] == 1:
    print("Prediction: Booking may be canceled")
else:
    print("Prediction: Booking may not be canceled")


# In[74]:


probabilities = best_model.predict_proba(new_booking)

print("Not canceled probability:", probabilities[0][0])
print("Canceled probability:", probabilities[0][1])


# In[ ]:




