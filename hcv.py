#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import ADASYN


# ## Data loading

# In[2]:


data = pd.read_csv("hcv_data.csv", sep=',')
data.head()


# # Preprocessing 

# In[3]:


data.drop(data.columns[data.columns.str.contains('unnamed', case = False)], axis = 1, inplace = True)


# In[4]:


data['Category'].unique()


# In[5]:


sex = {'m': 1,'f': 0}
category = {
    '0=Blood Donor': 0, '0s=suspect Blood Donor': 0, 
    '1=Hepatitis': 1, '2=Fibrosis': 1, '3=Cirrhosis': 1
}
data.Sex = [sex[item] for item in data.Sex]
data.Category = [category[item] for item in data.Category]
data.head()


# In[6]:


data.describe()


# In[7]:


data.shape


# In[8]:


data.dtypes


# ### Checking missing values 

# In[9]:


names = data.columns[data.isna().any()].tolist()
names 


# In[10]:


data.isna().sum()


# In[11]:


for name in names:
    data[name].fillna((data[name].mean()), inplace=True)


# In[12]:


data.isna().sum()


# ### The outliers

# In[13]:


plt.figure(dpi=150, figsize=(12,14))

plt.subplot(4, 2, 1)
sns.boxplot(data=data[['Age', 'Sex']], palette="magma_r")

plt.subplot(4, 2, 2)
sns.boxplot(data=data[['ALB', 'ALP', 'ALT', 'AST']], palette="magma_r")

plt.subplot(4, 2, 3)
sns.boxplot(data=data[['BIL', 'CHE', 'CHOL']], palette="magma_r")

plt.subplot(4, 2, 4)
sns.boxplot(data=data[['CREA', 'GGT', 'PROT']], palette="magma_r")

plt.suptitle("Boxplots to visualize outliers", fontsize=20)

plt.savefig('saved_figs/outliers.svg', format='svg')


# In[14]:


skewness = [data[feature].skew() for feature in data.columns[0:12].drop('Sex')]

plt.figure(figsize=(14,16), dpi=250)
for i, skew, feature in zip(range(0,11), skewness, data.columns[0:12].drop('Sex')):
    plt.subplot(11, 2, i+1)
    sns.distplot(data[feature], color="#411074", label="Skewness: %.2f"%(skew))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.suptitle("Skewness of features", fontsize=26, y=1.05)

plt.savefig('saved_figs/skewness.svg', format='png')


# In[15]:


# Q1 
q1 = data.quantile(0.25)
# Q3
q3 = data.quantile(0.75)
# IQR
IQR = q3 - q1
# Outlier range
upper = q3 + IQR * 1.5
lower = q1 - IQR * 1.5
upper_dict = dict(upper)
lower_dict = dict(lower)


# In[16]:


for i, v in data.items():
    v_col = v[( v<= lower_dict[i]) | (v >= upper_dict[i])]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
    print("Column {} outliers = {} => {}%".format(i, len(v_col), round((perc),3)))


# In[17]:


#Using log transformation
scaler = RobustScaler()
columns_to_scale = ["AST", "GGT", "BIL", 'ALT']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
print("The first 5 rows of transformed data are")
data.head()


# ### Plotting imbalanced data 

# In[18]:


sns.countplot(data['Category'], label='count').set(title='')
data['Category'].value_counts()
plt.savefig('saved_figs/class_count.svg', format='svg')


# ## Data visualisation 

# In[19]:


sns.countplot(data['Sex'], label='count')
data['Sex'].value_counts()


# In[20]:


categories = ['No HVC', 'HVC']

plt.figure(figsize=(10, 10))
for i, name in enumerate(categories):
    df = data[data.Category == i]
    sns.distplot(df['Age'], kde=True, label=name)

# Plot formatting
plt.legend(prop={'size': 12})
plt.title('Age distribution with division of HCV')
plt.xlabel('Age')
plt.ylabel('Density')


# ### Correlation and heatmap

# In[21]:


data.corr()


# In[22]:


plt.figure(figsize=(20,20))  
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.0%')
plt.savefig('saved_figs/heatmap.svg', format='svg')


# > Column Category outliers = 615 => 100.0%
# > Column Age outliers = 1 => 0.163%
# > Column Sex outliers = 0 => 0.0%
# > Column ALB outliers = 27 => 4.39%
# > Column ALP outliers = 14 => 2.276%
# > Column ALT outliers = 36 => 5.854%
# > Column AST outliers = 64 => 10.407%
# > Column BIL outliers = 47 => 7.642%
# > Column CHE outliers = 24 => 3.902%
# > Column CHOL outliers = 12 => 1.951%
# > Column CREA outliers = 12 => 1.951%
# > Column GGT outliers = 65 => 10.569%
# > Column PROT outliers = 20 => 3.252%

# In[23]:


X = data.drop('Category', axis=1).copy()
y = data['Category'].copy()


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size = 0.30, random_state = 0, stratify = y)


# ### Oversampling the data with ADASYN algorithm

# In[25]:


y_values = y_train.values
X_train, y_train = ADASYN(random_state=0).fit_resample(X_train, y_train)


# In[26]:


unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))


# # Training 

# In[27]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=0)) 
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=0))
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier()) 

tree = DecisionTreeClassifier(random_state=0)
tree = tree.fit(X_train, y_train)

forest = RandomForestClassifier(random_state=0)
forest.fit(X_train, y_train)

ada = AdaBoostClassifier(random_state=0)
ada.fit(X_train, y_train)

param_C = np.arange(0.01, 1.0, 0.01)
param_gamma = np.arange(0.01, 1.0, 0.01)
param_pca = np.arange(2, 25)
param_knn = np.arange(2, 25)

param_grid_lr = [{'logisticregression__C': param_C,'pca__n_components': param_pca}]

param_grid_svc = [{'svc__C': param_C, 
               'svc__kernel': ['linear']},
              {'svc__C': param_gamma, 
               'svc__gamma': param_C, 
               'svc__kernel': ['rbf']}]

param_grid_knn = [{'kneighborsclassifier__n_neighbors': param_knn}]


path = tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

random_grid_tree = {'ccp_alpha': ccp_alphas,
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.arange(1, 30)]}

random_grid_forest = {'n_estimators': [x for x in np.arange(10, 300, 5)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.arange(1, 20)],
               'min_samples_split': np.arange(1, 20),
               'min_samples_leaf': np.arange(1, 5),
               'bootstrap': [True, False]}

random_grid_ada = {'n_estimators': [x for x in np.arange(10, 400, 5)],
                  'learning_rate': np.arange(0.1, 4.1, 0.1)}

classifiers_1 = [pipe_lr, pipe_svc, pipe_knn]
parameters_1 = [param_grid_lr, param_grid_svc, param_grid_knn]
names_1 = ['Logistic Regression', 'SVC', 'KNN']

classifiers_2 = [tree, forest, ada]
parameters_2 = [random_grid_tree, random_grid_forest, random_grid_ada]
names_2 = ['Decision Tree', 'Random Forest', 'AdaBoost']

for clf, param, name in zip(classifiers_1, parameters_1, names_1):
    clf.fit(X_train, y_train)
    gs = GridSearchCV(estimator=clf, 
                      param_grid=param, 
                      scoring='accuracy', 
                      cv=5,
                      n_jobs=-1)
    
    
    gs = gs.fit(X_train, y_train)
    print(name)
    print(round(gs.best_score_, 2))
    print(gs.best_params_)
    y_pred = gs.predict(X_test)
    print("The test accuracy score of {} after hyper-parameter tuning is ".format(name), round(accuracy_score(y_test, y_pred),2))
    print("")

for clf, param, name in zip(classifiers_2, parameters_2, names_2):
    rs = RandomizedSearchCV(estimator=clf, 
                                   param_distributions=param, 
                                   n_iter=100, cv=5, verbose=2, 
                                   random_state=0, n_jobs=-1)

    # # Fit the random search model
    print(name)
    rs.fit(X_train, y_train)
    print(rs.best_params_)
    print('accuracy:', round(rs.best_score_, 2))
    y_pred = rs.predict(X_test)
    print("The test accuracy score of {} after hyper-parameter tuning is".format(name), round(accuracy_score(y_test, y_pred),2))
    print("")

    


# In[28]:


pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=11), LogisticRegression(C=0.67, random_state=0))
pipe_svc = make_pipeline(StandardScaler(), SVC(C=0.8, gamma=0.08, kernel='rbf', random_state=0, probability=True))
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))

pipe_tree = make_pipeline(StandardScaler(), DecisionTreeClassifier(ccp_alpha=0, random_state=0, max_features = 'sqrt',
                                                                     max_depth=28, criterion = 'gini'))

pipe_forest = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=175, random_state=0,
                                                                      min_samples_leaf=1, max_features = 'sqrt',
                                                                     max_depth=10, bootstrap=False,
                                                                    min_samples_split=6))

pipe_ada = make_pipeline(StandardScaler(), AdaBoostClassifier(n_estimators=385, random_state=0, learning_rate = 1.7))
classifiers = [pipe_lr, pipe_svc, pipe_knn, pipe_tree, pipe_forest, pipe_ada]
names = names_1
names += names_2


# In[29]:


for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=5, n_jobs=-1)
    print(name)
    print('The training k-fold accuracy score: %.3f +/- %.3f' % ((np.mean(scores)), np.std(scores)))


# # Prediction  

# In[30]:


for clf, name in zip(classifiers, names):
    probas = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(dpi=200, figsize=(14,7))
    plt.plot(fpr, tpr, color="purple", marker="o", label="ROC (AUC = %.2f)" % (roc_auc))
    plt.plot([0, 1], [0,1], linestyle="--", color="gray", label="Random guessing")
    plt.plot([0,0,1], [0,1,1], linestyle=":", color="black", label="Perfect performance")
    plt.legend(loc="best")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve for {}".format(name), fontsize=18)
    plt.savefig('saved_figs/'+name+'_ROC.svg', format='svg', dpi=fig.dpi)


# In[32]:


for clf, name in zip(classifiers, names):
    fig = plt.figure(dpi=200, figsize=(10,10))
    plot_confusion_matrix(clf, X_test, y_test, display_labels=categories)
    print('Test accuracy for {}:'.format(name), round(accuracy_score(y_test, clf.predict(X_test)), 2))
    print("")
    print(classification_report(y_test, clf.predict(X_test)))
    plt.savefig('saved_figs/'+name+'_conf_matrix.svg', format='svg', dpi=fig.dpi)
    plt.show()

