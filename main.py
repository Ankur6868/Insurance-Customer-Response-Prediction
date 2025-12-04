import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
sns.set(style='whitegrid')

import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score,confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report
# Load the dataset
df= pd.read_csv('data.csv')
# Data preprocessing
numerical_columns=['Age', 'Region_Code','Annual_Premium','Vintage']
categorical_columns=['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']
df.Gender.replace({'Male':1,'Female':2}, inplace=True)
df.Vehicle_Damage.replace({'Yes':1,'No':0}, inplace=True)
df.Vehicle_Age.replace({'< 1 Year':1,'1-2 Year':2,'> 2 Years':3}, inplace=True)
df.drop("id", inplace=True, axis=1)
# Select features
trainDF = df.loc[:,['Vehicle_Damage','Previously_Insured','Vehicle_Age','Response']]
X = trainDF.iloc[:,0:3]
y = trainDF.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

X_train_resampled,X_test_resampled,y_train_resampled,y_test_resampled = train_test_split(X_resampled,y_resampled,test_size=0.2)
clf_resampled = RandomForestClassifier(max_depth=10, random_state=0)

clf_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = clf_resampled.predict(X_test_resampled)
random_search = {'criterion': ['entropy', 'gini'],
               'max_depth': [2,3,4,5,6,7,10],
               'min_samples_leaf': [4, 6, 8],
               'min_samples_split': [5, 7,10],
               'n_estimators': [300]}

clf = RandomForestClassifier()
model_resampled = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 10,
                               cv = 4, verbose= 1, random_state= 101, n_jobs = -1)
model_resampled.fit(X_train_resampled,y_train_resampled)


joblib.dump(model_resampled, "rf_model.joblib")
