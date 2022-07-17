# ML predictions

# Import of packages and the preprocessed data set
import pandas as pd
import numpy as np
from datetime import datetime
import collections
import glob
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import calendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score




#Import the sessions data set
path = r'Path file'
df = pd.read_csv(path, index_col=None, header=0)

#Import the event data set
path = r'Path file'
dfEvent = pd.read_csv(path, index_col=None, header=0)
print("Data loaded")



print(len(df),len(dfEvent))
#df = df[df["number_of_total_events_per_session_view_cart"]>1]
dfEvent = dfEvent[dfEvent.user_session_id_new.isin(df.user_session_id_new)]
print(len(df),len(dfEvent))

# Customization of data types
#################### Convert the data types - session dataframe
# Convert the session length and max time between two events to seconds
df.session_length = pd.DataFrame(df.session_length.map(lambda x: pd.to_timedelta(x).seconds)).values
df.max_time_between_two_events = pd.DataFrame(df.max_time_between_two_events.map(lambda x: pd.to_timedelta(x).seconds)).values
df['week_of_the_month'] = np.nan
df.loc[df['day'] <= 7, 'week_of_the_month'] = 1
df.loc[(df['day'] > 7) & (df['day'] <= 15), 'week_of_the_month'] = 2
df.loc[(df['day'] > 15) & (df['day']<= 22), 'week_of_the_month'] = 3
df.loc[(df['day'] > 22) & (df['day'] <= 31), 'week_of_the_month'] = 4
df['month'] = df['month'].apply(lambda x: calendar.month_name[x])
df['weekDay'] = df['weekDay'].apply(lambda x: calendar.day_name[x])
df = df.drop(columns=["day","hour"])#,"average_value_of_carts",'average_value_of_views'
df = df.astype({"user_session_id_new":'category',"weekDay":'category',"dayTime":'category',"PurchaseSession":'int', "session_length":'int64',"max_time_between_two_events":'int64',"month": 'category','week_of_the_month':"category"})
df = df.drop(["event_time", "user_id", "number_of_purchases", "user_session_id_new"], axis = 1)
df = df.rename(columns={'weekDay': 'week_day', 'weekDayOrNot': 'week_day_or_not', "dayTime":"day_time","newVisitorOrNot":"new_visitor_or_not","PurchaseSession":"purchase_session"})
#################### Convert the data types - event dataframe
dfEvent= dfEvent.astype({"user_id": 'category',"user_session_id_new":'category',"day":'int64',"weekDay":'category',"dayTime":'category', "product_id":'category',"category_code":'category',"category_id":'category',"brand":"category"})
print("Converted")



# Split the dataset into train and test set
# train set
X_train = df.loc[df['set'] == "train"].copy(deep=True).drop(["purchase_session","set"],axis=1)
y_train = df.loc[df['set'] == "train"].copy(deep=True).purchase_session
# test set
X_test = df.loc[df['set'] == "test"].copy(deep=True).drop(["purchase_session","set"],axis=1)
y_test = df.loc[df['set'] == "test"].copy(deep=True).purchase_session
print("Length of training set:", len(X_train), len(y_train))
print("Length of test set:", len(X_test), len(y_test))



# Remove correlated features
### Check for correlations between features in X_train
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = X_train.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
corr_features = correlation(X_train, 0.9)
len(set(corr_features))
#print(corr_features)
X_train = X_train.drop(corr_features,axis=1)
X_test = X_test.drop(corr_features,axis=1)



# One hot encoding of categorial features
numerical_columns  = X_train.select_dtypes(include=np.number).columns.to_list()
categoricalColumns = X_train.select_dtypes(['category']).columns.to_list()
X_train = pd.get_dummies(X_train, columns = categoricalColumns, drop_first=True)
X_test = pd.get_dummies(X_test, columns = categoricalColumns, drop_first=True)




# Creating prediction models
def evaluation_analysis(true_label,predicted):
    print("Accuracy",metrics.accuracy_score(true_label, predicted))
    print("F1 score",metrics.f1_score(true_label, predicted))    
    print("Precision score",metrics.precision_score(true_label, predicted)) 
    print("Recall score",metrics.recall_score(true_label, predicted))
    
    
def evaluation_roc(true_label,predicted_proba):
    print("AUC",metrics.roc_auc_score(true_label, predicted_proba))









# Logistic Regression
C = np.logspace(-4, 4, 50)
penalty = ['l1', 'l2']
sampling_value=(0.2,0.3,0.5)
class_weight = ["balanced"]
params = dict(classification__C= C, classification__penalty=penalty, classification__class_weight = class_weight, sampling__sampling_strategy=sampling_value)

model = Pipeline([
        ("scaler", StandardScaler()),
        ('sampling', SMOTE()),
        ('classification', LogisticRegression(solver='liblinear'))
    ])
cv = StratifiedKFold(n_splits=5)
grid = RandomizedSearchCV(model, params, scoring='f1', cv=cv, n_jobs=-1, n_iter=100, random_state=1)
result = grid.fit(X_train, y_train)
print("Logistic regression results:")
print('Best Hyperparameters: %s' % result.best_params_)
# Make a prediction on the test set
bestParameterLR = result.best_estimator_
y_hat = bestParameterLR.predict(X_test)
y_hat_proba = bestParameterLR.predict_proba(X_test)
evaluation_analysis(y_test, y_hat)
evaluation_roc(y_test, y_hat_proba[:,1])



# Decision tree and random forest
# Decision tree
criterion = ["gini", "entropy"]
max_features = ['sqrt', 'log2']
class_weight = ["balanced"]
min_samples_leaf = [0.00001, 0.0001, 0.001, 0.01]
sampling_value=(0.2,0.3,0.5)

params = dict(classification__criterion=criterion, classification__class_weight = class_weight, classification__max_features=max_features,classification__min_samples_leaf=min_samples_leaf, sampling__sampling_strategy=sampling_value)

model = Pipeline([
        ("scaler", StandardScaler()),
        ('sampling', SMOTE()),
        ('classification', DecisionTreeClassifier())
    ])
cv = StratifiedKFold(n_splits=5)
grid = RandomizedSearchCV(model, params, scoring='f1', cv=cv, n_jobs=-1, n_iter=100, random_state=1)
result = grid.fit(X_train, y_train)
print("Decision tree results:")
print('Best Hyperparameters: %s' % result.best_params_)
# Make a prediction on the test set
#bestParameterDT = DecisionTreeClassifier(result.best_params_)
bestParameterDT = result.best_estimator_
y_hat = bestParameterDT.predict(X_test)
y_hat_proba = bestParameterDT.predict_proba(X_test)
evaluation_analysis(y_test, y_hat)
evaluation_roc(y_test, y_hat_proba[:,1])




# Random forest
criterion = ["gini", "entropy"]
n_estimators = [100, 200, 300]
max_features = ['sqrt', 'log2']
min_samples_leaf = [0.00001, 0.0001, 0.001, 0.01]
class_weight = ["balanced"]
sampling_value=(0.2,0.3,0.5)

params = dict(classification__criterion=criterion, classification__class_weight = class_weight, classification__n_estimators= n_estimators, classification__max_features=max_features,classification__min_samples_leaf=min_samples_leaf, sampling__sampling_strategy=sampling_value)

model = Pipeline([
        ("scaler", StandardScaler()),
        ('sampling', SMOTE()),
        ('classification', RandomForestClassifier())
    ])
cv = StratifiedKFold(n_splits=5)
grid = RandomizedSearchCV(model, params, scoring='f1', cv=cv, n_jobs=-1, n_iter=100, random_state=1)
result = grid.fit(X_train, y_train)
print("Random forest results:")
print('Best Hyperparameters: %s' % result.best_params_)
# Make a prediction on the test set
bestParameterRF =result.best_estimator_
y_hat = bestParameterRF.predict(X_test)
y_hat_proba = bestParameterRF.predict_proba(X_test)
evaluation_analysis(y_test, y_hat)
evaluation_roc(y_test, y_hat_proba[:,1])






# Gradient boosting machine
n_estimators = [100, 200, 300]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [2, 3, 4, 6, 8]
sampling_value=(0.2,0.3, 0.5)
max_features = ["sqrt","log2"]

params = dict(classification__n_estimators= n_estimators, classification__learning_rate=learning_rate,classification__max_depth=max_depth, classification__subsample=subsample, classification__max_features=max_features, sampling__sampling_strategy=sampling_value)
model = Pipeline([
        ("scaler", StandardScaler()),
        ('sampling', SMOTE()),
        ('classification', GradientBoostingClassifier())
    ])
cv = StratifiedKFold(n_splits=5)
grid = RandomizedSearchCV(model, params, scoring='f1', cv=cv, n_jobs=-1, n_iter=3, random_state=1)
result = grid.fit(X_train, y_train)
print("Gradient boosting machine results:")
print('Best Hyperparameters: %s' % result.best_params_)
# Make a prediction on the test set
bestParameterGB =result.best_estimator_
y_hat = bestParameterGB.predict(X_test)
y_hat_proba = bestParameterGB.predict_proba(X_test)
evaluation_analysis(y_test, y_hat)
evaluation_roc(y_test, y_hat_proba[:,1])










# AUC Plot

# Plot AUC curves in one plot
plt.figure(0).clf()

pred = bestParameterLR.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred),4)
plt.plot(fpr,tpr,label="Logistic regression, AUC="+str(auc))

pred =bestParameterDT.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc =round(metrics.roc_auc_score(y_test, pred),4)
plt.plot(fpr,tpr,label="Decision tree, AUC="+str(auc))



pred =bestParameterRF.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred),4)
plt.plot(fpr,tpr,label="Random forest, AUC="+str(auc))


pred =bestParameterGB.predict_proba(X_test)[:,1]
fpr, tpr, thresh = metrics.roc_curve(y_test, pred)
auc = round(metrics.roc_auc_score(y_test, pred),4)
plt.plot(fpr,tpr,label="Gradient boosting, AUC="+str(auc))

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(loc=0)
plt.title('Receiver Operating Characteristic - multicategory data set')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('File path', format='eps')




# Analyze the validation/test error for different sampling strategies
results_cv = pd.DataFrame.from_dict(result.cv_results_, orient='columns')
print(results_cv.columns)
sns.relplot(data=results_cv ,
    kind='line',
    x='param_sampling__sampling_strategy',
    y='mean_test_score')
plt.show()

























