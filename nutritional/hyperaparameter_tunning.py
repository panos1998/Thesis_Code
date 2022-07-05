#%%
import pandas as pd
from sklearn import tree
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import NuSVC
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
df = pd.read_csv(filepath_or_buffer='test.csv', usecols=range(20))
df['diabe'] = [0 for _ in range(119)]
df['diabe'] = np.where((df['FPG (mg/dL)']>126) & (df['Diastolic BP (mmHg)']>85) & (df['Systolic BP (mmHg)']>135), 1, 0)
df.to_csv('dataset_with_class.csv',index_label=False, index=False)
#
#print(df.groupby('diabe').count())
fig, ax = plt.subplots(2,2)
summary = df.groupby('diabe').describe().unstack()
print(summary) 
names = ['BMI','Wei','Wai','Sys', 'Dias','HDL','LDL','Chol','Ath', 'TAG', 'FPG','HbA','CRP','Oxi',
'Grain','Fruit', 'Veg','Pro', 'Dairy', 'Cal']
mean_list = [(summary.iloc[16*i+2], summary.iloc[16*i+3]) for i in range(0,20)]
healthy = [(feature, mean_item[0],'healthy') for feature,mean_item in zip(names,mean_list)] 
diseased = [(feature, mean_item[1],'diseased') for feature,mean_item in zip(names,mean_list)] 
blood_1 = healthy[0:8]+diseased[0:8]
blood_2 = healthy[9:11]+ diseased[9:11]
oxi= [healthy[13], diseased[13]]
final = blood_1+blood_2+oxi
stats_df = pd.DataFrame(final,columns=['Features','Values', 'Status'])
plt.suptitle('Healthy vs Diseased')
sns.barplot(palette='turbo',ax=ax[0,0],x='Features', y='Values', hue='Status',data=stats_df)
final = [healthy[-1],diseased[-1]]
stats_df = pd.DataFrame(final,columns=['Features','Values', 'Status'])
sns.barplot(palette='turbo',ax=ax[0,1],x='Features', y='Values', hue='Status',data=stats_df)

final = [healthy[8],diseased[8]] + healthy[11:13]+diseased[11:13]
stats_df = pd.DataFrame(final,columns=['Features','Values', 'Status'])
sns.barplot(palette='turbo',ax=ax[1,0],x='Features', y='Values', hue='Status',data=stats_df)

final = healthy[14:19]+ diseased[14:19]
stats_df = pd.DataFrame(final,columns=['Features','Values', 'Status'])
sns.barplot(palette='turbo',ax=ax[1,1],x='Features', y='Values', hue='Status',data=stats_df)
plt.tight_layout()
plt.plot()





#%%
df.drop(['FPG (mg/dL)', 'Diastolic BP (mmHg)', 'Systolic BP (mmHg)', 'HDL-c (mg/dl)',
'LDL-c (mg/dl)', 'Total Cholesterol (mg/dL)', 'Atherogenenciity index (Total/ HDL)',
'TAG (mg/dL)','HbA1c (%)', 'CRP (mg/L)'],
inplace=True, axis=1)
print(df.head())
imputer = IterativeImputer(min_value=[0 for _ in range(10)])
X_imputed = imputer.fit_transform(df.iloc[:,:-1])
diabe = df.iloc[:,-1]
df = pd.DataFrame(X_imputed, columns = df.columns[:-1])
df['diabe'] = diabe
print(df.isna().sum())
standarization = StandardScaler()
y = df.iloc[:,-1]
X = df.iloc[:,:-1]
X_standarized = standarization.fit_transform(X)
pickle.dump(standarization,open('scaler.sav','wb'))
#%%
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X_standarized)
pickle.dump(normalizer,open('normalizer.sav','wb'))
clf = NuSVC(nu=0.3)
grid = {'nu':[0.1, 0.2, 0.3], 'kernel':['linear', 'poly', 'rbf'], 'gamma':['scale', 'auto']}
forest_grid = {'n_estimators':[10, 25, 50],'min_samples_leaf':[1,2,5], 'min_samples_split':[1,2,5]}
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.5,stratify=y, random_state=0)
clf = GridSearchCV(NuSVC(probability=True), grid, cv=5,scoring="recall_macro")
clf.fit(X_train, y_train)
model = clf.best_estimator_
forest_clf=GridSearchCV(ExtraTreesClassifier(random_state=5), forest_grid,cv=4, scoring='recall_macro')
forest_clf.fit(X_train,y_train)
forest=forest_clf.best_estimator_
nu_svc = model
nu_svc_isotonic = CalibratedClassifierCV(nu_svc,cv=5,method='isotonic')
nu_svc_sigmoid = CalibratedClassifierCV(nu_svc,cv=5,method='sigmoid')
clf_list =[(nu_svc,'svc'),(nu_svc_isotonic,'isotonic svc'),(nu_svc_sigmoid,'sigmoid svc'),(forest,'forest')]
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=5,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display
#pickle.dump(nu_svc_isotonic,open('model.sav','wb'))
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (SVC) unbalanced")
# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0),(3,1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()

from collections import defaultdict
from sklearn.metrics import (
    brier_score_loss,
    log_loss
)

scores = defaultdict(list)
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores["Classifier"].append(name)

    for metric in [brier_score_loss, log_loss]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))
score_df = pd.DataFrame(scores).set_index("Classifier")
score_df.round(decimals=3)
print(score_df)


##############
result_forest = permutation_importance(forest, X_test, y_test, random_state=42)
unbalanced_forest = result_forest.importances_mean
unbalanced_tuple_forest = [(feature,unbalanced_item,'unbalanced') for feature,unbalanced_item in zip(df.columns[:-1],unbalanced_forest)]
print('Random Forest metrics')
y_true, y_pred = y_test, forest.predict(X_test)
tn, fp, fn, tp= confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tn+tp)/(tn+tp+fn+fp)
auc = roc_auc_score(y_true, y_pred)
print(f'auc: {auc}')
print(f'accuracy: {accuracy}    sensitivity: {sensitivity}    specificity: {specificity}')
report =classification_report(y_true, y_pred, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_csv('report_forest.csv')
print(report)
print()
#####
result = permutation_importance(clf, X_test, y_test, random_state=42)
unbalanced = result.importances_mean
unbalanced_tuple = [(feature,unbalanced_item,'unbalanced') for feature,unbalanced_item in zip(df.columns[:-1],unbalanced)]
print(unbalanced_tuple)
print(result.importances_mean)
print("Best parameters set found on development set:")
print()
#print(clf.best_params_)
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print('Nu-SVM metrics')
y_true, y_pred = y_test, clf.predict(X_test)
tn, fp, fn, tp= confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tn+tp)/(tn+tp+fn+fp)
auc = roc_auc_score(y_true, y_pred)
print(f'auc: {auc}')
print(f'accuracy: {accuracy}    sensitivity: {sensitivity}    specificity: {specificity}')
report =classification_report(y_true, y_pred, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_csv('report.csv')
print(report)
print()
#%%
sm = SMOTE(random_state=42,k_neighbors=8)
forest_grid = {'n_estimators':[10, 25,50],'min_samples_leaf':[1,2, 5], 'min_samples_leaf':[1,2,5]}
grid = {'nu':[0.1, 0.2, 0.3], 'kernel':['linear', 'poly', 'rbf'], 'gamma':['scale', 'auto']}
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.5,stratify=y, random_state=0)
X_train, y_train = sm.fit_resample(X_train, y_train)
clf = GridSearchCV(NuSVC(probability=True), grid,cv=5, scoring="recall_macro")
clf.fit(X_train, y_train)
forest = GridSearchCV(ExtraTreesClassifier(random_state=5), forest_grid,cv=5, scoring="recall_macro")
forest.fit(X_train,y_train)
model = clf.best_estimator_
forest = forest.best_estimator_
pickle.dump(normalizer, open('normalizer.sav', 'wb'))
#pickle.dump(model,open('model.sav','wb'))
balanced = permutation_importance(clf, X_test, y_test, random_state=42).importances_mean
balanced_tuple = [(feature,balanced_item,'balanced') for feature,balanced_item in zip(df.columns[:-1],balanced)]
data = unbalanced_tuple+balanced_tuple
#############
balanced_forest = permutation_importance(forest, X_test, y_test, random_state=42).importances_mean
balanced_tuple_forest = [(feature,balanced_item,'balanced') for feature,balanced_item in zip(df.columns[:-1],balanced_forest)]
data_forest= unbalanced_tuple_forest+balanced_tuple_forest
forest_dataframe =pd.DataFrame(data_forest,columns =['features', 'values', 'dataset'])
plt.figure(figsize=(15, 10), dpi=68)
plt.title('Random Forest SMOTE (k=8)')
sn.barplot(x='features',y='values', hue='dataset',data=forest_dataframe, palette='crest')
plt.show()
print('Random Forest SMOTE metrics')
y_true, y_pred = y_test, forest.predict(X_test)
report =classification_report(y_true, y_pred, output_dict=True)
tn, fp, fn, tp= confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tn+tp)/(tn+tp+fn+fp)
auc = roc_auc_score(y_true, y_pred)
print(f'auc: {auc}')
print(f'accuracy: {accuracy}    sensitivity: {sensitivity}    specificity: {specificity}')
report = pd.DataFrame(report).transpose()
report.to_csv('report_smote_forest.csv')
print(report)
############
print(data)
bar_dataframe = pd.DataFrame(data,columns =['features', 'values', 'dataset'])
plt.figure(figsize=(15, 10), dpi=68)
plt.title('Nu-SVM SMOTE(k=8)')
sn.barplot(x='features',y='values', hue='dataset',data=bar_dataframe, palette='crest')
plt.show()
print('Nu-SVM')
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
report =classification_report(y_true, y_pred, output_dict=True)
tn, fp, fn, tp= confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tn+tp)/(tn+tp+fn+fp)
auc = roc_auc_score(y_true, y_pred)
print(f'auc: {auc}')
print(f'accuracy: {accuracy}    sensitivity: {sensitivity}    specificity: {specificity}')
report = pd.DataFrame(report).transpose()
report.to_csv('report_smote.csv')
print(report)
print()
print('Calibration Analysis')
print()
nu_svc = model
nu_svc_isotonic = CalibratedClassifierCV(nu_svc,cv=5,method='isotonic')
nu_svc_sigmoid = CalibratedClassifierCV(nu_svc,cv=5,method='sigmoid')
clf_list =[(nu_svc,'svc'),(nu_svc_isotonic,'isotonic svc'),(nu_svc_sigmoid,'sigmoid svc'),(forest,'forest')]
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=5,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display
#pickle.dump(nu_svc_isotonic,open('model.sav','wb'))
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (SVC) SMOTE k=8")
# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0),(3,1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()


from collections import defaultdict
from sklearn.metrics import (
    brier_score_loss,
    log_loss
)

scores = defaultdict(list)
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores["Classifier"].append(name)

    for metric in [brier_score_loss, log_loss]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))
score_df = pd.DataFrame(scores).set_index("Classifier")
score_df.round(decimals=3)
print(score_df)
# %%
#ExtraTrees
import graphviz
xtr = forest
dot_data = tree.export_graphviz(forest.estimators_[8], out_file=None, 
                   feature_names=df.columns[:-1],  
                    class_names=['healthy','diabetic'],  
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data)
graph
xtr_isotonic = CalibratedClassifierCV(xtr,cv=5,method='isotonic')
xtr_sigmoid = CalibratedClassifierCV(xtr,cv=5,method='sigmoid')
clf_list =[(xtr,'ExtraTrees'),(xtr_isotonic,'isotonic ExtraTrees'),(xtr_sigmoid,'sigmoid ExtraTrees')]
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=5,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display
pickle.dump(xtr,open('model.sav','wb'))
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots (ExtraTrees) SMOTE (k=8)")
# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()


from collections import defaultdict
from sklearn.metrics import (
    brier_score_loss,
    log_loss
)

scores = defaultdict(list)
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    scores["Classifier"].append(name)

    for metric in [brier_score_loss, log_loss]:
        score_name = metric.__name__.replace("_", " ").replace("score", "").capitalize()
        scores[score_name].append(metric(y_test, y_prob[:, 1]))
score_df = pd.DataFrame(scores).set_index("Classifier")
score_df.round(decimals=3)
print(score_df)
fit = check_is_fitted(estimator=xtr,msg='not fitted')
check_smote = SMOTE(random_state=42,k_neighbors=7)
X_check, y_check=check_smote.fit_resample(X_test, y_test)
y_pred=xtr.predict_proba(X_check)[:,1]
fpr, tpr, threshold=roc_curve(y_check, y_pred)
scores = [(1-f_p_r,t_p_r,thr) for f_p_r, t_p_r ,thr in zip(fpr,tpr,threshold)]
print(max(scores,key= lambda score:score[0]+score[1]))
plt.plot(fpr, tpr)
plt.show()


# %%
