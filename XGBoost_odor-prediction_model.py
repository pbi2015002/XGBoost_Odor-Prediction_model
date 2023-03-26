
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import keras
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from scipy import interp
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from xgboost import cv
import xgboost as xgb
import warnings
import pickle

warnings.filterwarnings("ignore")

plt.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 0.5

dataset = pd.read_csv('Norm_Moldessss.csv')

X, y = dataset.iloc[:, : -1], dataset.iloc[:, -1]

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.20, random_state = 1)

#Apply PCA on trainning data
sc = StandardScaler()
X_scaled = sc.fit_transform(X_train1)

components = 0.999
pca = PCA(n_components = components)
pca.fit(X_scaled)

print("Cumulative Variances (Percentage):")
print(np.cumsum(pca.explained_variance_ratio_ * 100))
components = len(pca.explained_variance_ratio_)
print(f'Number of components: {components}')
# Make the scree plot
plt.plot(range(1, components + 1), 
np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")
plt.tight_layout()
plt.show()

pca_components = abs(pca.components_)
print(pca_components)

X1 = pca.transform(X_scaled)
print(X1.shape)
print(X1)

#Apply PCA on testing data
X_test = pca.transform(X_test1)
print(X_test.shape)
print(X_test)

strategy = {0:420, 1:420, 2:420, 3:420, 4:420, 5:420, 6:420}
sampling_strategy=strategy
oversample = SMOTE(sampling_strategy)
X_train, y1 = oversample.fit_resample(X1, y_train1)

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y1)
y_train = label_encoder.transform(y1)
y_test = label_encoder.transform(y_test1)

n_classes = len(np.unique(y_test))
print("number of classes = ", n_classes)

#this piece of code in triple double quotes to tune hyperparameters
"""
estimator = XGBClassifier(
    objective= 'multi:softprob',
    seed=42,
    use_label_encoder =False
)

parameters = {
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [250, 400, 600, 800, 1000],
    'learning_rate': [0.01, 0.05, 0.09, 0.1, 0.5],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1],
    'min_child_weight': [1, 2, 3, 4, 5],
    'subsample': [0.5, 0.6, 0.7, 0.8],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]	
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 10,
    refit = False,
    verbose= 10
)

grid_search.fit(X_train, y_train)

print(grid_search.best_estimator_)
print(grid_search.best_params_)
"""
#"""
model = XGBClassifier(objective = "multi:softprob", subsample = 0.8, colsample_bytree = 0.9, 
    min_child_weight=1, learning_rate =0.05, max_depth=4, gamma =0.1, n_estimators=1000)

#further evaluate model using 10-fold cross validation
#kfold = KFold(n_splits=10)
#results = cross_val_score(model, X_train, y_train, cv=kfold)
#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=1)
print(model)

filename = 'XGBoost_model.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

y_pred = loaded_model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, y_pred) 
print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
#print(predictions)

results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

fig, ax = plt.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

fig, ax = plt.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4, 5, 6])
print(cm)

target_names = ['sweet', 'nutty', 'pungent', 'floral', 'fruity', 'minty', 'woody']
print(classification_report(y_test, predictions, target_names=target_names))
report = classification_report(y_test, predictions, target_names=target_names)

fname = 'XGBoost_classification_report.csv'
with open(fname, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(report)
    writer.writerow(cm)

y_test = label_binarize(y_test, classes=np.arange(n_classes))
y_pred = label_binarize(y_pred, classes=np.arange(n_classes))

print("shape of the test array=", np.shape(y_test))
print("shape of the predicted array=", np.shape(y_pred))

lw = 0.5

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=1)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=1)

colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'pink'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

plt.figure(2)
plt.xlim(-0.05, 0.2)
plt.ylim(0.8, 1.05)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=1)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=1)

colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'pink'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])                                                    
    average_precision[i] = average_precision_score(y_test[:, i], y_pred[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())  
average_precision["micro"] = average_precision_score(y_test, y_pred, average="micro")
                                                     
plt.plot(recall[0], precision[0], label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.title('Average Precision-Recall curve: AUC={0:0.2f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

plt.plot(recall["micro"], precision["micro"],
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i in range(n_classes):
    plt.plot(recall[i], precision[i],
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for all classes')
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
#"""



