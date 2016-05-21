import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot
import math
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data


######################################
### Select starting features

# let features_list equal list of all possible features as starting point
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
	'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
	'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
	'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 
	'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
	'shared_receipt_with_poi', 'percent_emails_to_poi', 'percent_emails_from_poi']

# load the dictionary containing the dataset
with open("enron_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


######################################
### Remove outliers

# function to print rows with largest errors (>= 3 std) for each feature
def print_outliers(df):
	for column in df:
		temp = df[column].apply(lambda x: np.abs(x - df[column].mean()) / \
			   df[column].std()).sort_values(ascending=False).head(10)
		# print temp[temp >= 3]  # disabled


# convert data_dict to pandas df to get sense of data, detect outliers
df = pd.DataFrame.from_dict(data_dict, orient='index').\
	convert_objects(convert_numeric=True)

# get basic details of dataset
print "rows:", len(df)
print "features:", len(df.columns) - 1
print "data points:", df.count(numeric_only=False).sum()
print "data points poi:", df[df.poi==1].count(numeric_only=False).sum()
print "data points non-poi:", df[df.poi==0].count(numeric_only=False).sum()
print "data points individual features:"
print df.count(numeric_only=False)
print "missing values individual features:"
print len(df) - df.count(numeric_only=False)

# drop labels and text feature to allow for outlier detection
df = df.drop(['poi', 'email_address'], 1)

# print largest errors for each feature to allow visual inspection of outliers
print_outliers(df)  # disabled

### Process outliers
# TOTAL row is huge outlier, remove it
data_dict.pop('TOTAL', 0)

# In 'expenses' feature, 'KAMINSKI WINCENTY J' is big outlier. In 
# 'long_term_incentive' feature, 'LAVORATO JOHN J' is big outlier. 
# After inspection, I decided to keep both of these individuals in the dataset 
# becasue it is likely that their high values tell us something important about 
# them and their relationship to persons of interest.


######################################
### Engineer new features

# create new features:
# 1) percentage of emails sent that were sent to poi; and
# 2) percentage of emails received that were sent by poi
for key in data_dict:
	if data_dict[key]['from_messages'] == 'NaN':
		data_dict[key].update({'percent_emails_to_poi': 0,
			'percent_emails_from_poi': 0})
	else:
		data_dict[key].update({'percent_emails_to_poi': \
			100 * (float(data_dict[key]['from_this_person_to_poi']) /
			float(data_dict[key]['from_messages'])),
			'percent_emails_from_poi': \
			100 * (float(data_dict[key]['from_poi_to_this_person']) /
	 		float(data_dict[key]['to_messages']))})


### extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


######################################
### Select classifier

# create pipeline with SelectKBest selector and AdaBoost Classifier
estimators = [('kbest', SelectKBest(f_classif)), ('abc', AdaBoostClassifier(DecisionTreeClassifier()))]
clf = Pipeline(estimators)

######################################
### Tune classifier

### grid search to find optimal hyper-parameter levels

# arrange parameter grid
parameters = {'kbest__k': [3,4,5,6,7,8,9,10],
			  'abc__n_estimators': [40, 50, 60, 80, 100, 150, 200, 300],
 			  'abc__learning_rate': [0.5, 0.75, 1.0]}

# create cross-validation object to pass to GridSearchCV
cv = StratifiedShuffleSplit(labels, 100, random_state=42)

# execute grid search, using recall score to lead the parameter search process
grid_search = GridSearchCV(clf, param_grid=parameters, cv=cv, scoring="recall")
grid_search.fit(features, labels)

# get and print optimal paramaters
kbest_k = grid_search.best_params_['kbest__k']
n_est = grid_search.best_params_['abc__n_estimators']
l_rate = grid_search.best_params_['abc__learning_rate']
print " optimal kbest_k:", kbest_k
print "optimal n_est:", n_est
print "optimal l_rate:", l_rate

# re-initialize clf with optimal parameters
clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=n_est, 
	learning_rate=l_rate)

# split data into training and test groups using StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 100, random_state=42)
for train_indicies, test_indicies in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_indicies:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_indicies:
        features_test.append(features[jj])
        labels_test.append(labels[jj])

# select k best features
selector = SelectKBest(f_classif, k=kbest_k).fit(features_train, labels_train)
features_train_transformed = selector.transform(features_train)
features_test_transformed = selector.transform(features_test)

# get list of features ranked by importance
features = features_list[1:]
feature_scores = []
for i in range(0, len(features)):
	feature_scores.append({'feature': features[i],
		                   'score': selector.scores_[i],
		                   'selected': selector.get_support()[i]}) 
feature_scores = pd.DataFrame(feature_scores).sort_values(by='score', 
	ascending=False)
print feature_scores

# reduce features_list to most important features identified above
features_list_transformed = feature_scores.feature[feature_scores.selected == \
	True].tolist()

# fit model
clf.fit(features_train_transformed, labels_train)
pred = clf.predict(features_test_transformed)
print "accuracy:", accuracy_score(pred, labels_test)
print "precision:", precision_score(pred, labels_test)
print "recall:", recall_score(pred, labels_test)
print "f1 score:", f1_score(pred, labels_test)

# over-write features_list with list of best features
features_list = ['poi'] + features_list_transformed