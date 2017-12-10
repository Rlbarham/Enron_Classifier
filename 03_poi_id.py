#!/usr/bin/python
from __future__ import division
import sys
import pickle


sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tester import dump_classifier_and_data
from collections import Counter


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary','total_payments', 'exercised_stock_options', 'bonus', 
'restricted_stock', 'shared_receipt_with_poi', 'total_stock_value', 'deferred_income', 
'long_term_incentive', 'to_poi_proportion']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)

### Task 3: Create new feature(s). Add to the above features_list if you do so
### Store to my_dataset for easy export below.

# Basic fraction computation function
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """    
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0
    else:
        fraction = poi_messages/all_messages
        return fraction
    
# Compute from_poi_proportion and to_poi_proportion using computeFraction function
submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]
    
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    from_poi_proportion = computeFraction( from_poi_to_this_person, to_messages )
    data_point["from_poi_proportion"] = from_poi_proportion

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    to_poi_proportion = computeFraction( from_this_person_to_poi, from_messages )
    data_point["to_poi_proportion"] = to_poi_proportion

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scale the features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline

# NB: Other algorithms were trialled, with much more sophisticated tuning of parameters
# This work can be viewed in my misc_analyses.ipynb file. An unmodified GaussianNB 
# ended up performing best on tester.py with the feature and data preparation I pursued,
# hence using it here! 

# Three algorithms selected, pre-tuned in my misc_analyses ipynb file
mnb_clf = MultinomialNB(alpha=1.5, fit_prior=False)
gnb_clf = GaussianNB()
lreg_clf = LogisticRegression(C=0.05, class_weight='auto', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', penalty='l2', random_state=None,
          solver='liblinear', tol=0.1, verbose=0)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# NB: The tuning for these took place in the ipynb file, as did validation; here I am however showcasing some of 
# the basic, pre-validation differences in performance. Multinomial NB did turn out to be highest-performing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

def classifier_evaluation(truth, prediction):
    confusion_matrix = Counter()
    positives = [1]
    
    truth_split = [i in positives for i in truth]
    prediction_split = [i in positives for i in prediction]
    for x, y in zip(truth_split, prediction_split):
        confusion_matrix[x,y] += 1
   
    print confusion_matrix
    print "Precision Score: ", precision_score(truth, prediction)
    print "Recall Score: ", recall_score(truth, prediction)
    print "F1 Score: ", f1_score(truth, prediction)

mnb_clf.fit(features_train, labels_train)
labels_pred = mnb_clf.predict(features_test)
print "Provisional results: MultinomalNB: "
print classifier_evaluation(labels_test, labels_pred)

gnb_clf.fit(features_train, labels_train)
labels_pred = gnb_clf.predict(features_test)
print "Provisional results: GaussianNB: "
print classifier_evaluation(labels_test, labels_pred)

lreg_clf.fit(features_train, labels_train)
labels_pred = lreg_clf.predict(features_test)
print "Provisional results: Logistic Regression: "
print classifier_evaluation(labels_test, labels_pred)


# Based on the above and validation in misc_analyses, prioritized MultiNomialNB; used Pipeline here to export scaling with it
steps = [
    ('scaler', MinMaxScaler()),
    ('classifier', MultinomialNB(alpha=1.5, fit_prior=False))
]

clf = Pipeline(steps)
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)