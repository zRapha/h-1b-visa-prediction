import sys
import matplotlib
import numpy as np
import pandas as pd
import visuals as vs
from time import time
import pickle # To save algorithm trained
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, classification_report

# Total processing time
time_all = time()

### Reading random number of rows (used by performing test with reduced versions of dataset)
n = 10  # every 10th line = 10% of the lines
h1b_final = pd.read_csv('~/h1b_kaggle.csv', skiprows=lambda i: i % n != 0, index_col=0)

# Loading data
#h1b_final = pd.read_csv('~/h1b_kaggle.csv', nrows=2999999, index_col=0)
print('\nLoading h1b_kaggle.csv file..\n')
print 'Applications processed:', len(h1b_final)

### Removing rejected, invalidated and pending review applications
h1b_final = h1b_final[h1b_final['CASE_STATUS'] != 'REJECTED']
h1b_final = h1b_final[h1b_final['CASE_STATUS'] != 'INVALIDATED']
h1b_final = h1b_final[h1b_final['CASE_STATUS'] != 'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED']

# Restart the index after deleting rows and removing outlier wages above $1 billion
h1b_final.reset_index(drop=True, inplace=True)
h1b_final = h1b_final[(h1b_final['PREVAILING_WAGE']<1000000)]

### Features & outcome
outcome_raw = h1b_final['CASE_STATUS']
features_raw = h1b_final.drop(['CASE_STATUS','EMPLOYER_NAME','JOB_TITLE','lon', 'lat', 'WORKSITE'], axis = 1)

### Some statistics on prevailing wages of candidates
print("\nStatistics for H1-B Visa Applications:\n")
print("Minimum wage: ${:,.2f}".format(min(features_raw['PREVAILING_WAGE'])))
print("Maximum wage: ${:,.2f}".format(max(features_raw['PREVAILING_WAGE'])))
print("Mean wage: ${:,.2f}".format(np.mean(features_raw['PREVAILING_WAGE'])))
print("Median wage ${:,.2f}".format(np.median(features_raw['PREVAILING_WAGE'])))
print("Standard deviation of wage: ${:,.2f}".format(np.std(features_raw['PREVAILING_WAGE'])))

### Prepare the data
# Normalize numerical features by initializing a scaler and applying it to the features
scaler = MinMaxScaler()
numerical = ['YEAR', 'PREVAILING_WAGE']
features_raw[numerical] = scaler.fit_transform(features_raw[numerical])

### Data Preprocessing
time0 = time()
outcome_raw = outcome_raw.apply(lambda x: 1 if x == 'CERTIFIED' else x)
outcome_raw = outcome_raw.apply(lambda x: 2 if x == 'CERTIFIED-WITHDRAWN' else x)
outcome_raw = outcome_raw.apply(lambda x: 3 if x == 'DENIED' else x)
outcome = outcome_raw.apply(lambda x: 4 if x == 'WITHDRAWN' else x)

features = pd.get_dummies(features_raw)
encoded = list(features.columns)
print("\n{} total features after one-hot encoding.".format(len(encoded)))

# Show processing time in h:m:s
m, s = divmod(time()-time0, 60)
h, m = divmod(m, 60)
print("\nTime elapsed: %d:%02d:%02d" % (h, m, s))

### Evaluating Model performance with naive predictor
import warnings
warnings.filterwarnings('ignore')

predictions_naive = pd.Series(np.ones(len(outcome), dtype = int))

# Naive performance
accuracy = accuracy_score(outcome, predictions_naive)
fscore = fbeta_score(outcome, predictions_naive, beta=0.5, average='weighted')
precision = precision_score(outcome, predictions_naive, average='weighted')

print 'Accuracy score:', accuracy
print 'f1 score (weighted):', fscore
print 'Precision (weighted):', precision
print 'Recall (weighted):', recall_score(outcome, predictions_naive, average='weighted')
print '\nClassification Report:\n\n', classification_report(outcome, predictions_naive)

### Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size=0.2, random_state=19)

print("\nTraining and testing split was successful:")
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.\n".format(X_test.shape[0]))

### Creating a Training and Predicting Pipeline
def train_predict(learner, X_train, y_train, X_test, y_test):

    results = {}

    # Fit the learner to the training data
    start = time()
    learner = learner.fit(X_train, y_train)
    end = time()

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test set
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time()  # Get end time

    # Calculate times & score
    results['pred_time'] = end - start
    results['precision_train'] = precision_score(y_train, predictions_train, average='weighted')
    results['precision_test'] = precision_score(y_test, predictions_test, average='weighted')
    results['f_train'] = fbeta_score(y_train, predictions_train, beta=0.5, average='weighted')
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5, average='weighted')

    # Print all
    print "Training {}".format(learner.__class__.__name__)
    print "Precision result {}".format(results['precision_test'], learner.__class__.__name__)
    print "F-score result {}".format(results['f_test'], learner.__class__.__name__)
    print "Recall result {}".format(recall_score(y_test, predictions_test, average='weighted'), learner.__class__.__name__)

    # Return the results
    return results

### Train the three supervised learning models
time1 = time()

clf_A = LogisticRegression()
clf_B = RandomForestClassifier(n_estimators=10)
clf_C = DecisionTreeClassifier(random_state=12)

results = {}

for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    results[clf_name] = train_predict(clf, X_train, y_train, X_test, y_test)

# Show processing time in h:m:s
m, s = divmod(time() - time1, 60)
h, m = divmod(m, 60)
print("\nTime elapsed to train classifier: %d:%02d:%02d" % (h, m, s))

### Plot training & prediction times and scores for all classifiers

k = 0 # Training & Prediction times
vs.evaluate(results, 0, k)
k = 1 # Precision score
vs.evaluate(results, precision, k)
k = 2 # F-Score
vs.evaluate(results, fscore, k)

### Model Tuning for Decision Tree or Logistic Regression classifiers
time2 = time()

clf = LogisticRegression()  #clf = DecisionTreeClassifier(random_state=27)

# Create the parameters list to tune
param_grid = {'C': [1, 10, 100, 1000]}
#parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
#              'max_depth': [1, 3, 5], 'max_leaf_nodes': [2, 5, 10, 15, 30]}

# Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5, average='weighted')

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, param_grid, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("\nUnoptimized model\n------")
print("Precision score on testing data: {:.4f}".format(precision_score(y_test, predictions, average='weighted')))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5, average='weighted')))
print("\nOptimized Model\n------")
print("Final precision score on the testing data: {:.4f}".format(precision_score(y_test, best_predictions, average='weighted')))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5, average='weighted')))

# Show processing time in h:m:s
m, s = divmod(time() - time2, 60)
h, m = divmod(m, 60)
print("\nTime elapsed to tune classifier: %d:%02d:%02d" % (h, m, s))

### Make predictions using the final models
predictions_LR = clf_A.predict(X_test)
predictions_RF = clf_B.predict(X_test)
predictions_DT = clf_C.predict(X_test)

# Report the scores for the models
print("Random Forest:\n------")
print("Precision score on testing data: {:.4f}".format(precision_score(y_test, predictions_RF, average='weighted')))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions_RF, beta = 0.5, average='weighted')))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, predictions_RF, average='weighted')))
print("\nDecision Tree:\n------")
print("Precision score on the testing data: {:.4f}".format(precision_score(y_test, predictions_DT, average='weighted')))
print("F-score on the testing data: {:.4f}".format(fbeta_score(y_test, predictions_DT, beta = 0.5, average='weighted')))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, predictions_DT, average='weighted')))
print("\nLogistic Regression:\n------")
print("Precision score on the testing data: {:.4f}".format(precision_score(y_test, predictions_LR, average='weighted')))
print("F-score on the testing data: {:.4f}".format(fbeta_score(y_test, predictions_LR, beta = 0.5, average='weighted')))
print("Recall on testing data: {:.4f}".format(recall_score(y_test, predictions_LR, average='weighted')))

### Identifying most relevant features by using a supervised model with 'feature_importances_'

# Extract feature importances using Decision Tree Classifier (clf_B for Random Forest)
importances = clf_C.feature_importances_

# Visualization
vs.feature_plot(importances, X_train, y_train)

### Training using only most relevant features for less prediction time but at the cost of performance metrics
# Import functionality for cloning a model
from sklearn.base import clone
time0 = time()

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the best model found from grid search earlier - (clf_B: Random Forest & clf_C: Decision Tree)
clf = (clone(best_clf)).fit(X_train_reduced, y_train)
#clf = (clone(clf_C)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("\nFinal Model trained on full data\n------")
print("Precision on testing data: {:.4f}".format(precision_score(y_test, predictions_DT, average='weighted')))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions_DT, beta = 0.5, average='weighted')))
print("\nFinal Model trained on reduced data\n------")
print("Precision on testing data: {:.4f}".format(precision_score(y_test, reduced_predictions, average='weighted')))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5, average='weighted')))

# Save model using pickle
h1b_pickle_file = '~/saved_models/h1b_pickle.pkl'
h1b_pickle = open(h1b_pickle_file, 'wb')
pickle.dump(clf_A, h1b_pickle)
h1b_pickle.close()

# Show processing time in h:m:s
m, s = divmod(time() - time_all, 60)
h, m = divmod(m, 60)
print("\nTotal execution time: %d:%02d:%02d" % (h, m, s))
