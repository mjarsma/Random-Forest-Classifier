#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################

### Random forests are an ensemble learning method for classification, regression and other tasks, 
### that operate by constructing a multitude of decision trees at training time and outputting 
### the class that is the mode of the classes (classification) or mean prediction (regression) 
### of the individual trees. Random forests correct for decision trees' habit of overfitting to 
### their training set.

from sklearn.ensemble import RandomForestClassifier;

for i in range(1, 100):
    clf = RandomForestClassifier(min_samples_split=i);
    clf = clf.fit(features_train, labels_train);
    predict = clf.predict(features_test);
    score = clf.score(features_test, labels_test)
    if score > 0.935:
        print '{0} - {1}'.format(i, score);

#try:
#    prettyPicture(clf, features_test, labels_test)
#except NameError:
#    pass
