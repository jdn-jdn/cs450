import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.cross_validation import KFold, cross_val_score
# from sklearn.model_selection import KFold, cross_val_score

class HardCodedModel():
    def __init__(self):
        pass

    def predict(self, data):
        targets = []
        for instance in data:
            targets.append(self.predict_one(instance))
        return targets

    def predict_one(self, instance):
        return 0

class HardCodedClassifier():
    def __init__(self):
        pass

    def fit(self, data, targets):
        return HardCodedModel()

def main(argv):
    '''
    1. Load data
    '''
    iris = datasets.load_iris()

    # Show the data (the attributes of each instance)
    print('The data (attributes of each instance):\n',
          iris.data)
    print()

    # Show the target values (in numeric format) of each instance
    print('The target values (in numeric format) of each instance:\n', iris.target)
    print()

    # Show the actual target names that correspond to each number
    print('The actual target names that correspond to each number:\n', iris.target_names)
    print()

    '''
    2. Prepare training/test sets
    '''
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target,
                                                                          test_size=0.3)  # , random_state=42)

    print('The training set for data:\n', data_train)
    print()

    print('The testing set for data:\n', data_test)
    print()

    print('The training set for targets:\n', targets_train)
    print()

    print('The testing set for targets:\n', targets_test)
    print()

    '''
    3. Use an existing algorithm to create a model
    '''
    classifier = GaussianNB()
    model = classifier.fit(data_train, targets_train)

    '''
    ABOVE AND BEYOND: Adding n-fold cross validation
    '''
    n_folds = 10
    k_fold = KFold(len(iris.target), n_folds, shuffle = False, random_state = None)
    print('n-fold cross validation (n = %d):' % n_folds)

    scores = cross_val_score(classifier, iris.data, iris.target, cv = k_fold, n_jobs = 5)
    print('Scores:\n', scores)
    print()

    print("Mean accuracy score and %95 confidence interval of accuracy " +
          "score estimate:\n %%%.2f (+/- %%%0.2f)" % (scores.mean() * 100, scores.std() * 2 * 100))
    print()

    '''
    4. Use that model to make predictions
    '''
    targets_predicted = model.predict(data_test)

    # Compare the predicted targets to the actual targets
    print('Predicted targets:\n', targets_predicted)
    print()

    print('Actual targets:\n', targets_test)
    print()

    # Output the resulting accuracy
    accuracy_score = metrics.accuracy_score(targets_predicted, targets_test)
    accuracy_score_percentage = '{0:.2f}'.format((accuracy_score * 100))
    print('Accuracy classification score:\n', '%' + accuracy_score_percentage)
    print()

    '''
    5. Implement your own new "algorithm"
    '''
    # Instantiate new classifier
    classifier = HardCodedClassifier()

    # "Train" it with training data
    model = classifier.fit(data_train, targets_train)

    # Use it to make predictions on test data
    targets_predicted = model.predict(data_test)
    print('Predicted targets (HardCodedClassifier):\n', targets_predicted)
    print()

    # Determine accuracy of classifier's predictions and report the result as a percentage
    accuracy_score = metrics.accuracy_score(targets_predicted, targets_test)
    accuracy_score_percentage = '{0:.2f}'.format((accuracy_score * 100))
    print('Accuracy classification score (HardCodedClassifier):\n', '%' + accuracy_score_percentage)
    print()

if __name__ == "__main__":
    main(sys.argv)