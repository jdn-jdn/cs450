import sys
import numpy as np
from operator import itemgetter
from collections import Counter
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # for comparisons


class kNNClassifier:
    """
    kNN only remembers data, so building model through
    training unecessary: data is only stored.
    """

    def __init__(self, k = 3):
        """
        Constructor to set value of k, which defaults to 3.
        :param k: number of neighbors
        """

        self.k = k

    def fit(self, data_train, targets_train):
        """
        kNN only remembers data, thus building a model through
        training unecessary. Data is saved, by passing to a kNNModel object,
        which is returned.
        :param data_train: data used to build model and calculate nearest neighbors and most common class
        :param targets_train: corresponding target values used to build model
        :return: kNNModel object, which stores passed in training data and targets values
        """

        return kNNModel(self.k, data_train, targets_train)


class kNNModel:
    """
    kNN model given training data and targets, along with value for k, which are
    used to predict target values.
    """

    def __init__(self, k, data_train, targets_train):
        """
        Constructor to set training data anmd targets, as well as the value of k, which defaults to 3,
        via the kNNClassifier object instantiating this class.
        :param k: number of neighbors
        :param data_train: data used to build model and calculate nearest neighbors and most common class
        :param targets_train: corresponding classes used to build model
        """

        self.k = k
        self.data_train = data_train
        self.targets_train = targets_train

    def predict(self, data_test):
        """
        Returns list of predicted values. Loops through each instance found in passed-in
        test data. Member method predict_instance() implements most of kNN algorithm.
        :param data_test: data used to predict targets
        :return: list of predicted targets generated with member variables storing training data and targets;
        test data passed in as parameter
        """

        # loop through each instance in test data passed in and make predictions
        targets_predicted = list()
        for i in range( data_test.shape[0] ):
            # predict target value for current instance
            targets_predicted.append(
                self.predict_instance(self.k, self.data_train, self.targets_train, data_test[i, : ]))
        return targets_predicted

    def predict_instance(self, k, data_train, targets_train, data_test):
        """
        The implementation of k Nearest Neighbors algorithm.
        :param k: number of neighbors
        :param data_train: data used to build model and calculate nearest neighbors and most common class
        :param targets_train: corresponding target values; used to extract predicted targets
        :param data_test: data used to predict targets
        :return: most common class found in k-sized list of tuples holding data on
         smallest distances and corresponding target indexes (i.e. k nearest neighbors)
        """

        # calculate distances and store corresponding indexes for each instance in data.
        # distances_unsorted and distances_sorted hold tuples, having distance value at index 0 and
        # corresponding index value at index 1
        distances_unsorted = self.get_distances(data_train, data_test)

        # sort list of tuples based on distances (first item) in ascending order (smallest to largest)
        distances_sorted = sorted(distances_unsorted, key = itemgetter(0))

        # find k nearest neighbors; distances_sorted holds tuples, storing distances and indexes, so retrieve
        # corresponding target_train value for each distance value using the distance value's corresponding index
        # found above (second item in tuple)
        targets_predicted = list()
        for i in range(k):
            targets_predicted.append(targets_train[ distances_sorted[i][1] ])

        # find and return most common target class value
        # Counter()'s most_common method handles non-unique modes to break possible ties
        most_common_class = Counter(targets_predicted).most_common(1)[0][0]
        return most_common_class

    def get_distances(self, data_train, data_test):
        """
        Calculates distances and stores corresponding indexes for each instance of data.
        :return: list of tuples stroing distances and corresponding indexes
        """

        # distances list holds tuples, having distance value at index 0 and corresponding index value at index 1
        distances = list()
        for i in range( data_train.shape[0] ):
            # square root not necessary, so sum of squared differences calculated and stored in list of tuples,
            # along with corresponding indexes; indexes are used later to retrieve corresponding target class values
            # in targets_train
            distance = np.sum( (data_test - data_train[i, : ]) ** 2 )
            distances.append( (distance, i) )
        return distances


class RandomDatasetGenerator:
    """
    Randomly generated dataset made with sklearn.dataset's make_cassification method. Key parameters may
    be specified, otherwise default variable values used by method.
    """

    def __init__(self, n_samples, n_features, n_classes):
        """
        data, target, and DESCR member variables are defined using sklearn module and given parameters
        :param n_samples: number of samples
        :param n_features: number of features
        :param n_classes: number of classes (or labels)
        """
        random_dataset = \
            datasets.make_classification(n_samples = n_samples, n_features = n_features, n_classes = n_classes)
        self.data   = random_dataset[0]
        self.target = random_dataset[1]
        self.DESCR  = "Randomly Generated Dataset\n="


def load_datasets():
    """
    Iterates through various datasets provided by sklearn's datasets module, including one made with
    sklearn.datasets' artificial data generator (datasets.make_classification)
    :return: list of various datasets
    """

    datasets_list = list()

    datasets_list.append(datasets.load_iris())
    datasets_list.append(datasets.load_breast_cancer())
    datasets_list.append(datasets.load_digits())
    datasets_list.append(datasets.load_diabetes())
    datasets_list.append(RandomDatasetGenerator(100, 5, 1)) # see class definition for description of parameters

    return datasets_list


def main(argv):
    """
    Driver function for program.
    :param argv: system parameters
    :return: none
    """

    # load various datasets
    datasets_list = load_datasets()

    # run kNN algorithm on each dataset
    for current_dataset in datasets_list:

        print(("=" * 50) + "\nRunning kNN models on following dataset:\n")
        print("\t" + current_dataset.DESCR.split('=')[0])

        # split data and targets into training and testing sets
        data_train, data_test, targets_train, targets_test = train_test_split(current_dataset.data, current_dataset.target, test_size = 0.3)

        # set value for k
        k = 5

        # build model and save data
        classifier = kNNClassifier(k)
        model = classifier.fit(data_train, targets_train)

        # predict targets
        targets_predicted = model.predict(data_test)

        # assess accuracy of model by comparing model's predictions (targets_predicted) to
        # actual targets (targets_test)
        accuracy = accuracy_score(targets_test, targets_predicted)
        print("\t\t" + "Custom  kNN accuracy: %" + str(round(accuracy * 100, 2)))

        # build, predict, and compare results of sklearn implementation to custom implementation
        sklearn_classifier = KNeighborsClassifier(n_neighbors = k)
        sklearn_model = sklearn_classifier.fit(data_train, targets_train)

        sklearn_targets_predicted = sklearn_model.predict(data_test)

        sklearn_accuracy = accuracy_score(targets_test, sklearn_targets_predicted)
        print("\t\t" + "sklearn kNN accuracy: %" + str(round(sklearn_accuracy * 100, 2)))

        print("\nExiting...")


if __name__ == "__main__":
    main(sys.argv)