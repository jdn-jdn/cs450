import sys
import pandas as pd
import numpy as np
from collections import Counter
from operator import itemgetter
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


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

    def predict(self, data_test, dataset_option):
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
                self.predict_instance(self.k, self.data_train, self.targets_train, data_test[i, : ], dataset_option))
        return targets_predicted

    def predict_instance(self, k, data_train, targets_train, data_test, dataset_option):
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

        if dataset_option == "car_eval" or dataset_option == "diabetes":
            # find and return most common target class value
            # Counter()'s most_common method handles non-unique modes to break possible ties
            most_common_class = Counter(targets_predicted).most_common(1)[0][0]
            return most_common_class
        elif dataset_option == "auto_mpg":
            # find and return the average of the k nearest neighbors' values
            average_value = np.mean(targets_predicted)
            return average_value


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


def read_data(dataset_option):
    if dataset_option is "car_eval":
        """
        Returns a pandas Dataframe,
        - Number of Instances: 1728
        - Number of Attributes: 6  
        - Missing Attribute Values: none           
        - Attribute Values:
            1. buying       v-high, high, med, low
            2. maint        v-high, high, med, low
            3. doors        2, 3, 4, 5-more
            4. persons      2, 4, more
            5. lug_boot     small, med, big
            6. safety       low, med, high 
        - Class Values:  
            unacc, acc, good, vgood       
        """
        headers_car_eval = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class_evaluation"]
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
                    header = None, names = headers_car_eval)
    elif dataset_option is "diabetes":
        """
        - Number of Instances: 768
        - Number of Attributes: 8  
        - Missing Attribute Values: Yes           
        - For Each Attribute: (all numeric-valued)
            1. Number of times pregnant
            2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
            3. Diastolic blood pressure (mm Hg)
            4. Triceps skin fold thickness (mm)
            5. 2-Hour serum insulin (mu U/ml)
            6. Body mass index (weight in kg/(height in m)^2)
            7. Diabetes pedigree function
            8. Age (years)
        - Class Distribution: 
            Note: class value 1 is interpreted as "tested positive for diabetes".
            Class Value     Number of instances
            0               500
            1               268     
        """
        headers_diabetes = ["num_times_preg", "plasma_gluc_concent", "diastolic_blood_press",
                            "triceps_skin_fold_thickness", "2_hr_serum_insul", "body_mass_index",
                            "diab_pedogree_func", "age", "class_positive_diabetes"]
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
                    header = None, names = headers_diabetes)

        # All zero values for the biological variables other than number of times pregnant should be treated as missing
        # values. Source: https://www.kaggle.com/hinchou/case-study-on-pima-indian-diabetes
        df_X_nan = df[["plasma_gluc_concent", "diastolic_blood_press", "triceps_skin_fold_thickness", "2_hr_serum_insul",
                    "body_mass_index"]].replace(0, np.nan)

        return (df[["num_times_preg"]].join(df_X_nan)).join(df[["diab_pedogree_func", "age", "class_positive_diabetes"]])
    elif dataset_option is "auto_mpg":
        """
        - Number of Instances: 398
        - Number of Attributes: 8 
        - Missing Attribute Values: Yes - horsepower has 6 missing values          
        - Attribute Information
            1. mpg:           continuous
            2. cylinders:     multi-valued discrete
            3. displacement:  continuous
            4. horsepower:    continuous
            5. weight:        continuous
            6. acceleration:  continuous
            7. model year:    multi-valued discrete
            8. origin:        multi-valued discrete
        - Class Values: 
            9. car name:      string (unique for each instance) 
        """
        headers_auto_mpg = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year",
                            "origin", "class_car_name"]
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                    header = None, names = headers_auto_mpg, delim_whitespace = True, na_values = "?")


def handle_non_numeric_data(df, dataset_option):
    """
    :param df:
    :param dataset_option:
    :return:
    """
    if dataset_option is "car_eval":
        """
        - Attribute Values:
            1. buying       v-high, high, med, low
            2. maint        v-high, high, med, low
            3. doors        2, 3, 4, 5-more
            4. persons      2, 4, more
            5. lug_boot     small, med, big
            6. safety       low, med, high 
        - Class Values:  
            unacc, acc, good, vgood  
        """
        df["buying_cat"]           = df.buying.map({"vhigh": 4, "high": 3, "med": 2, "low": 1})
        df["maint_cat"]            = df.maint.map( {"vhigh": 4, "high": 3, "med": 2, "low": 1})
        df["doors_cat"]            = df.doors.map(  {"2": 2, "3": 3, "4": 4, "5more": 5})
        df["persons_cat"]          = df.persons.map({"2": 2, "4": 4, "more": 6})
        df["lug_boot_cat"]         = df.lug_boot.map({"small": 1, "med": 2, "big":  3})
        df["safety_cat"]           = df.safety.map(  {"low":   1, "med": 2, "high": 3})
        df["class_evaluation_cat"] = df.class_evaluation.map({"unacc": 0, "acc": 1, "good": 2, "vgood": 3})

        return df.drop(["buying", "maint", "doors", "persons", "lug_boot", "safety", "class_evaluation"], axis = 1)
    elif dataset_option is "diabetes":
        # No categorical data to be converted to numerical.
        return df
    elif dataset_option is "auto_mpg":
        """
        - Missing Attribute Values: Yes - horsepower has 6 missing values          
        - Attribute Information
            1. mpg:           continuous
            2. cylinders:     multi-valued discrete
            3. displacement:  continuous
            4. horsepower:    continuous
            5. weight:        continuous
            6. acceleration:  continuous
            7. model year:    multi-valued discrete
            8. origin:        multi-valued discrete
        - Class Values: 
            9. car name:      string (unique for each instance) 
        """
        df.replace('\s+', '_', regex = True, inplace = True)
        return pd.get_dummies(df, ["car_name"])


def handle_missing_data(df, dataset_option):
    """
    Filling with median values seemed best. Car evaluation dataset contained no missing values.
    :param df:
    :param dataset_option:
    :return:
    """
    if dataset_option is "car_eval":
        # - Missing Attribute Values: none
        return df
    elif dataset_option is "diabetes":
        return df.fillna(df.median())
    elif dataset_option is "auto_mpg":
        df = df.fillna(df.median())
        return df


def convert_to_numpy_array(df, dataset_option):
    """
    :param df:
    :return:
    """
    if dataset_option is "car_eval":
        return df.drop("class_evaluation_cat",    axis = 1).values, df[["class_evaluation_cat"]].values
    elif dataset_option is "diabetes":
        return df.drop("class_positive_diabetes", axis = 1).values, df[["class_positive_diabetes"]].values
    elif dataset_option is "auto_mpg":
        return df.drop("mpg",                     axis = 1).values, df[["mpg"]].values


def cross_validate(clf, X, y, dataset_option, cv = 1):
    """
    Custom-built cross validation function. Off-the-shelf validation did not work well with
    given parameters for Auto MPG dataset.
    :param clf:
    :param X:
    :param y:
    :param cv:
    :return:
    """
    scores = []
    test_size = 1 / cv
    num_instances = X.shape[0]
    increment_i = int(num_instances * test_size)
    start_i = 0

    print("Conducting " + str(cv) + "-fold cross validation...\n")
    for i in range(cv):
        if dataset_option == "car_eval":
            print("Conducting test #" + str(i + 1) + " of " + str(cv) + "...")

        X_test = X[start_i:start_i + increment_i]
        y_test = y[start_i:start_i + increment_i]

        X_train = X[0:start_i]
        y_train = y[0:start_i]
        if start_i + increment_i <= num_instances:
            X_train = np.append(X_train, X[start_i + increment_i:], axis = 0)
            y_train = np.append(y_train, y[start_i + increment_i:], axis = 0)

        start_i += increment_i

        model = clf.fit(X_train, y_train.ravel())
        y_predicted = model.predict(X_test, dataset_option)

        if dataset_option == "car_eval" or dataset_option == "diabetes":
            accuracy = accuracy_score(y_test, y_predicted)
            scores.append(accuracy)
        elif dataset_option == "auto_mpg":
            accuracy = stats.pearsonr(y_test.flatten(), y_predicted)
            scores.append(accuracy[0]) # item at index 1 is p-value

    print("\nTests complete.\nScores:\n\t" + str(scores))
    return scores


def run_test(dataset_option):
    """
	The main portion of the programs. Calls functions to handle core requirements.
	"""

    # Standard Requirement 1: Read data from text files.
    df = read_data(dataset_option)

    # Standard Requirement 2: Appropriately handle nosn-numeric data.
    df = handle_non_numeric_data(df, dataset_option)

    # Standard Requirement 3: Appropriately handle missing data.
    df = handle_missing_data(df, dataset_option)

    # Standard Requirement 4: Use of k-Fold Cross Validation.
    X, y = convert_to_numpy_array(df, dataset_option)

    k_neigh = 5
    k_fold  = 5

    # custom kNN implementation
    clf = kNNClassifier(k_neigh)

    # get scores with custom kNN implementation and custom cross validation function
    scores = cross_validate(clf, X, y, dataset_option, cv = k_fold)

    # kNN regression computes Pearson correlation for Auto MPG dataset
    if dataset_option == "car_eval" or dataset_option == "diabetes":
        print("\n>>>\tAccuracy (with custom cross validation):\n>>>\t\t%" + str(round(np.mean(scores) * 100, 2)))
        print("\n", "=" * 50, "\n")
    if dataset_option == "auto_mpg":
        print("\n>>>\tAverage Pearson correlation coefficient (with custom cross validation):\n>>>\t\t" + str(
            round(np.mean(scores), 2)))
        print("\n", "=" * 50, "\n")

    # Standard Requirement 5: Basic experimentation on the provided datasets.

    # Standardizing the dataset and comparing results:
    X_std = preprocessing.StandardScaler().fit_transform(X, y)

    clf = kNNClassifier(k_neigh)

    scores = cross_validate(clf, X_std, y, dataset_option, cv = k_fold)

    if dataset_option == "car_eval" or dataset_option == "diabetes":
        print("\n>>>\tAccuracy (with custom cross validation\n>>>\t\tand standardized X):\n>>>\t\t%" + str(round(np.mean(scores) * 100, 2)))
        print("\n", "=" * 50, "\n")
    if dataset_option == "auto_mpg":
        print("\n>>>\tAverage Pearson correlation coefficient (with custom cross validation\n>>>\t\tand standardized X):\n>>>\t\t" + str(
            round(np.mean(scores), 2)))
        print("\n", "=" * 50, "\n")

    # Trying a different values of k_neigh adn k_fold and comparing results:
    k_neigh = 17
    k_fold  = 10
    clf = kNNClassifier(k_neigh)

    scores = cross_validate(clf, X_std, y, dataset_option, cv = k_fold)

    if dataset_option == "car_eval" or dataset_option == "diabetes":
        print("\n>>>\tAccuracy (with custom cross validation\n>>>\t\tand standardized X\n>>>\t\tand k_fold = " +
              str(k_fold) + " and k_neigh = " + str(k_neigh) + "):\n>>>\t\t%" + str(round(np.mean(scores) * 100, 2)))
        print("\n", "=" * 50, "\n")
    if dataset_option == "auto_mpg":
        print("\n>>>\tAverage Pearson correlation coefficient (with custom cross validation\n>>>\t\tand " +
              "standardized X\n>>>\t\tand k_fold = " + str(k_fold) + " and k_neigh = " + str(k_neigh) + "):\n>>>\t\t" + str(
            round(np.mean(scores), 2)))
        print("\n", "=" * 50, "\n")

    # Using off-the-shelf kNN implemetation and comparing results:
    clf_sklearn = KNeighborsClassifier(n_neighbors = k_neigh)
    scores = cross_val_score(clf_sklearn, X_std, y.ravel(), cv = k_fold)
    print("\n>>>\tAccuracy (with SKLEARN cross validation\n>>>\t\tand standardized X\n>>>\t\tand k_fold = " +
          str(k_fold) + " and k_neigh = " + str(k_neigh) + "):\n>>>\t\t%" + str(round(np.mean(scores) * 100, 2)))
    print("\n", "=" * 50, "\n")


def main(argv):
    """
    Driver function for program.
    :param argv:
    :return: none
    """

    menu = {}
    menu['1'] = "- Car Evaluation Dataset"
    menu['2'] = "- Pima Indian Diabetes Dataset"
    menu['3'] = "- Auto MPG Dataset"
    menu['4'] = "- Exit"
    while True:
        options = menu.keys()
        sorted(options)
        for entry in options:
            print(entry, menu[entry])

        selection = input("\nPlease select a dataset and hit Enter: ")
        if   selection == '1':
            print()
            run_test(dataset_option = "car_eval")
        elif selection == '2':
            print()
            run_test(dataset_option = "diabetes")
        elif selection == '3':
            print()
            run_test(dataset_option = "auto_mpg")
        elif selection == '4':
            print()
            break
        else:
            print()
            print("Please enter valid input.")


if __name__ == "__main__":
    main(sys.argv)