import sys
import pandas as pd
import numpy as np
import pydot as pydot

from pptree import *
from anytree import Node, RenderTree, AnyNode
from scipy import stats
from sklearn import preprocessing
from sklearn import tree as sklearn_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sympy.physics.units import frequency
from anytree.exporter import DotExporter


class ID3DecisionTreeClassifier:
    """
        The ID3 Decision Tree algorithm classifier. Returns a model to build tree and predict targets.
    """

    def __init__(self):
        """
        This class only returns an ID3DecisionTreeModel object. No need for initialization of member variables.
        """
        pass

    def fit(self, data, classes, featureNames):
        """
        :param data: the given features for the dataset
        :param classes: the target values for the data
        :param featureNames: the names of the attributes in the data. Used for building tree.
        :return: a mondel object to build tree and predict targets
        """
        return ID3DecisionTreeModel(data, classes, featureNames)


class ID3DecisionTreeModel:
    """
        The ID3 Decision Tree algorithm model. Takes parameters given by classifier object to build actual tree
         and predict targets.
    """

    def __init__(self, data, classes, featureNames):
        """
        :param data: the given features for the dataset
        :param classes: the target values for the data
        :param featureNames: the names of the attributes in the data. Used for building tree.
        """
        self.data = data
        self.classes = classes
        self.featureNames = featureNames
        self.decision_tree = self.build_tree(data, classes, featureNames)

    def calc_entropy(self, p):
        """
        A method for computing entropy. The best feature to pick as the one to classify on is the one that yields the
        most information gain, i.e., the one with the lowest entropy.
        :param p: proportion of given attribute
        :return: calculated entropy
        """
        if p != 0:
            return -p * np.log2(p)
        else:
            return 0

    def calc_total_entropy(self, classes):
        """
        :param classes: target values corresponding to a set to calculate entropy. This method helps give the overall
        information gain to help make a decision, instead of using only the lowest entropy.
        :return: total entropy calculated
        """
        total_entropy = 0
        total_classes = classes.shape[0]
        unique_classes = np.unique(classes)

        # Determine the unique classes and initialize count to 0
        class_counts = {}
        for unique_class in unique_classes:
            class_counts[unique_class] = 0

        # Count occurences of class in classes parameter
        for unique_class in unique_classes:
            for aclass in classes:
                if unique_class == aclass:
                    class_counts[unique_class] += 1

        # Calculate total entropy for classes
        for aclass in class_counts:
            total_entropy += self.calc_entropy(float(class_counts[aclass]) / total_classes)

        return total_entropy

    def calc_info_gain(self, data, classes, feature):
        """
        Calculates information gain from adding given feature using entropy of S minus the sum of the feature entropy.
        :param data: the relevant data of the subset
        :param classes: the corresponding classes to the data
        :param feature: the feature to calculate gain for
        :return: total calculated information gain for feature in subset
        """
        gain = 0
        nData = len(data)

        # List the values that feature can take
        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0

        # Find where those values appear in data[feature] and the corresponding class
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(classes[dataIndex])
                dataIndex += 1

            # Get the values in newClasses
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass) == 0:
                    classValues.append(aclass)
            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex] += 1
                classIndex += 1

            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex]) / sum(classCounts))
            gain += float(featureCounts[valueIndex]) / nData * entropy[valueIndex]
            valueIndex += 1
        return gain

    def build_tree(self, data, classes, featureNames):
        """
        Implementation of the ID3 decision tree. Makes use of the calc_entropy(), calc_total_entropy(), and
        calc_info_gain() functions above.
        :param data: relevant data to partition and build nodes from. build_tree() called recursively, so data given
        changes.
        :param classes: the corresponding target values of the data parameter.
        :param featureNames: the names of the data attributes. Needed in node objects to help ask questions to targets
        predicted later.
        :return:
        """
        nData = data.shape[0]
        nFeatures = data.shape[1]

        default = classes[np.argmax(frequency)]
        if nData == 0 or nFeatures == 0:
            # Have reached an empty branch
            return AnyNode(classification = default[0])
            # return default
        # elif classes.count(classes[0]) == nData:
        elif classes.tolist().count(classes[0]) == nData:
            # Only 1 class remains
            return AnyNode(classification = classes[0][0])
            # return classes[0]
        else:
            # Choose which feature is best
            gain = np.zeros(nFeatures)
            for feature in range(nFeatures):
                g = self.calc_info_gain(data, classes, feature)
                totalEntropy = self.calc_total_entropy(classes)
                gain[feature] = totalEntropy - g
            bestFeature = np.argmax(gain)
            tree = AnyNode(feature = featureNames[bestFeature])
            # tree = {featureNames[bestFeature]: {}}

            # Find the possible feature values
            possible_values = list(np.unique(data[ : , bestFeature]))
            for value in possible_values:
                newData = []
                newClasses = []
                index = 0

                # Find the datapoints with each feature value
                for datapoint in data:
                    if datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            datapoint = datapoint[1: ]
                            newNames = featureNames[1: ]
                        elif bestFeature == nFeatures - 1:
                            datapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            datapoint = datapoint[:bestFeature]
                            # datapoint.extend(datapoint[bestFeature + 1:])
                            np.concatenate((datapoint, datapoint[bestFeature + 1:]))
                            newNames = featureNames[:bestFeature]
                            # newNames.extend(featureNames[bestFeature + 1:])
                            np.concatenate((newNames, featureNames[bestFeature + 1:]))
                        newData.append(datapoint)
                        newClasses.append(classes[index])
                    index += 1

                # Now recurse to the next level
                npNewData = np.array(newData)
                npNewClasses = np.array(newClasses)
                subtree = self.build_tree(npNewData, npNewClasses, newNames)

                # And on returning, add the subtree on to the tree
                subtree.parent = tree
                subtree.parent_feature_value = value
                # tree[featureNames[bestFeature]][value] = subtree

            return tree

    def classify(self, data_test, node):
        """
        Used to predict values. Using node objects, function can be called recursively to find a leaf node. Base case
        checks for a leaf; if true, returns predicted target value held within, otherwise, traverses tree in search
        of leaf.
        :param data_test: test data to predict on. data_test's feature values used to "ask questions" (check feature
        values), allowing traversal of tree.
        :param node: starts at root of decision_tree. If node given not a leaf, function called recursively until leaf
        found. Nodes hold target values, their parent's feature_value's, and feature value's
        :return: return predicted target when leaf
        """
        # If a leaf node, return target value held within (make prediction)
        if node.is_leaf:
            return node.classification
        # If not, traverse tree in search of leaf using value of test data's feature value
        else:
            feature_index = self.featureNames.index(node.feature)
            data_test_value = data_test[feature_index]

            children = list(node.children)
            for child in children:
                # Check children's parent_value's to see which node to travel to
                if child.parent_feature_value == data_test_value:
                    return self.classify(data_test, child)

    def predict(self, data_test):
        """
        Use test data's values to traverse tree to find target values based on test data's feature values.
        :param data_test: data to make prediction on.
        :return: target class predicted.
        """
        root = self.decision_tree
        data_test = data_test[0]

        # Make prediction
        prediction =  self.classify(data_test, root)

        return prediction

    def print_tree(self):
        """
        Prints tree two ways: 1. using anytree module (more compact) and 2. using pptree (more like traditional
        representation of trees.
        Note: off-the-shelf decision tree classifier exports a .dot file, whose code can be copied and pasted at
        http://webgraphviz.com/ to render visualization of tree for comparison. This code is found in  run_test().
        :return:
        """
        print("Printing tree using anytree:\n")
        print(RenderTree(self.decision_tree))

        print("\nPrinting tree using pptree:")
        print_tree(self.decision_tree)


def read_data(dataset_option):
    """
    Returns a pandas DataFrame of dataset.
    :param dataset_option: based on menu option, reads appropriate dataset with custom adjustments if necessary.
    :return: Pandas dataframe object
    """
    if dataset_option is "iris":
        """
        - Number of Instances: 150 (50 in each of three classes)
        - Number of Attributes: 4 numeric, predictive attributes and the class 
        - Missing Attribute Values: None        
        - Attribute Information:
            1. sepal length cm
            2. sepal width  cm
            3. petal length cm
            4. petal width  cm
        - Class Values:  
            5. class: 
            -- Iris Setosa
            -- Iris Versicolour
            -- Iris Virginica 
        - Summary Statistics:
        	              Min  Max   Mean    SD   Class Correlation
            sepal length: 4.3  7.9   5.84  0.83    0.7826   
            sepal width:  2.0  4.4   3.05  0.43   -0.4194
            petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
            petal width:  0.1  2.5   1.20  0.76    0.9565  (high!)   
        """
        headers_iris = ["sepal_length", "sepal_width", "petal_length", "petal_width", "classification"]
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                    header = None, names = headers_iris)
    elif dataset_option is "lenses":
        """
        - Number of Instances: 24
        - Number of Attributes: 4 (all nominal)  
        - Number of Missing Attribute Values:   0        
        - Attribute Information:
                1. age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
                2. spectacle prescription:  (1) myope, (2) hypermetrope
                3. astigmatic:     (1) no, (2) yes
                4. tear production rate:  (1) reduced, (2) normal
        - Class Information:
            -- 3 Classes
                1 : the patient should be fitted with hard contact lenses,
                2 : the patient should be fitted with soft contact lenses,
                3 : the patient should not be fitted with contact lenses.
            -- Class Distribution:
                1. hard contact lenses: 4
                2. soft contact lenses: 5
                3. no contact lenses:   15
        """
        headers_lenses = ["age", "spectacle_rx", "astimatic", "tear_prod_rate", "fit_classification"]
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data",
                    header = None, delim_whitespace = True, names = headers_lenses)
    elif dataset_option is "voting":
        """
        - Number of Instances: 435 (267 democrats, 168 republicans)
        - Number of Attributes: 16 + class name = 17 (all Boolean valued)
        - Missing Attribute Values: Denoted by "?"       
        - Attribute Information:
            2. handicapped-infants: 2 (y,n)
            3. water-project-cost-sharing: 2 (y,n)
            4. adoption-of-the-budget-resolution: 2 (y,n)
            5. physician-fee-freeze: 2 (y,n)
            6. el-salvador-aid: 2 (y,n)
            7. religious-groups-in-schools: 2 (y,n)
            8. anti-satellite-test-ban: 2 (y,n)
            9. aid-to-nicaraguan-contras: 2 (y,n)
            10. mx-missile: 2 (y,n)
            11. immigration: 2 (y,n)
            12. synfuels-corporation-cutback: 2 (y,n)
            13. education-spending: 2 (y,n)
            14. superfund-right-to-sue: 2 (y,n)
            15. crime: 2 (y,n)
            16. duty-free-exports: 2 (y,n)
            17. export-administration-act-south-africa: 2 (y,n)
        - Class Values: 
            1. Class Name: 2 (democrat, republican)
        """
        headers_voting = ["classification_party", "handicapped-infants", "water-project-cost-sharing",
                          "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
                          "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
                          "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending",
                          "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data",
                    header = None, names = headers_voting)


def handle_numeric_and_non_numeric_data(df, dataset_option):
    """
    Categorical data is discretized where needed.
    :param df: Pandas dataframe to handle.
    :param dataset_option: dataset chosen from menu. Allows focus to be placed on one dataset's customized handling.
    :return: handled dataset.
    """
    if dataset_option is "iris":
        """
        - Number of Instances: 150 (50 in each of three classes)
        - Number of Attributes: 4 numeric, predictive attributes and the class 
        - Summary Statistics:
        	              Min  Max   Mean    SD   Class Correlation
            sepal length: 4.3  7.9   5.84  0.83    0.7826   
            sepal width:  2.0  4.4   3.05  0.43   -0.4194
            petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
            petal width:  0.1  2.5   1.20  0.76    0.9565  (high!)
        - Header:   
            ["sepal_length", "sepal_width", "petal_length", "petal_width", "classification"]
        """
        # For Iris, I will use binning. These bins are based on boxplots calculated in Excel.
        # The values within mirror the attributes' boxplot values of Min, IQ1, IQ2, IQ3, and Max.
        sepal_length_bins = [0, 5.1, 5.8, 6.4,   8]
        sepal_width_bins  = [0, 2.8,   3, 3.3, 4.5]
        petal_length_bins = [0, 1.6, 4.4, 5.1,   7]
        petal_width_bins  = [0, 0.3, 1.3, 1.8, 2.6]

        # Attribute values are coded. Make sure test data reflects this.
        # group_names = ["short", "medium-short", "medium-long", "long"]
        group_names   = [      0,              1,             2,      3]

        df["sepal_length_grade"] = pd.cut(df["sepal_length"], sepal_length_bins, labels = group_names)
        df["sepal_width_grade" ] = pd.cut(df["sepal_width" ], sepal_width_bins,  labels = group_names)
        df["petal_length_grade"] = pd.cut(df["petal_length"], petal_length_bins, labels = group_names)
        df["petal_width_grade" ] = pd.cut(df["petal_width" ], petal_width_bins,  labels = group_names)

        return df.drop(["sepal_length", "sepal_width", "petal_length", "petal_width"], axis = 1)
    elif dataset_option is "lenses":
        # No data to be handled.
        return df
    elif dataset_option is "voting":
        # Converting "y," "n", and "?" to discrete, numerical values:
        # For reference: ? = 0, n = 1, and y = 2
        df_attributes = df.drop("classification_party", axis = 1)
        df_attributes = df_attributes.apply(LabelEncoder().fit_transform)
        return df[["classification_party"]].join(df_attributes)


def handle_missing_data(df, dataset_option):
    """
    Handle missing values in dataset.
    :param df: dataframe to work with.
    :param dataset_option: menu option given from user.
    :return: handled Pandas dataset.
    """
    if dataset_option is "iris":
        # - Missing Attribute Values: none
        return df
    elif dataset_option is "lenses":
        # - Missing Attribute Values: none
        return df
    elif dataset_option is "voting":
        # While not explicit that missing values were denoted as "?" and "?" may denote non-yay, non-nay vote, these
        # "?"-denoted values will be handled. Looking at some simple charts in Excel with dataset values made
        # partisanship among voters clear, e.g. Republicans tended to vote yay while Democrats tended to vote nay.
        # Therefore, values will be imputed for "?" values based on frequency of party votes, e.g. if Democrats tended
        # to vote yes on issue (feature), then if Democrat, value of "y" will be imputed; if Republican, then "n", etc.

        # Identify members of parties
        party_affiliation_dictionary = {}
        for row in range(df.shape[0]):
            party = df.get_value(row, "classification_party")
            party_affiliation_dictionary[row] = party

        # Identify w
        partisan_votes = {}
        column_names = list(df.columns.values)
        for column in range(df.shape[1]):
            partisan_votes[column] = {}
            partisan_votes[column]["democrat_yes"  ] = 0
            partisan_votes[column]["democrat_no"   ] = 0
            partisan_votes[column]["republican_yes"] = 0
            partisan_votes[column]["republican_no" ] = 0

            # Attribute 0 is class in this dataset; just move on
            if column == 0:
                continue
            else:
                for row in range(df.shape[0]):
                    vote = df.get_value(row, column_names[column])

                    # Tally votes based on decision and party
                    if   vote == 2 and party_affiliation_dictionary[row] == "democrat":
                        partisan_votes[column]["democrat_yes"  ] += 1
                    elif vote == 1 and party_affiliation_dictionary[row] == "democrat":
                        partisan_votes[column]["democrat_no"   ] += 1
                    elif vote == 2 and party_affiliation_dictionary[row] == "republican":
                        partisan_votes[column]["republican_yes"] += 1
                    elif vote == 1 and party_affiliation_dictionary[row] == "republican":
                        partisan_votes[column]["republican_no" ] += 1
                    else:
                        continue

        # Determine what value to impute based on partisan_votes dictionary's values
        for column in range(df.shape[1]):
            for row in range(df.shape[0]):
                # Get the y, n, or ? value
                vote = df.get_value(row, column_names[column])

                # Do this for "?" values only
                if vote == 0:
                    party_affiliation = party_affiliation_dictionary[row]

                    # If Congress person Democrat, vote with majority of own party
                    if party_affiliation == "democrat":
                        democrat_yes_votes = partisan_votes[column]["democrat_yes"]
                        democrat_no_votes  = partisan_votes[column]["democrat_no" ]

                        if (democrat_yes_votes >= democrat_no_votes):
                            df.set_value(row, column_names[column], 2)
                        else:
                            df.set_value(row, column_names[column], 1)
                    # If Congress person Republican, vote with majority of own party
                    elif party_affiliation == "republican":
                        republican_yes_votes = partisan_votes[column]["republican_yes"]
                        republican_no_votes  = partisan_votes[column]["republican_no" ]

                        if (republican_yes_votes >= republican_no_votes):
                            df.set_value(row, column_names[column], 2)
                        else:
                            df.set_value(row, column_names[column], 1)
                else:
                    continue
        return df


def convert_to_numpy_array(df, dataset_option):
    """
    Convert Pandas dataframe to NumPy arrays. ALso return feature name to be used to build decision tree.
    :param df: dataframe to partition and convert
    :param dataset_option: used to partition desired dataset by user
    :return: data (X), targets (y), and feature names
    """
    if dataset_option is "iris":
        return df.drop("classification", axis = 1).values, \
               df[["classification"      ]].values, \
               list(df.drop("classification", axis = 1))   # Names of everything but the classes are needed
    elif dataset_option is "lenses":
        return df.drop("fit_classification", axis = 1).values, \
               df[["fit_classification"  ]].values,\
               list(df.drop("fit_classification", axis = 1))
    elif dataset_option is "voting":
        return df.drop("classification_party", axis = 1).values, \
               df[["classification_party"]].values,\
               list(df.drop("classification_party", axis = 1))


def run_test(dataset_option):
    """
	The main portion of the programs. Calls functions to handle core requirements.
	"""

    # Read data from text files.
    df = read_data(dataset_option)

    # Appropriately handle numeric or non-numeric data.
    df = handle_numeric_and_non_numeric_data(df, dataset_option)

    # Appropriately handle missing data.
    df = handle_missing_data(df, dataset_option)

    # Partition Pandas dataframe into NumPy arrays; also get feature names to build tree with
    X, y, featureNames = convert_to_numpy_array(df, dataset_option)

    # Instantiate custom decision tree and train with dataset values
    clf = ID3DecisionTreeClassifier()
    clf = clf.fit(X, y, featureNames)

    # Prints two versions of trees
    clf.print_tree()

    # Basic experimentation. Predicting encoded numeric values for Iris and voting datasets. Please see comments in
    # handle_numeric_and_non_numeric_data() for more on how encoding and discretizing of data was done.
    print("My prediction:")
    if dataset_option == "iris":
        # print(    clf.predict([['short', 'long', 'short', 'short']]))     # IRIS:   Should predict Iris-setosa
        print("\t>", clf.predict([[      0,      3,       0,      0]]))
    elif dataset_option == "lenses":                                        # LENSES: Should predict 3
        print("\t>", clf.predict([[1, 1, 1, 1]]))
    elif dataset_option == "voting":                                        # VOTING: Should predict democrat
        #     print(clf.predict([['n','y','y','n','y','y','n','n','n','n','n','n','y','y','y','y']]))
        print("\t>", clf.predict([[ 1,  2,  2,  1,  2,  2,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2]]))

    # Using off-the-shelf decision tree implemetation and comparing results:
    clf = sklearn_tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

    # Please see comments in handle_numeric_and_non_numeric_data() for more on how encoding was done if not clear, but
    # these are the same values as above, used to compare to custom tree's prediction.
    print("\nsklearn prediction:")
    if dataset_option == "iris":                                            # IRIS:   Should predict Iris-setosa
        print("\t>", clf.predict([[0, 3, 0, 0]])[0])
    elif dataset_option == "lenses":                                        # LENSES: Should predict 3
        print("\t>", clf.predict([[1, 1, 1, 1]])[0])
    elif dataset_option == "voting":                                        # VOTING: Should predict democrat
        print("\t>", clf.predict([[1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]])[0])

    # This will export graphviz code to view sklearn tree to a .dot file.
    # You can copy and paste the code at the website to see the tree if you'd like: http://webgraphviz.com
    sklearn_tree.export_graphviz(clf, out_file="tree.dot", feature_names=featureNames, proportion=False)

    print("\n\ngraphviz code to view a visualization of the sklearn tree was exported to a .dot file (tree.dot).\n"
          "To view sklearn tree simply copy and paste the code at http://webgraphviz.com.\n")

    print("\n", "* " * 25, "\n")


def main(argv):
    """
    Driver function for program.
    :param argv:
    :return: none
    """

    menu = {}
    menu['1'] = "- Iris Dataset"
    menu['2'] = "- Lenses Dataset"
    menu['3'] = "- Voting Records Dataset"
    menu['4'] = "- Exit"
    while True:
        options = menu.keys()
        sorted(options)
        for entry in options:
            print(entry, menu[entry])

        selection = input("\nPlease select a dataset and hit Enter: ")
        if   selection == '1':
            print()
            run_test(dataset_option = "iris")
        elif selection == '2':
            print()
            run_test(dataset_option = "lenses")
        elif selection == '3':
            print()
            run_test(dataset_option = "voting")
        elif selection == '4':
            print()
            break
        else:
            print()
            print("\t\t\tPlease enter valid input.\n")


if __name__ == "__main__":
    main(sys.argv)