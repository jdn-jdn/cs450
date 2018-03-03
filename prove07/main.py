import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class MultiLayeredPerceptronClassifier:
    """
    A multi-layer perceptron neural network.
    """

    def __init__(self, nodes_in_layers, num_epochs, learning_rate):
        """
        Initialize attributes.
        """
        self.nodes_in_layers = nodes_in_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.bias_value = -1
        self.delta_activations = []
        self.delta_inputs = []
        self.deltas = []
        self.activations = np.array([])
        self.accuracies = []
        self.num_layers = 0

    def fit(self, training_data, training_targets):
        """
        Trains using back-propagation, implementing the basic MLP algorithm.
        :param training_data:
        :param training_targets:
        :return:
        """
        self.training_data = training_data
        self.training_targets = training_targets
        self.encoded_training_targets = self.encode_targets(training_targets)
        self.weights = self.initialize_weights(self.nodes_in_layers)
        self.num_layers = len(self.nodes_in_layers)
        bias = np.full((1, 1), self.bias_value)
        self.activations = np.zeros(self.training_targets.shape)

        # for all iterations or until all outputs are correct
        for epoch in range(self.num_epochs):

            # loops over the input vectors
            for row in range(self.training_data.shape[0]):
                data = self.training_data[row]                      # value of data initially inputs from input vector
                inputs = np.concatenate((bias.ravel(), data))
                inputs = np.reshape(inputs, (1, inputs.shape[0]))   # inputs are the data input nodes plus a bias node

                # loops over the layers
                for layer in range(self.num_layers):
                    num_nodes = self.nodes_in_layers[layer]
                    num_inputs = inputs.shape[1]
                    outputs = np.zeros((1, num_nodes))

                    # calculates outputs values
                    # loops over the activation nodes
                    for node in range(num_nodes):
                        # loop over the input nodes with dot product function
                        outputs[0][node] = np.dot(inputs, self.weights[layer][ : , node])
                        outputs[0][node] = self.sigmoid(outputs[0][node])

                    self.delta_activations.append(outputs)          # keep track for delta calculations
                    self.delta_inputs.append(inputs)
                    data = outputs                                  # outputs become new data for next layer
                    inputs = np.concatenate((bias, data), 1)        # inputs = data plus bias

                # initializes error value matrices
                for activation in self.delta_activations:
                    self.deltas.append(np.zeros(activation.shape))  # matrix will be same shape as corresponding activation

                # computes error values
                for layer in range((self.num_layers - 1), -1, -1):  # begin calculations at last layer
                    for node in range(self.deltas[layer].shape[1]): # calculate errors for each layer's nodes
                        if layer == self.num_layers - 1:            # calculations different for error at output layer
                            self.deltas[layer][0][node] = \
                                (self.delta_activations[layer][0][node] - self.encoded_training_targets[row][node]) * \
                                (self.delta_activations[layer][0][node]) * \
                                (1.0 - self.delta_activations[layer][0][node])
                        else:
                            self.deltas[layer][0][node] = \
                                (self.delta_activations[layer][0][node]) * \
                                (1.0 - self.delta_activations[layer][0][node]) * \
                                (np.dot(self.weights[layer + 1][node + 1], np.transpose(self.deltas[layer + 1])))

                # updates weights
                for layer in range(self.num_layers):
                        self.weights[layer] -= self.learning_rate * (np.dot(np.transpose(self.delta_inputs[layer]), self.deltas[layer]))

                # stores prediction in activations array
                outputs_list = list(outputs.flatten())
                self.activations[row] = outputs_list.index(max(outputs_list))

                # clear error lists for next row's calculations
                self.deltas = []
                self.delta_inputs = []
                self.delta_activations = []

            # Prints information while training
            # Stores accuracies for plotting
            self.accuracies.append(accuracy_score(self.training_targets.flatten(), self.activations))
            print('Epoch:', epoch)
            print("Accuracy: %" +
                  str(round(accuracy_score(self.training_targets.flatten(), self.activations) * 100, 3)))
            print()

        # Final accuracy on training set after fitting
        print("Final accuracy: %" +
              str(round(accuracy_score(self.training_targets.flatten(), self.activations) * 100, 3)))
        print()


    def encode_targets(self, training_targets):
        """
        Returns numpy array of encoded values.
        :param training_targets: the values to be encoded.
        :return: a numpy array of shape (training_targets.shape[0], number of unique targets in training targets).
        """
        return pd.get_dummies(training_targets.reshape(training_targets.shape[0], )).values

    def initialize_weights(self, nodes_in_layers):
        """
        Initializes weights matrices, stored in a list "weights," to random, small values.
        :param nodes_in_layers:
        :return:
        """
        weights = []
        for layer in range(len(nodes_in_layers)):
            if layer == 0:                  # number of weights for first layer determined by number of input attributes
                weights.append(np.random.uniform(-1.0, 1.0, (self.training_data.shape[1] + 1, nodes_in_layers[layer])))
            else:                           # number of weights elsewhere determined by number of columns in previous + 1
                weights.append(np.random.uniform(-1.0, 1.0, (nodes_in_layers[layer - 1] + 1,  nodes_in_layers[layer])))
        return weights

    def sigmoid(self, output_value_at_node):
        """
        Differentiable activation function.
        :param output_value_at_node: weighted sum of inputs and weights to be mapped to value between 0.0 and 1.0.
        :return: continuous value between 0.0 and 1.0.
        """
        # The largest value that can be computed by exp is just slightly larger than 709. Any value of
        # output_value_at_node < -709 will be made to be -709; this results in a value of extremely close
        # to one, so doing this doesn't effect training integrity
        if output_value_at_node < -709:
            return 1.0 / (1.0 + math.exp(-709))
        else:
            return 1.0 / (1.0 + math.exp(-output_value_at_node))


    def predict(self, test_data):
        """
        Run test data through network with trained weights and return predictions.
        :return: the predicted classes
        """
        bias = np.full((1, 1), self.bias_value)
        test_activations = np.zeros((test_data.shape[0], 1))

        # loop over the input vectors
        for row in range(test_data.shape[0]):
            data = test_data[row]                      # value of data initially inputs from input vector
            inputs = np.concatenate((bias.ravel(), data))
            inputs = np.reshape(inputs, (1, inputs.shape[0]))   # inputs are the data input nodes plus a bias node

            # loop over the layers
            for layer in range(self.num_layers):
                num_nodes = self.nodes_in_layers[layer]
                num_inputs = inputs.shape[1]
                outputs = np.zeros((1, num_nodes))

                # calculate outputs values
                # loop over the activation nodes
                for node in range(num_nodes):
                    # loop over the input nodes
                    outputs[0][node] = np.dot(inputs, self.weights[layer][ : , node])
                    outputs[0][node] = self.sigmoid(outputs[0][node])

                data = outputs                                  # outputs become new data
                inputs = np.concatenate((bias, data), 1)        # inputs = data plus bias

            # store prediction in activations array
            outputs_list = list(outputs.flatten())
            test_activations[row] = outputs_list.index(max(outputs_list))

        # The predictions
        return test_activations


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
    elif dataset_option is "cancer" or dataset_option is "digits" or dataset_option is "xor":
        # Unnecessary
        pass


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

        # Converting targets to encoded, numerical values:
        encoded_targets = {"classification":
                               {"Iris-setosa": 0,
                                "Iris-versicolor": 1,
                                "Iris-virginica": 2}}
        df.replace(encoded_targets, inplace = True)
        return df.drop(["sepal_length", "sepal_width", "petal_length", "petal_width"], axis = 1)
    elif dataset_option is "diabetes":
        # # Unnecessary, no categorical data to be converted to numerical.
        return df
    elif dataset_option is "cancer" or dataset_option is "digits" or dataset_option is "xor":
        # Unnecessary
        pass


def handle_missing_data(df, dataset_option):
    """
    Handle missing values in dataset.
    :param df: dataframe to work with.
    :param dataset_option: menu option given from user.
    :return: handled Pandas dataset.
    """
    if   dataset_option is "iris":
        # # Unnecessary, no missing attribute values
        return df
    elif dataset_option is "diabetes":
        return df.fillna(df.median())
    elif dataset_option is "cancer" or dataset_option is "digits" or dataset_option is "xor":
        # Unnecessary
        pass


def convert_to_numpy_array(df, dataset_option):
    """
    Convert Pandas dataframe to NumPy arrays. ALso return feature name to be used to build decision tree.
    :param df: dataframe to partition and convert
    :param dataset_option: used to partition desired dataset by user
    :return: data (X), targets (y), and feature names
    """
    if dataset_option is "iris":
        return df.drop("classification",          axis = 1).values, df[["classification"         ]].values
    elif dataset_option is "diabetes":
        return df.drop("class_positive_diabetes", axis = 1).values, df[["class_positive_diabetes"]].values
    elif dataset_option is "cancer":
        return datasets.load_breast_cancer().data, datasets.load_breast_cancer().target
    elif dataset_option is "digits":
        return datasets.load_digits().data, datasets.load_digits().target
    elif dataset_option is "xor":
        return np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]]), \
               np.array([0, 1, 1, 0])


def get_parameters(set_of_targets, num_features):
    """
    Get parameters from console to be used in network.
    :return: nodes_in_layers - a list with the number of nodes per layer; num_epochs - number of iterations to train on;
    learning_rate - the learning rate
    """
    nodes_in_layers = []
    num_layers = int(input('Enter the number of layers\n'
                           '\te.g., 3 = 2 hidden layers and 1 output layer\n>\t'))
    for layer in range(num_layers):
        if layer == num_layers - 1:
            nodes_in_layers.append(int(input('\t\tEnter the number of nodes in the output layer\n'
                                             '\t\t\tNote: there are ' +
                                             str(len(set_of_targets)) +
                                             ' unique targets in this dataset\n>\t')))
        else:
            nodes_in_layers.append(int(input('\t\tEnter the number of nodes in hidden layer ' + str(layer + 1) + '\n'
                                             '\t\t\tNote: there are ' +
                                             str(num_features) +
                                             ' features in this dataset\n>\t')))
    num_epochs = int(input('Enter the number of epochs\n>\t'))
    learning_rate = float(input('Enter the learning rate\n>\t'))
    print()
    return nodes_in_layers, num_epochs, learning_rate


def run_test(dataset_option, selection_name):
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
    X, y = convert_to_numpy_array(df, dataset_option)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

    #  Normalize data set
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Get parameters for network
    nodes_in_layers, num_epochs, learning_rate = get_parameters(set(y_train.flatten()), X_train.shape[1])

    # Instantiate the classifier object and train
    clf = MultiLayeredPerceptronClassifier(nodes_in_layers, num_epochs, learning_rate)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Using off-the-shelf multi-layered Perceptron and comparing results:
    #   Note: ‘sgd’ refers to stochastic gradient descent
    sklearn_clf_nodes_in_layers = tuple(nodes_in_layers)
    sklearn_clf = MLPClassifier(hidden_layer_sizes = sklearn_clf_nodes_in_layers,
                                learning_rate = 'constant',
                                learning_rate_init = learning_rate,
                                max_iter = num_epochs,
                                solver='sgd')

    sklearn_clf.fit(X_train, y_train)
    sklearn_predictions = sklearn_clf.predict(X_test)

    # Confusion matrices for custom and sklearn neural networks
    print('Confusion matrix for custom neural network:')
    print(confusion_matrix(y_test, predictions))
    print('\nConfusion matrix for sklearn neural network:')
    print(confusion_matrix(y_test, sklearn_predictions))

    # Classification reports for custom and sklearn neural networks
    print('\n\nClassification report for custom neural network:')
    print(classification_report(y_test, predictions))
    print('Classification report for sklearn neural network:')
    print(classification_report(y_test, sklearn_predictions))

    # Accuracy scores for custom and sklearn neural networks
    print("Accuracy score for custom neural network: %" +
          str(round(accuracy_score(y_test, predictions) * 100, 3)))
    print()
    print("Accuracy score for sklearn neural network: %" +
          str(round(accuracy_score(y_test, sklearn_predictions) * 100, 3)))
    print()

    # Plotting and showing progress throughout training
    plt.title('Training Progress for the ' + selection_name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    parameters = 'Layers: ' + str(len(nodes_in_layers))
    for layer, nodes in enumerate(nodes_in_layers):
        if layer == len(nodes_in_layers) - 1:
            parameters += '\n    Nodes in output layer: ' + str(nodes)
        else:
            parameters += '\n    Nodes in hidden layer #' + str(layer + 1) + ': ' + str(nodes)
    parameters += '\nEpochs: ' + str(num_epochs) + '\nLearning rate: ' + str(learning_rate)
    plt.legend(loc = 4, title = parameters, fancybox = True)
    plt.axis([0, num_epochs, 0, 1])
    plt.plot([i for i in range(num_epochs)], clf.accuracies, 'bo')
    plt.show()

    print("\n", "* " * 25, "\n")


def main(argv):
    """
    Driver function for program.
    :param argv:
    :return: none
    """

    menu = {}
    menu['1'] = "- Iris Dataset"
    menu['2'] = "- Pima Indian Diabetes Dataset"
    menu['3'] = "- Breast Cancer Wisconsin Dataset"
    menu['4'] = "- Digits Dataset"
    menu['5'] = "- XOR Dataset"
    menu['6'] = "- Exit"
    while True:
        options = menu.keys()
        sorted(options)
        for entry in options:
            print(entry, menu[entry])

        selection = input("\nPlease select a dataset and hit Enter: ")
        if   selection == '1':
            print()
            run_test("iris", menu[selection][2: ])
        elif selection == '2':
            print()
            run_test("diabetes", menu[selection][2: ])
        elif selection == '3':
            print()
            run_test("cancer", menu[selection][2: ])
        elif selection == '4':
            print()
            run_test("digits", menu[selection][2: ])
        elif selection == '5':
            print()
            run_test("xor", menu[selection][2: ])
        elif selection == '6':
            print()
            break
        else:
            print()
            print("\t\t\tPlease enter valid input.\n")


if __name__ == "__main__":
    main(sys.argv)