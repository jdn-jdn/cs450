import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


def prepare_dataset(dataset_option):
    """
    Returns selected dataset in NumPy arrays of data (X) and targets (y)
    :param dataset_option:
    :return: NumPy arrays of data (X) and targets (y)
    """
    if dataset_option == "transfusion":
        df = pd.read_csv('C:/Users/josed/PycharmProjects/cs450/prove11/blood-transfusion-service-center.csv')

        X, y = df.drop("Class", axis = 1).values, df[["Class"]].values

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

        return X_train, X_test, y_train, y_test
    elif dataset_option == "phoneme":
        df = pd.read_csv('C:/Users/josed/PycharmProjects/cs450/prove11/phoneme.csv')

        X, y = df.drop("Class", axis = 1).values, df[["Class"]].values

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

        return X_train, X_test, y_train, y_test
    elif dataset_option == "digits":
        df = pd.read_csv('C:/Users/josed/PycharmProjects/cs450/prove11/pendigits.csv')

        X, y = df.drop("class", axis = 1).values, df[["class"]].values

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

        return X_train, X_test, y_train, y_test
    elif dataset_option == "magic":
        df = pd.read_csv('C:/Users/josed/PycharmProjects/cs450/prove11/magic-telescope.csv')

        X, y = df.drop(columns=["ID", "class:"], axis = 1).values, df[["class:"]].values

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

        return X_train, X_test, y_train, y_test


def run_test(dataset_option):
    # Trying different, regular learning algorithms
    X_train, X_test, y_train, y_test = prepare_dataset(dataset_option)

    # KNeighborsClassifier
    m = KNeighborsClassifier(n_neighbors = 3)
    print(m.fit(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for KNeighborsClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # SVC
    m = SVC()
    print(m.fit(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for SVC: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # DecisionTreeClassifier
    m = DecisionTreeClassifier()
    print(m.fit(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for DecisionTreeClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # LogisticRegression
    m = LogisticRegression()
    print(m.fit(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for DecisionTreeClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # GaussianNB
    m = GaussianNB()
    print(m.fit(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for GaussianNB: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # MLPClassifier
    m = MLPClassifier()
    print(m.fit(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for MLPClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # SGDClassifier
    m = SGDClassifier()
    print(m.fit(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for SGDClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # Bagging
    m = MLPClassifier()
    bag = BaggingClassifier(
        m,
        max_samples = 0.5,
        max_features = 2,
        n_jobs = 2,
        oob_score = True
    )
    print(bag.fit(X_train, y_train))
    # print(bag.oob_score_)
    # print(bag.predict_proba(X_test))
    # print(bag.score(X_test, y_test))
    y_pred = bag.predict(X_test)
    print(">>> Accuracy for BaggingClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # Random Forests
    m = RandomForestClassifier(n_estimators = 20, oob_score = True)
    print(m.fit(X_train, y_train))
    # print(m.score(X_test, y_test))
    y_pred = bag.predict(X_test)
    print(">>> Accuracy for RandomForestClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # AdaBoost
    m = AdaBoostClassifier(base_estimator = None, n_estimators = 100)
    print(m.fit(X_test, y_test))
    # print(m.score(X_test, y_test))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for AdaBoostClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # Gradient Tree Boosting
    m = GradientBoostingClassifier(n_estimators = 10, warm_start = True)
    print(m.fit(X_train, y_train))
    # print(m.score(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for GradientBoostingClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()

    # Voting Classifier
    m = VotingClassifier(
        estimators = [
            ('lr', MLPClassifier()),
            ('rf', RandomForestClassifier()),
            ('gnb', SVC())
        ]
    )
    print(m.fit(X_train, y_train))
    # print(m.score(X_train, y_train))
    y_pred = m.predict(X_test)
    print(">>> Accuracy for VotingClassifier: %" + str(round(100 * accuracy_score(y_test, y_pred), 3)))
    print()


def main():
    """
    Driver function for program.
    :return: None
    """
    menu = {}
    menu['1'] = "- Blood Transfusion Service Center Dataset"
    menu['2'] = "- Phoneme Dataset"
    menu['3'] = "- Pen-Based Recognition of Handwritten Digits Dataset"
    menu['4'] = "- Major Atmospheric Gamma Imaging Cherenkov Telescope (MAGIC) Dataset"
    menu['5'] = "- Exit"

    while True:
        options = menu.keys()
        sorted(options)
        for entry in options:
            print(entry, menu[entry])

        selection = input("\nPlease select a dataset and press [Enter]: ")
        if   selection == '1':
            print()
            run_test("transfusion")
        elif selection == '2':
            print()
            run_test("phoneme")
        elif selection == '3':
            print()
            run_test("digits")
        elif selection == '4':
            print()
            run_test("magic")
        elif selection == '5':
            print()
            break
        else:
            print("\n\t\t\tPlease enter valid input.\n")


if __name__ == "__main__":
    main()