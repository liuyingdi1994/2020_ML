import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

random_state = 666
np.random.seed(random_state)


def load_data():
    # load data
    X, y = datasets.make_moons(n_samples=300, noise=0.2)
    # shuffle
    shuffle_indexes = np.random.permutation(len(X))
    X, y = X[shuffle_indexes], y[shuffle_indexes]
    return X, y


def create_pipeline(kernel='poly', coef0=0.0, degree=3, C=1.0):
    return Pipeline([
        ('standard_scaler', StandardScaler()),
        ('svc', SVC(kernel=kernel, coef0=coef0, degree=degree, C=C))
    ])


if __name__ == '__main__':
    X, y = load_data()
    # plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r')
    # plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b')
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    svc_pipeline = create_pipeline(kernel='poly', coef0=1.0, degree=3, C=1.0)
    svc_pipeline.fit(X_train, y_train)

    score = svc_pipeline.score(X_test, y_test)
    print(score)  # 0.9666666666666667

    pass
