from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def pca_svm(x_train, y_train, x_test):
    # params
    percentage_n_components = 0.80
    kernel = 'rbf'
    C = 1
    max_iteration = 100000
    gamma = "scale"

    # print(x_train.shape)
    # PCA
    model_pca = PCA(n_components=0.80)
    model_pca.fit(x_train)
    x_train_pca = model_pca.transform(x_train)
    x_test_pca = model_pca.transform(x_test)

    # SVM
    model_svc = svm.SVC(
        kernel=kernel, C=C, gamma=gamma, max_iter=max_iteration, decision_function_shape='ovr', probability=True
    )
    model_svc.fit(x_train_pca, y_train)
    prediction = np.argmax(model_svc.predict_proba(x_test_pca), axis=1)

    return prediction, model_svc.n_support_


def pca_knn(x_train, y_train, x_test):
    # PCA
    model_pca = PCA(n_components=0.80)
    model_pca.fit(x_train)
    x_train_pca = model_pca.transform(x_train)
    x_test_pca = model_pca.transform(x_test)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=13, n_jobs=-1)
    neigh.fit(x_train_pca, y_train)
    prediction = np.argmax(neigh.predict_proba(x_test_pca), axis=1)

    print(neigh.classes_)

    return prediction
