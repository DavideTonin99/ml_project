import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from preprocess import do_preprocessing

# PATH
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
DATASET_PATH = os.path.join(BASE_PATH, 'reduced')
DATASET_PROCESSED_PATH = os.path.join(BASE_PATH, 'processed')
DATASET_SCALED_PATH = os.path.join(BASE_PATH, 'scaled')
PROCESSED_IMAGES = os.path.join(BASE_PATH, "processed_images.npy")
LABELS = os.path.join(BASE_PATH, "labels.npy")
OUTPUT_ANALYSIS_PATH = os.path.join(BASE_PATH, 'output_analysis')

# FLAGS
PREPROCESSING = False
GREYSCALE = False
GENERATE_DATASET = False

# PARAMS
N_ITERATIONS = 20
N_RANDOM_STATE = 5


def generate_dataset(data, labels, path, n_iterations=20, n_random_state=5):
    for random_state in range(0, n_random_state):
        # we iterate over random_state to be sure that traning set data changes from an iteration to another
        # so, we have more variability in the test
        for iteration in range(0, n_iterations):
            print(f"start random_state {random_state}, iteration {iteration}")

            # train-test split
            x_train, x_test, y_train, y_test = train_test_split(
                data, labels, test_size=0.3, shuffle=True, random_state=random_state
            )

            x_train_per_channel, x_test_per_channel = [], []
            for i in range(3):
                x_train_per_channel.append(x_train[:, :, :, i].reshape(1, -1)[0])
                x_test_per_channel.append(x_test[:, :, :, i].reshape(1, -1)[0])

            x_train_per_channel = np.array(x_train_per_channel).T
            x_test_per_channel = np.array(x_test_per_channel).T

            scaler = StandardScaler()
            scaler.fit(x_train_per_channel)

            x_train_scaled = scaler.transform(x_train_per_channel)
            x_test_scaled = scaler.transform(x_test_per_channel)

            x_train = x_train_scaled.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
            x_test = x_test_scaled.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

            x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
            np.save(os.path.join(path, f"rs{random_state}_it{iteration}_x_train.npy"), x_train)
            np.save(os.path.join(path, f"rs{random_state}_it{iteration}_x_test.npy"), x_test)
            np.save(os.path.join(path, f"rs{random_state}_it{iteration}_labels_train.npy"), y_train)
            np.save(os.path.join(path, f"rs{random_state}_it{iteration}_labels_test.npy"), y_test)


print(f"start {sys.argv[0]}")

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
if not os.path.exists(DATASET_PROCESSED_PATH):
    os.makedirs(DATASET_PROCESSED_PATH)
if not os.path.exists(DATASET_SCALED_PATH):
    os.makedirs(DATASET_SCALED_PATH)
if not os.path.exists(OUTPUT_ANALYSIS_PATH):
    os.makedirs(OUTPUT_ANALYSIS_PATH)

if PREPROCESSING:
    do_preprocessing(DATASET_PATH, DATASET_PROCESSED_PATH, GREYSCALE)
processed_data = np.load(PROCESSED_IMAGES)
labels = np.load(LABELS)

if GENERATE_DATASET:
    generate_dataset(data=processed_data, labels=labels, path=DATASET_SCALED_PATH, n_iterations=N_ITERATIONS, n_random_state=N_RANDOM_STATE)
# EXECUTION
results = []
support_vectors = []

for random_state in range(0, N_RANDOM_STATE):
    # we iterate over random_state to be sure that traning set data changes from an iteration to another
    # so, we have more variability in the test
    for iteration in range(0, N_ITERATIONS):
        print(f"start random_state {random_state}, iteration {iteration}")

        x_train = np.load(os.path.join(DATASET_SCALED_PATH, f"rs{random_state}_it{iteration}_x_train.npy"))
        x_test = np.load(os.path.join(DATASET_SCALED_PATH, f"rs{random_state}_it{iteration}_x_test.npy"))
        y_train = np.load(os.path.join(DATASET_SCALED_PATH, f"rs{random_state}_it{iteration}_labels_train.npy"))
        y_test = np.load(os.path.join(DATASET_SCALED_PATH, f"rs{random_state}_it{iteration}_labels_test.npy"))

        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        """
        # train-test split
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.3, shuffle=True, random_state=random_state
        )
        x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
        """

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

        # result analysis and plotting
        n_support_vectors = model_svc.n_support_
        print(f"n. support vectors: {n_support_vectors}, n. train samples: {len(y_train)}")
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average='macro')
        recall = recall_score(y_test, prediction, average='macro')

        confusion_mat = confusion_matrix(y_test, prediction)
        display_confusion_mat = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        display_confusion_mat.plot()
        display_confusion_mat.figure_.savefig(os.path.join(OUTPUT_ANALYSIS_PATH, 'confusion_matrix.png'))

        results.append([accuracy, precision, recall])
        support_vectors.append(n_support_vectors)

# RESULT ANALYSIS
results = np.array(results)
support_vectors = np.array(support_vectors)
mean_results, std_results, mean_support_vectors = np.mean(results, axis=0), np.std(results, axis=0), np.mean(
    support_vectors, axis=0)
relative_std = 100 * np.divide(std_results, mean_results)

# print and save output to file
output_str = f"n_iterations: {(N_RANDOM_STATE * N_ITERATIONS)}\n" \
             f"n_train_samples: {len(y_train)}, n_test_samples: {len(y_test)}\n" \
             f"mean_support_vectors: {mean_support_vectors}\n" \
             f"mean_accuracy: {mean_results[0]:.2f} +- {std_results[0]:.2f} std ({relative_std[0]:.2f}% RSC) \n" \
             f"mean_precision: {mean_results[1]:.2f} +- {std_results[1]:.2f} std ({relative_std[1]:.2f}% RSC) \n" \
             f"mean_recall: {mean_results[2]:.2f} +- {std_results[2]:.2f} std ({relative_std[2]:.2f}% RSC) \n"
print(output_str)
f = open(os.path.join(OUTPUT_ANALYSIS_PATH, "stats_output_exec_1.txt"), "a")
f.write(output_str)
f.close()
exit(0)
