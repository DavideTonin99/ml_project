import sys
import os
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from preprocess import do_preprocessing
from utils import parse_yaml, prepare_folder_structure, generate_dataset, analyze_results
from classifiers import pca_svm, pca_knn

print(f"start {sys.argv[0]}")
yaml_dict = parse_yaml("conf.yaml")

# PATH
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), yaml_dict["path"]["BASE_PATH"])
DATASET_PATH = os.path.join(BASE_PATH, yaml_dict["path"]["DATASET_PATH"])
DATASET_PROCESSED_PATH = os.path.join(BASE_PATH, yaml_dict["path"]["DATASET_PROCESSED_PATH"])
DATASET_SCALED_PATH = os.path.join(BASE_PATH, yaml_dict["path"]["DATASET_SCALED_PATH"])
PROCESSED_IMAGES = os.path.join(BASE_PATH, yaml_dict["path"]["PROCESSED_IMAGES"])
LABELS = os.path.join(BASE_PATH, yaml_dict["path"]["LABELS"])
OUTPUT_ANALYSIS_PATH = os.path.join(BASE_PATH, yaml_dict["path"]["OUTPUT_ANALYSIS_PATH"])

# FLAGS
PREPROCESSING = yaml_dict["flag"]["PREPROCESSING"]
GREYSCALE = yaml_dict["flag"]["GREYSCALE"]
GENERATE_DATASET = yaml_dict["flag"]["GENERATE_DATASET"]

# PARAMS
N_ITERATIONS = yaml_dict["exec"]["N_ITERATIONS"]
N_RANDOM_STATE = yaml_dict["exec"]["N_RANDOM_STATE"]

path_list = [BASE_PATH, DATASET_PATH, DATASET_PROCESSED_PATH, DATASET_SCALED_PATH, OUTPUT_ANALYSIS_PATH]
prepare_folder_structure(path_list)

if PREPROCESSING:
    do_preprocessing(DATASET_PATH, DATASET_PROCESSED_PATH, GREYSCALE)

processed_data = np.load(PROCESSED_IMAGES)
labels = np.load(LABELS)

if GENERATE_DATASET:
    generate_dataset(data=processed_data, labels=labels, path=DATASET_SCALED_PATH, n_iterations=N_ITERATIONS,
                     n_random_state=N_RANDOM_STATE)
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

        n_support_vectors = []
        # prediction, n_support_vectors = pca_svm(x_train, y_train, x_test)
        prediction = pca_knn(x_train, y_train, x_test)

        # result analysis and plotting
        if len(n_support_vectors) > 0:
            print(f"n. support vectors: {n_support_vectors}, n. train samples: {len(y_train)}")
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average='macro')
        recall = recall_score(y_test, prediction, average='macro')

        confusion_mat = confusion_matrix(y_test, prediction)
        display_confusion_mat = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        display_confusion_mat.plot()
        display_confusion_mat.figure_.savefig(os.path.join(OUTPUT_ANALYSIS_PATH, 'confusion_matrix.png'))

        results.append([accuracy, precision, recall])
        # print([accuracy, precision, recall])
        support_vectors.append(n_support_vectors)

# RESULT ANALYSIS
results = np.array(results)
support_vectors = np.array(support_vectors)
analyze_results(results=results,
                support_vectors=support_vectors,
                n_iterations=(N_RANDOM_STATE * N_ITERATIONS),
                len_y_train=len(y_train),
                len_y_test=len(y_test),
                output_analysis_path=OUTPUT_ANALYSIS_PATH)

exit(0)
