from Dataset import Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import os
import numpy as np
from utils import *

conf = Dataset.parse_conf(conf_type="yaml")
dataset = Dataset(conf=conf, scaler="MinMaxScaler", auto_preprocess=False)

# EXECUTION
results = []
support_vectors = []

for random_state in range(0, Dataset.CONF["exec"]["N_RANDOM_STATE"]):
    # we iterate over random_state to be sure that traning set data changes from an iteration to another
    # so, we have more variability in the test
    for iteration in range(0, Dataset.CONF["exec"]["N_ITERATIONS"]):
        print()
        print("="*30)
        print()
        print(f"start random_state {random_state}, iteration {iteration}")

        dataset.load_data(f"rs{random_state}_it{iteration}")
        # pca
        dataset.pca_fit(n_components=0.8)
        dataset.pca_transform()
        # svm
        dataset.svm_fit_predict()

        # result analysis and plotting
        if len(dataset.svm_support_vectors) > 0:
            print(f"n. support vectors: {dataset.svm_support_vectors}, n. train samples: {len(dataset.y_train)}")
        accuracy = accuracy_score(dataset.y_test, dataset.svm_prediction)
        precision = precision_score(dataset.y_test, dataset.svm_prediction, average='macro')
        recall = recall_score(dataset.y_test, dataset.svm_prediction, average='macro')

        confusion_mat = confusion_matrix(dataset.y_test, dataset.svm_prediction)
        display_confusion_mat = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        display_confusion_mat.plot()
        display_confusion_mat.figure_.savefig(os.path.join(Dataset.CONF["folder"]["OUTPUT_ANALYSIS"], 'confusion_matrix.png'))

        results.append([accuracy, precision, recall])
        print([accuracy, precision, recall])
        support_vectors.append(dataset.svm_support_vectors)

# RESULT ANALYSIS
results = np.array(results)
support_vectors = np.array(support_vectors)
analyze_results(results=results,
                support_vectors=support_vectors,
                n_iterations=(Dataset.CONF["exec"]["N_RANDOM_STATE"] * Dataset.CONF["exec"]["N_ITERATIONS"]),
                len_y_train=len(dataset.y_train),
                len_y_test=len(dataset.y_test),
                output_analysis_path=Dataset.CONF["folder"]["OUTPUT_ANALYSIS"],
                params=[])