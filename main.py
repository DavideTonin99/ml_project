from Dataset import Dataset
from utils import *

# Load Dataset
conf = Dataset.parse_conf(conf_type="yaml")
dataset = Dataset(conf=conf, scaler="MinMaxScaler", auto_preprocess=True)

# Configure classificators params
pca_obj = {
    "n_components": 0.8
}

svm_obj = {
    "kernel": 'rbf',
    "C": 1,
    "max_iteration": 10000,
    "gamma": "scale"
}

knn_obj = {
    "n_neighbors": 5,
    "metric": "euclidean"
}

# Predict
dataset.iterate_predictions(pca_obj=pca_obj, svm_obj=svm_obj)

# Save results
analyze_results(results=dataset.results,
                support_vectors=dataset.svm_support_vectors,
                n_iterations=(Dataset.CONF["exec"]["N_RANDOM_STATE"] * Dataset.CONF["exec"]["N_ITERATIONS"]),
                len_y_train=len(dataset.y_train),
                len_y_test=len(dataset.y_test),
                output_analysis_path=Dataset.CONF["folder"]["OUTPUT_ANALYSIS"],
                params=dataset.classificator_params)
dataset.empty_stats()
