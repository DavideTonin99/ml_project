import yaml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sys


def parse_yaml(yaml_path):
    stream = open(yaml_path, 'r')
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)

    # print(yaml_dict)
    return yaml_dict


def prepare_folder_structure(path_list):
    for p in path_list:
        if not os.path.exists(p):
            os.makedirs(p)


def generate_dataset(data, labels, path, n_iterations=20, n_random_state=5):
    print("start generate_dataset %s" % sys.argv[0])

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


def analyze_results(results, support_vectors, n_iterations, len_y_train, len_y_test, output_analysis_path):
    mean_results, std_results, mean_support_vectors = np.mean(results, axis=0), np.std(results, axis=0), np.mean(
        support_vectors, axis=0)
    relative_std = 100 * np.divide(std_results, mean_results)

    # print and save output to file
    output_str = f"n_iterations: {n_iterations}\n" \
                 f"n_train_samples: {len_y_train}, n_test_samples: {len_y_test}\n" \
                 f"mean_support_vectors: {mean_support_vectors}\n" \
                 f"mean_accuracy: {mean_results[0]:.2f} +- {std_results[0]:.2f} std ({relative_std[0]:.2f}% RSC) \n" \
                 f"mean_precision: {mean_results[1]:.2f} +- {std_results[1]:.2f} std ({relative_std[1]:.2f}% RSC) \n" \
                 f"mean_recall: {mean_results[2]:.2f} +- {std_results[2]:.2f} std ({relative_std[2]:.2f}% RSC) \n"
    print(output_str)
    f = open(os.path.join(output_analysis_path, "stats_output_exec_1.txt"), "a")
    f.write(output_str)
    f.close()
