import yaml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
import cv2

def parse_yaml(yaml_path):
    stream = open(yaml_path, 'r')
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)

    # print(yaml_dict)
    return yaml_dict


def prepare_folder_structure(path_list):
    for p in path_list:
        if not os.path.exists(p):
            os.makedirs(p)


def get_dirs_in_dir(path):
    """
    Get all dirname inside path
    :param path: path to check
    :return: list of dirname
    """
    return [x[0] for x in os.walk(path)]


def get_files_dir(path):
    """
    Get all filename inside path
    :param path: path to check
    :return: list of filename
    """
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def resize_images(dim, images, src, dst, greyscale_flag=False):
    """
    :param images: lista di immagini da elaborare
    :param src: path originale, e.g. "./dataset"
    :param dst: path destination, e.g. "./dataset_processed"
    :param greyscale_flag: attiva/disattiva greyscale
    :return: lista di oggetti. ogni oggetto ha 2 attributi: data (dati immagine elaborata), path (path dove salvare l'immagine)
    """
    resized_list = []
    for filepath in images:
        # IMREAD_COLOR e non IMREAD_UNCHANGED perchè alcune immagini hanno anche il quarto canale che sminchia la
        # concatenazione. IMREAD_COLOR forza 3 canali
        read_mode = cv2.IMREAD_COLOR
        # if greyscale_flag:
        #    read_mode = cv2.IMREAD_GRAYSCALE

        img = cv2.imread(filepath, read_mode)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        dest_file_path = filepath.replace(src, dst)
        resized_list.append({"data": resized_img, "path": dest_file_path})

    return resized_list


def save_images(images, paths):
    """
    """
    for idx, img in enumerate(images):
       cv2.imwrite(paths[idx], img)

def analyze_results(results, support_vectors, n_iterations, len_y_train, len_y_test, output_analysis_path, params):
    mean_results, std_results, mean_support_vectors = np.mean(results, axis=0), np.std(results, axis=0), np.mean(
        support_vectors, axis=0)
    relative_std = 100 * np.divide(std_results, mean_results)

    # print and save output to file
    output_str = f"params: {str(params)}\n" \
                 f"n_iterations: {n_iterations}\n" \
                 f"n_train_samples: {len_y_train}, n_test_samples: {len_y_test}\n" \
                 f"mean_support_vectors: {mean_support_vectors}\n" \
                 f"mean_accuracy: {mean_results[0]:.2f} +- {std_results[0]:.2f} std ({relative_std[0]:.2f}% RSC) \n" \
                 f"mean_precision: {mean_results[1]:.2f} +- {std_results[1]:.2f} std ({relative_std[1]:.2f}% RSC) \n" \
                 f"mean_recall: {mean_results[2]:.2f} +- {std_results[2]:.2f} std ({relative_std[2]:.2f}% RSC) \n"
    print(output_str)
    f = open(os.path.join(output_analysis_path, "stats_output_exec_1.txt"), "a")
    f.write(output_str)
    f.close()
