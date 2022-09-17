import os
from itertools import repeat

import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import utils
from matplotlib import pyplot as plt


class Dataset:
    """
    """
    CONF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf.yaml")
    CONF = {}

    def __init__(self, conf=None, scaler='MinMaxScaler', auto_preprocess=False):
        self.svm_support_vectors = []
        self.prediction = []
        self.results = []
        self.classificator_params = {}

        Dataset.CONF = conf if conf is not None else Dataset.parse_conf()
        Dataset.prepare_folder_structure(Dataset.CONF["folder"])

        self.set_scaler(scaler)

        if auto_preprocess:
            if Dataset.CONF["flag"]["PREPROCESSING"]:
                Dataset.resize(source_path=Dataset.CONF["folder"]["DATASET"],
                               dest_path=Dataset.CONF["folder"]["RESIZED"],
                               save_images_resized=Dataset.CONF["exec"]["SAVE_IMAGES_RESIZED"])
            if Dataset.CONF["flag"]["GENERATE_DATASET"]:
                self.generate_scaled_data()

    def load_data(self, filename):
        print(f"Load data: {filename}")
        self.x_train = np.load(os.path.join(Dataset.CONF["folder"]["SCALED"], f"{filename}_x_train.npy"))
        self.x_test = np.load(os.path.join(Dataset.CONF["folder"]["SCALED"], f"{filename}_x_test.npy"))
        self.y_train = np.load(os.path.join(Dataset.CONF["folder"]["SCALED"], f"{filename}_labels_train.npy"))
        self.y_test = np.load(os.path.join(Dataset.CONF["folder"]["SCALED"], f"{filename}_labels_test.npy"))

        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)

    def pca_fit(self, n_components=0.8):
        print(f"PCA fit, n_components {n_components}")
        self.model_pca = PCA(n_components)
        self.model_pca.fit(self.x_train)

    def pca_transform(self):
        print("PCA transform")
        self.x_train_pca = self.model_pca.transform(self.x_train)
        self.x_test_pca = self.model_pca.transform(self.x_test)

        # salva output ricostruzione dopo pca
        n_images = len(self.y_train)

        reprojected_img = np.dot(self.x_train_pca, self.model_pca.components_.T.transpose())
        im_h = im_w = 250
        for i in range(n_images):
            plt.figure()

            temp_img = reprojected_img[i, :]
            img_min = temp_img.min()
            if img_min < 0:
                temp_img = temp_img - img_min
            else:
                temp_img = temp_img + img_min
            img_max = temp_img.max()
            temp_img = temp_img / img_max

            temp_img = temp_img.reshape((im_h, im_w, 3))
            plt.imshow(temp_img)
            name = "C:\\Users\\Manu\\Desktop\\pca_output_color_10\\" + str(i)
            plt.savefig(name)

    def svm_fit_predict(self, kernel='rbf', C=1, max_iteration=10000, gamma="scale"):
        print("SVM fit & predict probability")
        self.model_svc = svm.SVC(
            kernel=kernel, C=C, gamma=gamma, max_iter=max_iteration, decision_function_shape='ovr', probability=True
        )
        # print(
        #     f"SVM params: n of classes {len(self.model_svc.classes_)} n_iter_{self.model_svc.n_iter_}, n_features_in_{self.model_svc.n_features_in_}")
        self.model_svc.fit(self.x_train_pca, self.y_train)
        prediction = np.argmax(self.model_svc.predict_proba(self.x_test_pca), axis=1)
        self.prediction = prediction

    def knn_fit_predict(self, n_neighbors=5, metric="euclidean"):
        print("KNN fit & predict probability")

        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1, metric=metric)
        # print(
        #     f"KNN params: n of classes {len(neigh.classes_)} effective_metric_{neigh.effective_metric_}, n_features_in_{neigh.n_features_in_}")
        neigh.fit(self.x_train_pca, self.y_train)
        prediction = np.argmax(neigh.predict_proba(self.x_test_pca), axis=1)
        self.prediction = prediction

    def iterate_predictions(self, pca_obj=None, svm_obj=None, knn_obj=None, cnn=None):
        # we iterate over random_state to be sure that training set data changes from an iteration to another so, we have more variability in the test
        for random_state in range(0, Dataset.CONF["exec"]["N_RANDOM_STATE"]):
            for iteration in range(0, Dataset.CONF["exec"]["N_ITERATIONS"]):
                print()
                print("=" * 30)
                print()
                print(f"start random_state {random_state}, iteration {iteration}")

                self.load_data(f"rs{random_state}_it{iteration}")

                if pca_obj:
                    self.pca_fit(n_components=pca_obj["n_components"])
                    self.pca_transform()
                if svm_obj:
                    self.classificator_params = svm_obj

                    self.svm_fit_predict(
                        kernel=svm_obj["kernel"],
                        C=svm_obj["C"],
                        max_iteration=svm_obj["max_iteration"],
                        gamma=svm_obj["gamma"]
                    )
                if knn_obj:
                    self.classificator_params = knn_obj

                    self.knn_fit_predict(
                        n_neighbors=knn_obj["n_neighbors"],
                        metric=knn_obj["metric"]
                    )

                self.compute_stats(random_state, iteration)

    def compute_stats(self, random_state, iteration):
        accuracy = accuracy_score(self.y_test, self.prediction)
        precision = precision_score(self.y_test, self.prediction, average='macro')
        recall = recall_score(self.y_test, self.prediction, average='macro')
        # print([accuracy, precision, recall])

        self.save_confusion_matrix(random_state, iteration)

        self.results.append([accuracy, precision, recall])
        self.svm_support_vectors.append(self.model_svc.n_support_)

    def empty_stats(self):
        self.results = []
        self.svm_support_vectors = []

    def save_confusion_matrix(self, random_state, iteration):
        confusion_mat = confusion_matrix(self.y_test, self.prediction)
        display_confusion_mat = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
        display_confusion_mat.plot()
        display_confusion_mat.figure_.savefig(
            os.path.join(Dataset.CONF["folder"]["OUTPUT_ANALYSIS"],
                         f'confusion_matrix_{random_state}_it{iteration}.png'))

    def set_scaler(self, scaler):
        """
        """
        if scaler == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif scaler == 'StandardScaler':
            self.scaler = StandardScaler()
        else:
            raise Exception("Not implemented")

    def generate_scaled_data(self, data=None, labels=None, scaled_out_path=None, n_iterations=None,
                             n_random_state=None):
        """
        """
        print("Generate scaled data")
        print("=" * 30)
        data = data if data is not None else np.load(Dataset.CONF['filename']['RESIZED_IMAGES'])
        labels = labels if labels is not None else np.load(Dataset.CONF['filename']['LABELS'])
        scaled_out_path = scaled_out_path if scaled_out_path is not None else Dataset.CONF['folder']['SCALED']

        n_iterations = n_iterations if n_iterations is not None else Dataset.CONF['exec']['N_ITERATIONS']
        n_random_state = n_iterations if n_iterations is not None else Dataset.CONF['exec']['N_RANDOM_STATE']

        for random_state in range(0, n_random_state):
            # we iterate over random_state to be sure that traning set data changes from an iteration to another
            # so, we have more variability in the test
            for iteration in range(0, n_iterations):
                print(f"Start random_state: {random_state}, iteration: {iteration}")

                # train-test split
                x_train, x_test, y_train, y_test = train_test_split(
                    data, labels, test_size=0.3, shuffle=True, random_state=random_state
                )

                # reshape the dataset (x_train and x_test):
                # from: 4-dimensions (n_images, x, y, rgb)
                # to: 2-dimensions (n_images * x * y, rgb) => n. rows, 3 columns
                x_train_per_channel, x_test_per_channel = [], []
                for i in range(3):
                    x_train_per_channel.append(x_train[:, :, :, i].reshape(1, -1)[0])
                    x_test_per_channel.append(x_test[:, :, :, i].reshape(1, -1)[0])

                x_train_per_channel = np.array(x_train_per_channel).T
                x_test_per_channel = np.array(x_test_per_channel).T

                # fit the scaler on the 3 channels
                self.scaler.fit(x_train_per_channel)

                x_train_scaled = self.scaler.transform(x_train_per_channel)
                x_test_scaled = self.scaler.transform(x_test_per_channel)

                # reshape to return to the original shape of the images
                x_train = x_train_scaled.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
                x_test = x_test_scaled.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

                # save dataset and labels to reuse in classification
                x_train, x_test, y_train, y_test = np.array(x_train), np.array(x_test), np.array(y_train), np.array(
                    y_test)
                np.save(os.path.join(scaled_out_path, f"rs{random_state}_it{iteration}_x_train.npy"), x_train)
                np.save(os.path.join(scaled_out_path, f"rs{random_state}_it{iteration}_x_test.npy"), x_test)
                np.save(os.path.join(scaled_out_path, f"rs{random_state}_it{iteration}_labels_train.npy"), y_train)
                np.save(os.path.join(scaled_out_path, f"rs{random_state}_it{iteration}_labels_test.npy"), y_test)

    @staticmethod
    def parse_conf(conf_type="yaml"):
        """
        Parse configuration from file
        :param: conf_type
        :return: conf : configuration in the format key - value
        """
        print("Loading configuration from file")
        print("=" * 30)
        conf = None
        if conf_type == "yaml":
            stream = open(Dataset.CONF_PATH, 'r')
            conf = yaml.load(stream, Loader=yaml.FullLoader)
        else:
            raise Exception("Not implemented")

        conf['folder']['BASE'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                              conf["folder"]["BASE"])
        for k in ["folder", "filename"]:
            for key, value in conf[k].items():
                conf[k][key] = os.path.join(conf["folder"]["BASE"], value)

        for k in conf.keys():
            print(f"* {k} *")
            for key, value in conf[k].items():
                print(f"\t{key}: {value}")
        return conf

    @staticmethod
    def prepare_folder_structure(path_list):
        """
        Generate folders, if not exist
        :param path_list: {key: path, ...}
        :return: None
        """
        for key, path in path_list.items():
            if not os.path.exists(path):
                os.makedirs(path)

    @staticmethod
    def resize(source_path, dest_path, save_images_resized=False):
        """
        Resize the images
        :return:
        """
        print("Resize images")
        print("=" * 30)
        dirs = utils.get_dirs_in_dir(source_path)
        dirs.pop(0)  # remove first element "./dataset/"

        resized_images = []
        paths = []
        labels = []

        for index, curr_dir in enumerate(dirs):
            print(f"curr_dir {curr_dir}")
            src = curr_dir
            dst = curr_dir.replace(src, dest_path)

            if not os.path.exists(dst):
                os.makedirs(dst)
            files = utils.get_files_dir(src)
            resized = utils.resize_images(dim=(Dataset.CONF["img"]["width"], Dataset.CONF["img"]["height"]),
                                          images=files, src=src, dst=dst)
            for img in resized:
                paths.append(img["path"])
                resized_images.append(img["data"])
                labels.append(index)

        resized_images = np.array(resized_images)
        if save_images_resized:
            utils.save_images(resized_images, paths)

        np.save(Dataset.CONF['filename']['RESIZED_IMAGES'], resized_images)
        np.save(Dataset.CONF['filename']['LABELS'], labels)
