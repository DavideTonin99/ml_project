import os
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import utils


class Dataset:
    """
    """
    CONF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf.yaml")
    CONF = {}

    def __init__(self, conf=None, scaler='MinMaxScaler', auto_preprocess=False):
        Dataset.CONF = conf if conf is not None else Dataset.parse_conf()
        Dataset.prepare_folder_structure(Dataset.CONF["folder"])

        self.set_scaler(scaler)

        if auto_preprocess:
            if Dataset.CONF["flag"]["PREPROCESSING"]:
                Dataset.resize(source_path=Dataset.CONF["folder"]["DATASET"], dest_path=Dataset.CONF["folder"]["RESIZED"], save_images_resized=Dataset.CONF["exec"]["SAVE_IMAGES_RESIZED"])
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
        print("PCA fit")
        self.model_pca = PCA(n_components)
        self.model_pca.fit(self.x_train)

    def pca_transform(self):
        print("PCA transform")
        self.x_train_pca = self.model_pca.transform(self.x_train)
        self.x_test_pca = self.model_pca.transform(self.x_test)

    def svm_fit_predict(self, kernel='rbf', C=1, max_iteration=10000, gamma="scale"):
        print("SVM fit & predict probability")
        self.model_svc = svm.SVC(
            kernel=kernel, C=C, gamma=gamma, max_iter=max_iteration, decision_function_shape='ovr', probability=True
        )
        self.model_svc.fit(self.x_train_pca, self.y_train)
        prediction = np.argmax(self.model_svc.predict_proba(self.x_test_pca), axis=1)
        self.svm_prediction = prediction
        self.svm_support_vectors = self.model_svc.n_support_

    def knn_fit(self):
        pass

    def knn_transform(self):
        pass

    def set_scaler(self, scaler):
        """
        """
        if scaler == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif scaler == 'StandardScaler':
            self.scaler = StandardScaler()
        else:
            raise Exception("Not implemented")

    def generate_scaled_data(self, data=None, labels=None, scaled_out_path=None, n_iterations=None, n_random_state=None):
        """
        """
        print("Generate scaled data")
        print("="*30)
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
        print("="*30)
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
        print("="*30)
        dirs = utils.get_dirs_in_dir(source_path)
        dirs.pop(0)  # remove first element "./dataset/"

        resized_images = []
        paths = []
        labels = []

        for index, curr_dir in enumerate(dirs):
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
