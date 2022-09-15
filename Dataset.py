import os
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import utils


class Dataset:
    CONF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf.yaml")
    CONF = {}

    def __init__(self, conf=None, scaler='MinMaxScaler'):
        Dataset.CONF = conf if conf is not None else Dataset.parse_conf()

        Dataset.CONF['folder']['BASE'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                      Dataset.CONF["folder"]["BASE"])
        for k in ["folder", "file"]:
            for key, value in Dataset.CONF[k].items():
                Dataset.CONF[k][key] = os.path.join(Dataset.CONF[k]["BASE"], value)
        Dataset.prepare_folder_structure(Dataset.CONF["folder"])

        self.scaler = self.set_scaler(scaler)

    def set_scaler(self, scaler):
        if scaler == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise Exception("Not implemented")

    def generate_scaled_data(self, data=None, labels=None, scaled_out_path=None, n_iterations=None, n_random_state=None):
        data = data if data is not None else np.load(Dataset.CONF['filename']['RESIZED'])
        labels = labels if labels is not None else np.load(Dataset.CONF['filename']['LABELS'])
        scaled_out_path = scaled_out_path if scaled_out_path is not None else Dataset.CONF['folder']['DATASET_SCALED']

        n_iterations = n_iterations if n_iterations is not None else Dataset.CONF['exec']['n_iterations']
        n_random_state = n_iterations if n_iterations is not None else Dataset.CONF['exec']['n_random_state']

        for random_state in range(0, n_random_state):
            # we iterate over random_state to be sure that traning set data changes from an iteration to another
            # so, we have more variability in the test
            for iteration in range(0, n_iterations):
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
        conf = None
        if conf_type == "yaml":
            stream = open(Dataset.CONF_PATH, 'r')
            conf = yaml.load(stream, Loader=yaml.FullLoader)
        else:
            raise Exception("Not implemented")
        return conf

    @staticmethod
    def prepare_folder_structure(path_list):
        """
        Generate folders, if not exist
        :param path_list:
        :return:
        """
        for p in path_list:
            if not os.path.exists(p):
                os.makedirs(p)

    @staticmethod
    def resize(source_path, dest_path, save_images_resized=False):
        """
        Resize the images
        :return:
        """
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
            resized_images = utils.resize_images(dim=(Dataset.CONF["img"]["width"], Dataset.CONF["img"]["height"]),
                                                 images=files, src=src, dst=dst)
            for img in resized_images:
                paths.append(img["path"])
                resized_images.append(img["data"])
                labels.append(index)

        resized_images = np.array(resized_images)
        if save_images_resized:
            utils.save_images(resized_images, paths)

        np.save(Dataset.CONF['filename']['RESIZED_IMAGES'], resized_images)
        np.save(Dataset.CONF['filename']['LABELS'], labels)
