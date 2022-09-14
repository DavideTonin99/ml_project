import cv2
from os import listdir, makedirs, walk
from os.path import isfile, join, exists
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np

def get_files_dir(path):
    """
    ritorna una lista tutti i file presenti in path
    """
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


def get_dirs_in_dir(path):
    """
    ritorna una lista tutte le dirs presenti in path
    """
    return [x[0] for x in walk(path)]


def crop_image(img):
    """
    croppa l'immagine in modo che sia di forma quadrata: l*l
    prende come dimensione la dimensione piu piccola dell immagine
    """
    # cropImg = img[rowStart:rowEnd, colsStart:colsEnd]

    crop_img = []
    width = img.shape[1]
    height = img.shape[0]

    if width > height:
        margin = int((width - height) / 2)
        crop_img = img[0:height, margin:(width - margin)]
    else:
        margin = int((height - width) / 2)
        crop_img = img[margin:(height - margin):, 0:width]

    # cv2.imshow("cropped", crop_img)

    return crop_img


def resize_images(dim, images, src, dst, greyscale_flag):
    """
    :param images: lista di immagini da elaborare
    :param src: path originale, e.g. "./dataset"
    :param dst: path destination, e.g. "./dataset_processed"
    :param greyscale_flag: attiva/disattiva greyscale
    :return: lista di oggetti. ogni oggetto ha 2 attributi: data (dati immagine elaborata), path (path dove salvare l'immagine)
    """
    resized_list = []
    img_width = dim
    img_height = dim
    dim = (img_width, img_height)

    for filepath in images:
        # print(filepath)

        # # IMREAD_COLOR e non IMREAD_UNCHANGED perchè alcune immagini hanno anche il quarto canale che sminchia la concatenazione. IMREAD_COLOR forza 3 canali
        read_mode = cv2.IMREAD_COLOR
        if greyscale_flag:
            read_mode = cv2.IMREAD_GRAYSCALE

        img = cv2.imread(filepath, read_mode)

        # if img.shape[0] != img.shape[1]:
        #     print("crop", filepath)
        #     img = crop_image(img)

        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # resized_img_flat = resized_img.reshape((1, -1))  # ogni immagine diventa una riga e 100*100*3 colonne

        dest_file_path = filepath.replace(src, dst)
        resized_list.append({"data": resized_img, "path": dest_file_path})

    return resized_list


def save_images(images, paths):
    """
    """
    for idx, img in enumerate(images):
        # test = img.reshape((dim, dim, -1))
        # plt.imshow(test)
        # plt.show()
        cv2.imwrite(paths[idx], img)


def do_resize(dim, src, dst, greyscale_flag):
    """
    richiama tutta la pipeline di funzioni in ordine
    output su disco: immagini di dimensioni ridotta
    """
    if not exists(dst):
        makedirs(dst)

    files = get_files_dir(src)
    resized = resize_images(dim, files, src, dst, greyscale_flag)
    return resized


def do_preprocessing(src, dst, greyscale_flag):
    """
    esegue preprocessing della cartella specificata
    """
    print("start do_preprocessing %s" % sys.argv[0])
    img_dimension = 250

    dirs = get_dirs_in_dir(src)
    dirs.pop(0)  # remove first element "./dataset/"

    proc_images = []
    paths = []
    labels = []

    for idx, curr_dir in enumerate(dirs):
        resized_imgs_curr_folder = do_resize(img_dimension, curr_dir, curr_dir.replace(src, dst),
                                                         greyscale_flag)

        for img in resized_imgs_curr_folder:
            paths.append(img["path"])

            """
            if len(proc_images) == 0:
                proc_images = img["data"]  # se è la prima volta, inizializza array con img corrente
            else:
                print(img["data"].shape)
                proc_images = np.vstack((proc_images, img["data"]))  # se non è la prima volta mettili uno sopra l'altro
            """
            proc_images.append(img["data"])
            labels.append(idx)

    proc_images = np.array(proc_images)
    # scaled_images = do_standard_scale(proc_images)
    save_images(proc_images, paths)

    np.save('dataset/processed_images', proc_images)
    np.save('dataset/labels', labels)

    print("end do_preprocessing %s" % sys.argv[0])
