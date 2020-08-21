import math
import os
import random

import cv2
import numpy as np
import torch

from config import SUPPORT_IMAGES_DIR, CLASS_NUM, SAMPLE_NUM_PER_CLASS, QUERY_NUM_PER_CLASS


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def get_oneshot_batch():
    classes_name = os.listdir(SUPPORT_IMAGES_DIR)
    classes = list(range(0, len(classes_name)))
    chosen_classes = random.sample(classes, CLASS_NUM)
    # PRÉPARER DES MATRICES VIDES : SUPPORT IMAGES, SUPPORT LABELS, QUERY IMAGES, QUERY LABELS
    support_images = np.zeros((CLASS_NUM * SAMPLE_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((CLASS_NUM * SAMPLE_NUM_PER_CLASS, CLASS_NUM, 224, 224), dtype=np.float32)
    query_images = np.zeros((CLASS_NUM * QUERY_NUM_PER_CLASS, 3, 224, 224), dtype=np.float32)
    query_labels = np.zeros((CLASS_NUM * QUERY_NUM_PER_CLASS, CLASS_NUM, 224, 224), dtype=np.float32)
    zeros = np.zeros((CLASS_NUM * QUERY_NUM_PER_CLASS, 1, 224, 224), dtype=np.float32)
    class_cnt = 0
    for i in chosen_classes:
        imgnames = os.listdir(f'{SUPPORT_IMAGES_DIR}/{classes_name[i]}/label')
        indexs = list(range(0, len(imgnames)))
        chosen_index = random.sample(indexs, SAMPLE_NUM_PER_CLASS + QUERY_NUM_PER_CLASS)
        j = 0
        for k in chosen_index:
            # process image
            image = cv2.imread(
                f"{SUPPORT_IMAGES_DIR}/{classes_name[i]}/image/{imgnames[k].replace('.png', '.jpg')}")
            if image is None:
                print(f"{SUPPORT_IMAGES_DIR}/{classes_name[i]}/image/{imgnames[k].replace('.png', '.jpg')}")
                break
            image = image[:, :, ::-1]
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            # labels
            label = cv2.imread(f"{SUPPORT_IMAGES_DIR}/{classes_name[i]}/label/{imgnames[k]}")[:, :, 0]
            if j < SAMPLE_NUM_PER_CLASS:
                support_images[j] = image
                support_labels[j][class_cnt] = label
            else:
                query_images[j - SAMPLE_NUM_PER_CLASS] = image
                query_labels[j - SAMPLE_NUM_PER_CLASS][class_cnt] = label
            j += 1

        class_cnt += 1
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat((support_images_tensor, support_labels_tensor), dim=1)

    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor, zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor, chosen_classes










def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        pass
        # plt.imshow(rgb)
        # plt.show()
    else:
        return rgb

