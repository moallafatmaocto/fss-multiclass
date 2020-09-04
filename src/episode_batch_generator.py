import os
import random
from typing import List, Any

import pandas as pd
import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision


def get_random_N_classes(class_list, n):
    """
    :param class_list: a list of all class names
    :param n: number of random classes to select
    :return: a list of n random class names
    """
    classes = list(range(0, len(class_list)))
    chosen_classes = random.sample(classes, n)
    return chosen_classes


def init_tensors(K, N):
    """
    :param K: number of samples per class (K-shot)
    :param N: number of different classes (N-way)
    :return: initial : support support_images, support_labels, query_images, query_labels, gt_query_label
    """
    support_images = np.zeros((K * N, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((K * N, N, 224, 224), dtype=np.float32)
    query_images = np.zeros((K * N, 3, 224, 224), dtype=np.float32)
    query_labels_init = np.zeros((K * N, N, 224, 224), dtype=np.float32)
    gt_query_label = np.zeros((K * N, N, 224, 224), dtype=np.float32)
    return support_images, support_labels, query_images, query_labels_init, gt_query_label


def get_classnames(dataset_path, data_name='FSS', pascal_batch=None, train=True):
    """
    :param dataset_path: /data/
    :param data_name: 'FSS' or 'pascal5i'
    :param train: True or False
    :param pascal_batch: None or 0,1,2,3
    :return: list of al classnames of the dataset considered for training (if True) or testing otherwise
    """
    if train:
        process_type = 'train'
    else:
        process_type = 'test'

    if data_name == 'FSS':
        all_class_names = os.listdir(f'{dataset_path}/{process_type}/')

    if data_name == 'pascal5i':
        class_list = os.listdir(f'{dataset_path}/{pascal_batch}/{process_type}/')
        all_class_names = [class_name[:-4] for class_name in class_list if '.txt' in class_name]
    return all_class_names


def get_support_query_indexes_per_class(K, class_name, dataset_path, data_name='FSS', train=True, pascal_batch=None):
    """
    :param K: number of images per class (K-way)
    :param class_name: string can be 'dog', 'elephant' ...
    :param dataset_path: /data/
    :param data_name: FSS or pascal5i
    :param train: True or False
    :param pascal_batch: None or 0,1,2,3
    :return: two lists of support and query indexes
    """
    if train:
        process_type = 'train'
    else:
        process_type = 'test'

    if data_name == 'FSS':
        images_name_list = os.listdir(f'{dataset_path}/{process_type}/{class_name}')
        image_names = [class_name for class_name in images_name_list if '.jpg' in class_name]

    elif data_name == 'pascal5i':
        file_path = f'{dataset_path}/{str(pascal_batch)}/{process_type}/{class_name}.txt'
        with open(file_path, encoding="utf-8") as file:
            image_names = [l.rstrip("\n") for l in file]

    sample_support_query_indexes = random.sample(list(range(0, len(image_names))),
                                                 2 * K)  # sample the same number of images in query and support
    support_indexes = sample_support_query_indexes[:K]
    query_indexes = sample_support_query_indexes[K:]
    return support_indexes, query_indexes, image_names


def get_image_and_corresponding_mask(data_path, image_name, class_name, data_name='FSS', train=True, pascal_batch=None):
    """
    :param data_path: /data/
    :param image_name: example '1.jpg'
    :param class_name: example 'elephant'
    :param data_name: 'FSS' or 'pascal5i'
    :param train: True or False
    :param pascal_batch: None or 0,1,2,3
    :return: image and corresponding mask (np.ndarray)
    """
    if train:
        process_type = 'train'
    else:
        process_type = 'test'

    if data_name == 'FSS':
        image_file_path = f'{data_path}/{process_type}/{class_name}/{str(image_name)}'
        mask_file_path = f'{data_path}/{process_type}/{class_name}/{str(image_name)}'

    if data_name == 'pascal5i':
        image_file_path = f'{data_path}/{str(pascal_batch)}/{process_type}/origin/{image_name}.jpg'
        mask_file_path = f'{data_path}/{str(pascal_batch)}/{process_type}/groundtruth/{image_name}.jpg'
    if not os.path.isfile(image_file_path):
        raise Exception(" Image not found")
    if not os.path.isfile(mask_file_path):
        raise Exception(" Mask not found")
    image = cv2.imread(image_file_path)
    mask = cv2.imread(mask_file_path, 0)

    # Resize
    if np.shape(image)[1] != 224:
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    if np.shape(mask)[1] != 224:
        mask = cv2.resize(mask, dsize=(224, 224),
                          interpolation=cv2.INTER_CUBIC)
    return image, mask


def episode_batch_generator(N, K, dataset_path, data_name='FSS', pascal_batch=None, train=True):
    """
    :param K: number of samples per class (K-shot)
    :param N: number of different classes (N-way)
    :param dataset_path: /data/
    :param train: True or False
    :param pascal_batch:
    :return: Returns: 5 arguments as the input_support_tensor ( support images + support labels), input_query_tensor (
    query_images + initialized to zero masks) , gt_query_label (query ground truth labels), chosen_classes (N samples
    class names)

    """
    # 0) Init tensors:
    support_images, support_labels, query_images, query_labels_init, gt_query_label = init_tensors(K, N)
    # 1) Select N classes randomly:
    all_classes_names = get_classnames(dataset_path, data_name, pascal_batch, train)
    chosen_classes = get_random_N_classes(all_classes_names, N)

    # 2) For each class of the N chosen classes :

    for class_idx, class_number in enumerate(chosen_classes):
        class_name = all_classes_names[class_number]
        support_indexes, query_indexes, image_names = get_support_query_indexes_per_class(K, class_name, dataset_path,
                                                                                          data_name, train,
                                                                                          pascal_batch)
        for idx, image_number in enumerate(support_indexes):
            # function get mask and image
            image_name = image_names[image_number]
            support_image, corresponding_support_label = get_image_and_corresponding_mask(dataset_path, image_name,
                                                                                          class_name, data_name,
                                                                                          train,
                                                                                          pascal_batch)
            support_images[idx + class_idx] = np.transpose(support_image[:, :, ::-1] / 255.0, (2, 0, 1))
            support_labels[idx + class_idx][class_idx] = corresponding_support_label / 255.0

        for idx, image_number in enumerate(query_indexes):
            image_name = image_names[image_number]
            query_image, gt = get_image_and_corresponding_mask(dataset_path, image_name,
                                                               class_name, data_name,
                                                               train,
                                                               pascal_batch)

            query_images[idx + class_idx - K] = np.transpose(query_image[:, :, ::-1] / 255.0, (2, 0, 1))
            gt_query_label[idx + class_idx - K][class_idx] = gt / 255.0

    # 3) concat images and labels, save ground_truth labels(gt)
    input_support_tensor = torch.cat((torch.from_numpy(support_images), torch.from_numpy(support_labels)), dim=1)
    gt_support_label_tensor = torch.from_numpy(support_labels)

    input_query_tensor = torch.cat((torch.from_numpy(query_images), torch.from_numpy(query_labels_init)), dim=1)
    gt_query_label_tensor = torch.from_numpy(gt_query_label)
    return input_support_tensor, input_query_tensor, gt_support_label_tensor, gt_query_label_tensor, chosen_classes
