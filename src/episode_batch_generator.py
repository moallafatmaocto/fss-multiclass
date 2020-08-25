import os
import random

import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision


def get_random_N_classes(class_list, n):
    classes = list(range(0, len(class_list)))
    chosen_classes = random.sample(classes, n)
    return chosen_classes


def episode_batch_generator(N, K, dataset_path):
    """

    Args: N: Number of classes, N-way, number of classes that the algorithm sees per episode
    K: Number of images per class
    datasetpath: Path of the dataset containing the training set: this folder contains subfolders with the
    class type as a name and for each subfolder we have image.jpg and label.png

    Returns: 4 arguments as the input_support_tensor ( support images + support labels), input_query_tensor (
    query_images + initialized to zero masks) , gt_query_label (query ground truth labels), chosen_classes (N samples
    class names)

    """
    # 1) Select N classes :
    all_classes_names = os.listdir(dataset_path)
    print('N',N,'length_list', len(all_classes_names))
    chosen_classes = get_random_N_classes(all_classes_names, N)

    support_images = np.zeros((K * N, 3, 224, 224), dtype=np.float32)
    support_labels = np.zeros((K * N, N, 224, 224), dtype=np.float32)
    query_images = np.zeros((K * N, 3, 224, 224), dtype=np.float32)
    query_labels_init = np.zeros((K * N, N, 224, 224), dtype=np.float32)
    gt_query_label = np.zeros((K * N, N, 224, 224), dtype=np.float32)

    # 2) For each class of the N chosen classes :

    for class_idx, class_name in enumerate(chosen_classes):

        images_name_list = os.listdir(f'{dataset_path}/{all_classes_names[class_name]}')
        image_names = [class_name for class_name in images_name_list if '.jpg' in class_name]
        sample_support_query_indexes = random.sample(list(range(1, len(image_names) + 1)),
                                                     2 * K)  # sample the same number of images in query and support
        support_indexes = sample_support_query_indexes[:K]
        query_indexes = sample_support_query_indexes[K:]

        # a) Prepare the support set:
        for idx, image_name_dataset in enumerate(support_indexes):

            if not os.path.isfile(
                    dataset_path + '/' + all_classes_names[class_name] + '/' + str(image_name_dataset) + '.jpg'):
                raise Exception(" Support image not found")
            if not os.path.isfile(
                    dataset_path + '/' + all_classes_names[class_name] + '/' + str(image_name_dataset) + '.png'):
                raise Exception(" Support label not found")

            support_image = cv2.imread(
                f"{dataset_path}/{all_classes_names[class_name]}/{str(image_name_dataset) + '.jpg'}")
            if np.shape(support_image)[1] != 224:
                support_image = cv2.resize(support_image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

            corresponding_support_label = cv2.imread(
                f"{dataset_path}/{all_classes_names[class_name]}/{str(image_name_dataset) + '.png'}", 0)
            if np.shape(corresponding_support_label)[1] != 224:
                corresponding_support_label = cv2.resize(corresponding_support_label, dsize=(224, 224),
                                                         interpolation=cv2.INTER_CUBIC)

            support_images[idx + class_idx] = np.transpose(support_image[:, :, ::-1] / 255.0, (2, 0, 1))
            support_labels[idx + class_idx][class_idx] = corresponding_support_label / 255.0

        # b) Prepare the queryset:
        for idx, image_name_dataset in enumerate(query_indexes):
            if not os.path.isfile(
                    dataset_path + '/' + all_classes_names[class_name] + '/' + str(image_name_dataset) + '.jpg'):
                raise Exception(" Query image not found")
            if not os.path.isfile(
                    dataset_path + '/' + all_classes_names[class_name] + '/' + str(image_name_dataset) + '.png'):
                raise Exception(" Query label not found")

            query_image = cv2.imread(
                dataset_path + '/' + all_classes_names[class_name] + '/' + str(image_name_dataset) + '.jpg')
            if np.shape(query_image)[1] != 224:
                query_image = cv2.resize(query_image, dsize=(224, 224),
                                         interpolation=cv2.INTER_CUBIC)

            gt = cv2.imread(dataset_path + '/' + all_classes_names[class_name] + '/' + str(image_name_dataset) + '.png',
                            0)
            if np.shape(gt)[1] != 224:
                gt = cv2.resize(gt, dsize=(224, 224),
                                interpolation=cv2.INTER_CUBIC)

            query_images[idx + class_idx - K] = np.transpose(query_image[:, :, ::-1] / 255.0, (2, 0, 1))
            gt_query_label[idx + class_idx - K][class_idx] = gt / 255.0

    # 3) concat images and labels, save ground_truth labels(gt)
    input_support_tensor = torch.cat((torch.from_numpy(support_images), torch.from_numpy(support_labels)), dim=1)
    gt_support_label_tensor = torch.from_numpy(support_labels)

    input_query_tensor = torch.cat((torch.from_numpy(query_images), torch.from_numpy(query_labels_init)), dim=1)
    gt_query_label_tensor = torch.from_numpy(gt_query_label)
    return input_support_tensor, input_query_tensor, gt_support_label_tensor, gt_query_label_tensor, chosen_classes
