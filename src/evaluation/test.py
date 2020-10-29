import os
import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from cnn_encoder import CNNEncoder
from helpers import weights_init
from relation_network import RelationNetwork
import warnings
from IoU import iou, positive_areas_union
from episode_batch_generator import episode_batch_generator, get_classnames

warnings.filterwarnings("ignore")  # To delete, danger!


def save_different_masks(class_num, masks):
    """

    :param class_num: number of trained classes
    :param masks: label
    :return: concatenated labels
    """
    new_mask = np.resize(masks[0], (224, 224))
    for class_id in range(1, class_num):
        new_mask = np.concatenate((new_mask, np.resize(masks[class_id], (224, 224))), axis=0)
    return new_mask


def main(test_path: str, class_num: int, sample_num_per_class: int, model_index: int, encoder_save_path: str,
         network_save_path: str, data_name: str, pascal_batch: int, threshold: float, test_result_path: str):
    print('Import trained model and feature encoder ...')
    feature_encoder = CNNEncoder(class_num)
    relation_network = RelationNetwork()
    directory = os.getcwd() + '/'
    model_type = str(class_num) + '_way_' + str(sample_num_per_class) + '_shot_' + str(data_name) + '_' + str(
        pascal_batch) + ".pkl"

    if os.path.exists(encoder_save_path):

        available_feature_encoder_list = os.listdir(encoder_save_path)
        feature_encoder_list = [encoder for encoder in available_feature_encoder_list if
                                model_type in encoder]

        feature_encoder.load_state_dict(
            torch.load(directory + '/' + encoder_save_path + '/' + feature_encoder_list[model_index]))
        print("load feature encoder success")
    else:
        raise Exception('Can not load feature encoder: %s' % encoder_save_path)

    if os.path.exists(network_save_path):
        available_relation_network_list = os.listdir(network_save_path)
        relation_network_list = [network for network in available_relation_network_list if
                                 model_type in network]
        relation_network.load_state_dict(
            torch.load(directory + '/' + network_save_path + '/' + relation_network_list[model_index]))
        print("load relation network success")
    else:
        raise Exception('Can not load relation network: %s' % network_save_path)

    # Refactor  this part like train.py
    print("Testing on 5 images...")
    # Remove old results from folder
    if os.path.exists('test_results'):
        os.system('rm -r test_results')
    # Add new results from folder
    if not os.path.exists('test_results'):
        os.makedirs('test_results')

    GPU = torch.cuda.device_count()
    print("Number of GPU available:", GPU)

    # Init images
    support_image = np.zeros((sample_num_per_class, 3, 224, 224), dtype=np.float32)
    support_label = np.zeros((sample_num_per_class, 1, 224, 224), dtype=np.float32)

    # Get testlist
    class_labels = get_classnames(test_path, data_name, pascal_batch, train=False)
    classiou_list = dict()

    for idx_class, classname in enumerate(class_labels):
        support_tensor, query_tensor, gt_support_label_tensor, gt_query_label_tensor, chosen_classes = episode_batch_generator(
            class_num, sample_num_per_class, test_path, data_name, pascal_batch, train=False)

        print("Feature Encoding ...")
        # calculate features
        support_features, _ = feature_encoder(Variable(support_tensor))
        support_features = support_features.view(class_num, sample_num_per_class, 512, 7, 7)  # N * K * 512 * 7 *7
        support_features = torch.sum(support_features, 1).squeeze(1)  # N * 512 * 7 * 7
        support_features_ext = torch.transpose(support_features.repeat(sample_num_per_class, 1, 1, 1, 1), 0,
                                               1)  # K * N * 512 * 7 *7
        query_features, ft_list = feature_encoder(Variable(query_tensor))
        # Calculate relations
        query_features_ext = query_features.view(class_num, sample_num_per_class, 512, 7, 7)  # N * K * 512 * 7 *7
        relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, 1024, 7,
                                                                                       7)  # flattened N*k*N * 1024 * 7 * 7
        print("Relation Network comparison ...")
        output = relation_network(relation_pairs, ft_list).view(-1, class_num, 224, 224)  # 224 pour le décodeur
        print('output', output.size())
        output_ext = output.repeat(class_num, 1, 1, 1)
        print('output ext', output_ext.size())
        print('gt_query_label_tensor', gt_query_label_tensor.size())
        classiou = 0
        for i in range(sample_num_per_class):
            # get prediction
            if GPU > 0:
                pred = output_ext.data.cuda().cpu().numpy()[i]
                ground_truth_label = gt_query_label_tensor.cuda().cpu().numpy()[i]
            else:
                pred = output_ext.data.cpu().numpy()[i]
                ground_truth_label = gt_query_label_tensor.cpu().numpy()[i]
            print('max values pred', np.max(pred))
            pred = cv2.threshold(pred, threshold, 1, cv2.THRESH_BINARY)[1]
            # pred[pred <= threshold] = 0.
            # pred[pred > threshold] = 1.

            print('Save results for class %s' % classname)
            resized_pred = save_different_masks(class_num, pred)
            print('null prediction ?', np.sum(resized_pred) == 0)
            cv2.imwrite('%s/%s_predicted.png' % (test_result_path, classname),
                        resized_pred)
            resized_ground_truth_label = save_different_masks(class_num, ground_truth_label)
            print('null ground truth ?', np.sum(resized_ground_truth_label) == 0)

            cv2.imwrite('%s/%s_true.png' % (test_result_path, classname),
                        resized_ground_truth_label)

            ## TO DO print prediction vs ground truth (without 0 1)
            # compute IOU
            iou_list_of_score = []
            for class_id in range(class_num):
                iou_list_of_score.append(iou(resized_ground_truth_label[224 * class_id:224 * (class_id + 1)],
                                             resized_pred[224 * class_id:224 * (class_id + 1)]))
            classiou += np.max(iou_list_of_score)
        classiou_list[classname] = classiou / (1.0 * sample_num_per_class)
        print('Mean class iou for %s = %.5f' % (classname, classiou_list[classname]))
    print('Total mean IoU for the dataset %s = %.5f ' % (data_name, np.mean(list(classiou_list.values()))))
