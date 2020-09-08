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


# Here import trained models

def get_class_names_from_test_folder(files_list):
    class_list = []
    for name in files_list:
        if '.png' in name:
            position = name.find('_', name.find('_') + 1)
            class_list.append(name[position + 1:-4])
    return class_list


def main(test_path: str, class_num: int, sample_num_per_class: int, model_index: int, encoder_save_path: str,
         network_save_path: str, data_name: str, pascal_batch: int):
    print('Import trained model and feature encoder ...')
    feature_encoder = CNNEncoder(class_num)
    relation_network = RelationNetwork()
    directory = os.getcwd() + '/'
    model_type = str(class_num) + '_way_' + str(sample_num_per_class) + '_shot_' + str(data_name) + '_' + str(
        pascal_batch) + ".pkl"

    if os.path.exists(encoder_save_path):

        available_feature_encoder_list = os.listdir(encoder_save_path)
        #print(encoder_save_path,'available_feature_encoder_list',available_feature_encoder_list)
        feature_encoder_list = [encoder for encoder in available_feature_encoder_list if
                                model_type in encoder]
        #print('feature_encoder_list', feature_encoder_list )

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
    # Set results folder:
    ## Remove old results from folder
    if os.path.exists('test_results'):
        os.system('rm -r test_results')
    ## Add new results from folder
    if not os.path.exists('test_results'):
        os.makedirs('test_results')

    GPU = torch.cuda.device_count()
    print("Number of GPU available:", GPU)

    # Init images
    support_image = np.zeros((sample_num_per_class, 3, 224, 224), dtype=np.float32)
    support_label = np.zeros((sample_num_per_class, 1, 224, 224), dtype=np.float32)
    stick = np.zeros((224 * 4, 224 * 5, 3), dtype=np.uint8)

    # Get testlist
    # testnames = os.listdir('%s' % test_path)
    # class_labels = get_class_names_from_test_folder(testnames)

    class_labels = get_classnames(test_path, data_name, pascal_batch,train=False)
    print('Testing images in class:', class_labels)
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
        # calculate relations
        query_features_ext = query_features.view(class_num, sample_num_per_class, 512, 7, 7)  # N * K * 512 * 7 *7
        relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, 1024, 7,
                                                                                       7)  # flattened N*k*N * 1024 * 7 * 7
        print("Relation Network comparison ...")
        output = relation_network(relation_pairs, ft_list).view(-1, class_num, 224, 224)  # 224 pour le décodeur
        output_ext = output.repeat(class_num, 1, 1, 1)

        classiou = 0
        for i in range(sample_num_per_class):
            # get prediction
            if GPU > 0:
                print('Running on GPU')
                pred = output_ext.data.cuda().cpu().numpy()[i]
            else:
                pred = output_ext.data.cpu().numpy()[i]
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            print('pred', pred)
            # vis
            demo = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB) * 255
            stick[224 * 3:224 * 4, 224 * i:224 * (i + 1), :] = demo.copy()
            # compute IOU
            print('gt', gt_query_label_tensor.numpy()[i])
            print('union', positive_areas_union(gt_query_label_tensor.numpy()[i], pred))
            iou_score = iou(gt_query_label_tensor.numpy()[i], pred)
            classiou += iou_score
        classiou_list[classname] = classiou / (1.0*sample_num_per_class)
        print('Mean class iou for', classname, ' = ', classiou_list[classname])
    print('Total mean IoU for the dataset ',data_name, ' = ', np.mean(list(classiou_list.values())))

# if __name__ == '__main__':
#   main()
