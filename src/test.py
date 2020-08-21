import os
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from cnn_encoder import CNNEncoder
from config import FEATURE_MODEL, RELATION_MODEL, FINETUNE, LEARNING_RATE, EPISODE, START_EPISODE, CLASS_NUM, \
    SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, TRAIN_RESULT_PATH, MODEL_SAVE_PATH, RESULT_SAVE_FREQ, DISPLAY_QUERY, \
    MODEL_SAVE_FREQ, QUERY_IMAGES_DIR
from helpers import weights_init, get_oneshot_batch, decode_segmap

from relation_network import RelationNetwork
import warnings

from config import FEATURE_ENCODER_SAVE_PATH, RELATION_NETWORK_SAVE_PATH

from IoU import iou

warnings.filterwarnings("ignore")  # To delete, danger!


# Here import trained models

def main():
    print('Import trained model ...')
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()
    directory = os.getcwd() + '/'

    if os.path.exists(FEATURE_ENCODER_SAVE_PATH):
        feature_encoder_list = os.listdir(FEATURE_ENCODER_SAVE_PATH)
        # print('model_list', feature_encoder_list)
        feature_encoder.load_state_dict(
            torch.load(directory + '/' + FEATURE_ENCODER_SAVE_PATH + '/' + feature_encoder_list[-1]))
        print("load feature encoder success")
    else:
        raise Exception('Can not load feature encoder: %s' % FEATURE_ENCODER_SAVE_PATH)

    if os.path.exists(RELATION_NETWORK_SAVE_PATH):
        relation_network_list = os.listdir(RELATION_NETWORK_SAVE_PATH)
        relation_network.load_state_dict(
            torch.load(directory + '/' + RELATION_NETWORK_SAVE_PATH + '/' + relation_network_list[-1]))
        print("load relation network success")
    else:
        raise Exception('Can not load relation network: %s' % RELATION_NETWORK_SAVE_PATH)

    print("Testing on 5 images...")
    classname = '../new_data/test'
    testnames = os.listdir('%s' % QUERY_IMAGES_DIR)
    stick = np.zeros((224 * 4, 224 * 5, 3), dtype=np.uint8)
    for cnt, testname in enumerate(testnames):
        print('image :', testname)
        if cv2.imread(classname + '/%s' % (testname)) is None:
            print(cv2.imread(classname + '/%s' % (testname)))
            continue

        samples, sample_labels, batches, batch_labels, _ = get_oneshot_batch()
        # forward
        sample_features, _ = feature_encoder(Variable(samples))
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, 512, 7, 7)
        sample_features = torch.sum(sample_features, 1).squeeze(1)  # 1*512*7*7
        batch_features, ft_list = feature_encoder(Variable(batches))
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 1024, 7, 7)
        output = relation_network(relation_pairs, ft_list).view(-1, CLASS_NUM, 224, 224)
        classiou = 0
        for i in range(0, batches.size()[0]):
            # get prediction
            pred = output.data.cpu().numpy()[i][0]
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            # vis
            demo = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB) * 255
            stick[224 * 3:224 * 4, 224 * i:224 * (i + 1), :] = demo.copy()
            # print('batch',batch_labels)
            testlabel = batch_labels.numpy()[i][0]


            # compute IOU
            iou_score = iou(testlabel, pred)
            classiou += iou_score
        # classiou /= 5.0  # because of batch size
        print('Class/image iou:', classiou)


if __name__ == '__main__':
    main()
