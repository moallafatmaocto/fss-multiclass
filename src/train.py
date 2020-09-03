import os
import random
import time
import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from cnn_encoder import CNNEncoder
from helpers import weights_init, get_oneshot_batch, decode_segmap
from relation_network import RelationNetwork

import warnings

from episode_batch_generator import episode_batch_generator

warnings.filterwarnings("ignore")  # To delete, danger!

print('start training')


def main(finetune: bool, feature_model: str, relation_model: str, learning_rate: int,
         start_episode: int, nbr_episode: int, class_num: int, sample_num_per_class: int,
         batch_num_per_class: int, train_result_path: str, model_save_path: str,
         result_save_freq: int, display_query: int, model_save_freq: int,
         encoder_save_path: str, network_save_path: str, dataset_path: str,data_name:str,pascal_batch:int):
    # Step 1: init neural networks

    print("init neural networks")
    GPU = torch.cuda.device_count()
    print("Number of GPU available:", GPU)

    feature_encoder = CNNEncoder(class_num)
    relation_network = RelationNetwork()

    relation_network.apply(weights_init)
    # Activate GPU:

    if GPU > 0:
        feature_encoder.cuda()
        relation_network.cuda()
        print('Using a GPU')

    # OK
    # fine-tuning: True if using a pretrained model
    if finetune:
        if os.path.exists(feature_model):
            feature_encoder.load_state_dict(torch.load(feature_model))
            print("load feature encoder success")
        else:
            print('starting from scratch')
        if os.path.exists(relation_model):
            relation_network.load_state_dict(torch.load(relation_model))
            print("load relation network success")
        else:
            print('starting from scratch')

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=learning_rate)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=nbr_episode // 10, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=learning_rate)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=nbr_episode // 10, gamma=0.5)

    print("Training...")

    last_accuracy = 0.0
    start = time.time()
    loss_sum = 0
    for episode in range(start_episode, nbr_episode):
        print('episode', episode)
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        # SAMPLE LES IMAGES
        input_support_tensor, input_query_tensor, gt_support_label_tensor, gt_query_label_tensor, chosen_classes = episode_batch_generator(
            class_num, sample_num_per_class, dataset_path, data_name, pascal_batch,train=True)

        print("Feature Encoding ...")
        # calculate features
        if GPU > 0:
            support_features, _ = feature_encoder(Variable(input_support_tensor).cuda())
        else:
            support_features, _ = feature_encoder(Variable(input_support_tensor))

        support_features = support_features.view(class_num, sample_num_per_class, 512, 7, 7)  # N * K * 512 * 7 *7
        support_features = torch.sum(support_features, 1).squeeze(1)  # N * 512 * 7 * 7
        support_features_ext = torch.transpose(support_features.repeat(sample_num_per_class, 1, 1, 1, 1), 0,
                                               1)  # K * N * 512 * 7 *7
        if GPU > 0:
            query_features, ft_list = feature_encoder(Variable(input_query_tensor).cuda())
        else:
            query_features, ft_list = feature_encoder(Variable(input_query_tensor))  # N*K * 512 * 7 *7
        # calculate relations
        query_features_ext = query_features.view(class_num, sample_num_per_class, 512, 7, 7)  # N * K * 512 * 7 *7

        relation_pairs = torch.cat((support_features_ext, query_features_ext), 2).view(-1, 1024, 7,
                                                                                       7)  # flattened N*k*N * 1024 * 7 * 7
        print("Relation Network comparison ...")
        output = relation_network(relation_pairs, ft_list).view(-1, class_num, 224, 224)  # 224 pour le décodeur
        output_ext = output.repeat(class_num, 1, 1, 1)

        if GPU > 0:
            mse = nn.MSELoss().cuda()
            loss = mse(output_ext, Variable(gt_query_label_tensor).cuda())
        else:
            mse = nn.MSELoss()
            loss = mse(output_ext, Variable(gt_query_label_tensor))

        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()
        loss_sum += loss.cpu().data.numpy()
        if (episode + 1) % 10 == 0:
            end = time.time()
            print("After", episode + 1, " episodes, the mean loss =", loss_sum / 10.0)
            print("The Last 10 episodes trained in", end - start, 'mean training time = ', (end - start) / 10.)
            start = time.time()
            loss_sum = 0

        if not os.path.exists(train_result_path):
            os.makedirs(train_result_path)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(encoder_save_path):
            os.makedirs(encoder_save_path)

        print("Results visualization ...")
        # training result visualization
        if (episode + 1) % result_save_freq == 0:
            support_output = np.zeros((224 * 2, 224 * sample_num_per_class, 3), dtype=np.uint8)
            query_output = np.zeros((224 * 3, 224 * display_query, 3), dtype=np.uint8)
            chosen_query = random.sample(list(range(0, batch_num_per_class)), sample_num_per_class)
            for i in range(class_num):
                for j in range(sample_num_per_class):
                    supp_img = (np.transpose(input_support_tensor.numpy()[j], (1, 2, 0)) * 255).astype(np.uint8)[:, :,
                               :3][:, :,
                               ::-1]
                    support_output[0:224, j * 224:(j + 1) * 224, :] = supp_img
                    supp_label = gt_support_label_tensor.numpy()[j][0]
                    supp_label[supp_label != 0] = chosen_classes[i]
                    supp_label = decode_segmap(supp_label)
                    support_output[224:224 * 2, j * 224:(j + 1) * 224, :] = supp_label

                for cnt, x in enumerate(chosen_query):
                    query_img = (np.transpose(input_query_tensor.numpy()[x], (1, 2, 0)) * 255).astype(np.uint8)[:, :,
                                :3][:, :,
                                ::-1]
                    query_output[0:224, cnt * 224:(cnt + 1) * 224, :] = query_img
                    query_label = gt_query_label_tensor.numpy()[x][0]  # only apply to one-way setting
                    query_label[query_label != 0] = chosen_classes[i]
                    query_label = decode_segmap(query_label)
                    query_output[224:224 * 2, cnt * 224:(cnt + 1) * 224, :] = query_label

                    query_pred = output.detach().cpu().numpy()[x][0]
                    query_pred = (query_pred * 255).astype(np.uint8)
                    result = np.zeros((224, 224, 3), dtype=np.uint8)
                    result[:, :, 0] = query_pred
                    result[:, :, 1] = query_pred
                    result[:, :, 2] = query_pred
                    query_output[224 * 2:224 * 3, cnt * 224:(cnt + 1) * 224, :] = result
            cv2.imwrite('%s/%s_query.png' % (train_result_path, episode), query_output)
            cv2.imwrite('%s/%s_support.png' % (train_result_path, episode), support_output)

        print('Episode loss', loss.cpu().data.numpy())
        # save models
        print("Save Models ...")
        if (episode + 1) % model_save_freq == 0:
            torch.save(feature_encoder.state_dict(), str(
                "./%s/feature_encoder_" % encoder_save_path + str(episode) + '_' + str(
                    class_num) + "_way_" + str(
                    sample_num_per_class) + "shot.pkl"))
            torch.save(relation_network.state_dict(), str(
                "./%s/relation_network_" % network_save_path + str(episode) + '_' + str(
                    class_num) + "_way_" + str(
                    sample_num_per_class) + "shot.pkl"))
            print("save networks for episode:", episode)
