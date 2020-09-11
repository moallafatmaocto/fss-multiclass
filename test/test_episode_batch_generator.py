from unittest import TestCase
from unittest.mock import patch
import numpy as np

from src.episode_batch_generator import get_random_N_classes, generate_images_and_masks_query_and_support


class TestEpisodeGenerator(TestCase):
    def test_get_random_N_classes_returns_random_N_classes(self):
        # Given
        class_list = ['cat', 'dog', 'rabbit', 'zebra', 'lion', 'bird']

        # When
        random_N_classes = get_random_N_classes(class_list, n=2)

        # Then
        self.assertEqual(len(random_N_classes), 2)

    @patch('src.episode_batch_generator.get_support_query_indexes_per_class')
    @patch('src.episode_batch_generator.get_image_and_corresponding_mask')
    def test_generate_images_and_masks_returns_correct_tensors(self, get_image_and_corresponding_mask_mocked,
                                                               get_support_query_indexes_per_class_mocked):
        # Given
        get_support_query_indexes_per_class_mocked.side_effect = [([0], [1], ['1.jpg', '2.jpg']),
                                                                  ([2], [3], ['5.jpg', '3.jpg', '3.jpg', '4.jpg'])]
        get_image_and_corresponding_mask_mocked.side_effect = [
            (np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]]).reshape((3, 3, 1)),
             np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])),
            (np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]]).reshape((3, 3, 1)),
             np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])),
            (np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]]).reshape((3, 3, 1)),
             np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]])),
            (np.array([[510, 510, 510], [510, 510, 510], [510, 510, 510]]).reshape((3, 3, 1)),
             np.array([[255, 255, 255], [255, 255, 255], [255, 255, 255]]))]
        chosen_classes = {1: 'dog', 5: 'bird'}
        data_name = "FSS"
        dataset_path = '../data'

        # When
        gt_query_label, query_images, query_labels_init, support_images, support_labels = generate_images_and_masks_query_and_support(
            K=1, N=2, chosen_classes=chosen_classes, data_name=data_name,
            dataset_path=dataset_path, pascal_batch=None, train=True, channel=1, dimension=3)

        # Then
        self.assertEqual(np.sum(gt_query_label[0,0]), 9)
        self.assertEqual(np.sum(gt_query_label[0,1]), 0)
        self.assertEqual(np.sum(gt_query_label[1,0]), 0)
        self.assertEqual(np.sum(gt_query_label[1,1]), 9)
        self.assertEqual(np.sum(query_images[0]), 9)
        self.assertEqual(np.sum(query_images[1]), 18)
        # self.assertEqual(query_labels_init.all(), 0)
        # self.assertEqual(support_images.all(), 0)
        # self.assertEqual(support_labels.all(), 0)
