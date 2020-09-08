from unittest import TestCase

from src.split_train_test_dataset import get_random_n_classes, get_train_and_test_image_lists


class TestSplitTrainTestDatasets(TestCase):

    def test_get_random_n_classes_should_return_a_list_of_n_items(self):
        # Given
        classes = ['orange', 'apple', 'tomato', 'banana']
        n = 3

        # When
        chosen_classes = get_random_n_classes(classes, n)

        # Then
        self.assertEqual(len(chosen_classes), n)

    def test_get_random_n_classes_should_raise_ValueError_when_sample_size_is_larger_than_population_size(self):
        # Given
        classes = ['orange', 'apple', 'tomato']
        n = 4

        # Then
        self.assertRaises(ValueError, get_random_n_classes, classes, n)

    def test_get_train_and_test_image_lists_should_return_two_shuffled_lists(self):
        # Given
        classes = ['orange', 'apple', 'tomato', 'banana', 'potato']
        train_classes_number = 3

        # When
        train_classes, test_classes = get_train_and_test_image_lists(classes, train_classes_number)

        # Then
        self.assertEqual(len(train_classes), train_classes_number)
        self.assertEqual(len(test_classes), 2)
