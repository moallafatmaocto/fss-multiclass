from unittest import TestCase

from src.episode_batch_generator import get_random_N_classes


class TestEpisodeGenerator(TestCase):
    def test_get_random_N_classes_returns_random_N_classes(self):
        # Given
        class_list = ['cat', 'dog', 'rabbit', 'zebra', 'lion', 'bird']

        # When
        random_N_classes = get_random_N_classes(class_list, n=2)

        # Then
        self.assertEqual(len(random_N_classes), 2)
