import os, shutil
import random
from typing import Tuple, List


def get_random_n_classes(class_list: List, n: int) -> List:
    classes = list(range(0, len(class_list)))
    print('total number of initial classes', len(classes))
    chosen_classes = random.sample(classes, n)
    return chosen_classes


def get_train_and_test_image_lists(class_list: List, train_class_number: int) -> Tuple[List, List]:
    print('class_list', class_list)
    chosen_train_classes = [class_list[idx] for idx in (get_random_n_classes(class_list, train_class_number))]
    print('Number of train classes', len(chosen_train_classes))
    chosen_test_classes = [class_name for class_name in class_list if (class_name not in (chosen_train_classes))]
    print('Number of test classes', len(chosen_test_classes), chosen_test_classes)

    return chosen_train_classes, chosen_test_classes


def create_images_directories(base_path: str, directory: str) -> str:
    path = base_path + directory
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def move_to_new_dir(classes, original_path, new_path):
    print('Start Moving ...')
    for class_name in classes:
        print('Moving class ', class_name)
        print('current', os.getcwd(), 'original', original_path)
        shutil.move(original_path + class_name, new_path)
    print('Moving ended.')


if __name__ == '__main__':
    base_path = '../data/'
    train_classes_number = 760
    class_list = os.listdir(base_path)
    train_classes, test_classes = get_train_and_test_image_lists(class_list, train_classes_number)
    train_path = create_images_directories(base_path, 'train')
    test_path = create_images_directories(base_path, 'test')
    move_to_new_dir(train_classes, base_path, train_path)
    move_to_new_dir(test_classes, base_path, test_path)
