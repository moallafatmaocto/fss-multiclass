import os, shutil
import random

train_category_number = 760
test_category_number = 240


def get_random_N_classes(class_list, n):
    classes = list(range(0, len(class_list)))
    print('total number of initial classes', len(classes))
    chosen_classes = random.sample(classes, n)
    return chosen_classes

# Set up directory
data_path = 'data/'

if not os.path.exists('../data/train'):
    os.mkdir('../data/train')
if not os.path.exists('../data/test'):
    os.mkdir('../data/test')
original_data_path = '../data/'
new_path_train = original_data_path + 'train'
new_path_test = original_data_path + 'test'

# Determine randomly the dataset
class_list = os.listdir('../data/')
print('class_list', class_list)
class_list.remove('train')
class_list.remove('test')
class_list.remove('.DS_Store')
chosen_train_classes_idx = get_random_N_classes(class_list, train_category_number)
chosen_train_classes = [class_list[idx] for idx in chosen_train_classes_idx]
print('Number of train classes', len(chosen_train_classes))
chosen_test_classes = [class_name for class_name in class_list if
                       (class_name not in (chosen_train_classes) and class_name != 'test' and class_name != 'train')]

# chosen_test_classes = [ class_name for class_name in [0,1,2,3,4] if class_name not in [0,1]]
print('Number of test classes', len(chosen_test_classes), chosen_test_classes)


def move_to_new_dir(classes, original_path, new_path):
    print('Start Moving ...')
    for class_name in classes:
        print('Moving class ', class_name)
        print('current', os.getcwd(), 'original', original_path)
        shutil.move(original_path + class_name, new_path)
    print('Moving ended.')


move_to_new_dir(chosen_train_classes, original_data_path, new_path_train)
move_to_new_dir(chosen_test_classes, original_data_path, new_path_test)
