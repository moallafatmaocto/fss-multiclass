FEATURE_DIM = 64
RELATION_DIM = 8
CLASS_NUM = 1
# change to 1 and then 10 , add 5 extra classes, and calculate iou
SAMPLE_NUM_PER_CLASS = 5
QUERY_NUM_PER_CLASS = 5
EPISODE = 10
TEST_EPISODE = 7
# Mettre learning rate à 0.005
LEARNING_RATE = 0.005
HIDDEN_UNIT = 1
DISPLAY_QUERY = 5
EXCLUDE_CLASS = 6
FEATURE_MODEL = ''
RELATION_MODEL = ''
FINETUNE = True
START_EPISODE = 0

SUPPORT_IMAGES_DIR = '../new_data/support'
QUERY_IMAGES_DIR = '../new_data/test'
TRAIN_RESULT_PATH = 'result_newvgg_1shot'
MODEL_SAVE_PATH = 'relation_network_trained'
RELATION_NETWORK_SAVE_PATH = 'relation_network_trained'
FEATURE_ENCODER_SAVE_PATH = 'feature_encoder_trained'
RESULT_SAVE_FREQ = 1
MODEL_SAVE_FREQ = 1
