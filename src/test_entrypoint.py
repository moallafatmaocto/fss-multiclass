import click

from test import main


@click.command()
@click.option('--test-path', default='../dataset_test', type=str, help='Unseen images to test the dataset')
@click.option('-- class-num', '-N', default=1, type=int, help='number of different classes of the trained model')
@click.option('-- sample-num-per_class', '-K', default=5, type=int, help='number of images per class')
@click.option('-- model-index', default=-1, type=int, help='Negative number describing which pretrained model to take')
@click.option('--encoder-save-path', type=str, default='feature_encoder_trained',
              help='Path where the feature encoders are saved after training')
@click.option('--network-save-path', type=str, default='relation_network_trained',
              help='Path where the relation networks are saved after training')
def entry_point_test(test_path: str, class_num: int, sample_num_per_class: int, model_index: int,
                     encoder_save_path: str, network_save_path: str):
    main(test_path, class_num, sample_num_per_class, model_index, encoder_save_path, network_save_path)


if __name__ == '__main__':
    entry_point_test()
