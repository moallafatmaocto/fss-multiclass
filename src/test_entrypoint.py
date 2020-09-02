import click

from test import main


@click.command()
@click.option('--test-path', default='../dataset_test', type=str)
@click.option('--class-num', '-N', default=1, type=int, required=True)
@click.option('--sample-num-per_class', '-K', default=5, type=int, required=True)
@click.option('--model-index', default=-1, type=int)
@click.option('--encoder-save-path', '-encoder', type=str, default='feature_encoder_trained')
@click.option('--network-save-path', '-network', type=str, default='relation_network_trained')
def entry_point_test(test_path: str, class_num: int, sample_num_per_class: int, model_index: int,
                     encoder_save_path: str, network_save_path: str):
    main(test_path, class_num, sample_num_per_class, model_index, encoder_save_path, network_save_path)


if __name__ == '__main__':
    entry_point_test()
