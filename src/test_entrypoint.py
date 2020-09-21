import click

from test import main


@click.command()
@click.option('--test-path', default='../data', type=str)
@click.option('--class-num', '-N', default=1, type=int, required=True)
@click.option('--sample-num-per_class', '-K', default=5, type=int, required=True)
@click.option('--model-index', default=-1, type=int)
@click.option('--encoder-save-path', '-encoder', type=str, default='feature_encoder_trained')
@click.option('--network-save-path', '-network', type=str, default='relation_network_trained')
@click.option('--data-name', type=str, default='FSS', required=True, help='FSS or pascal5i')
@click.option('--pascal-batch', type=int, default=None, help='None or 0,1,2,3 if pascal5i as a dataset')
@click.option('--threshold', default=0.5, type=float)
@click.option('--test-result-path', default='result_newvgg_1shot', type=str)
def entry_point_test(test_path: str, class_num: int, sample_num_per_class: int, model_index: int,
                     encoder_save_path: str, network_save_path: str, data_name: str, pascal_batch: int,
                     threshold: float, test_result_path: str):
    main(test_path, class_num, sample_num_per_class, model_index, encoder_save_path, network_save_path, data_name,
         pascal_batch, threshold, test_result_path)


if __name__ == '__main__':
    entry_point_test()
