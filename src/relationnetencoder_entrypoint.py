import click

from train import main


@click.command()
@click.option('--finetune', default=True, type=bool, help='Finetuning the results')
@click.option('--feature-model', default='', type=str, help='Path of the pre-trained feature model if it exists')
@click.option('--relation-model', default='', type=str, help='Path of the pre-trained relation model if it exists')
@click.option('--learning-rate', '-lr', default=0.001, type=float, help='Learning rate of the optimiser')
@click.option('--start-episode', '-start', default=0, type=int, help='Start episode when training')
@click.option('--nbr-episode', '-episode', default=10, type=int, help='Number of episodes when training')
@click.option('--class-num', '-N', type=int, required=True, help='Number of classes to train, i.e N-way')
@click.option('--sample-num-per_class', '-K', type=int, required=True,
              help='Number of images per class to train , i.e K-shot')
@click.option('--batch-num-per_class', '-batch', type=int, default=3, help='Number of batches per image')
@click.option('--train-result-path', type=str, default='result_newvgg_1shot', help='Path of the results after training')
@click.option('--model-save-path', type=str, default='relation_network_trained',
              help='Path of the relation network after training')
@click.option('--result-save-freq', type=int, default=10, help='frequency of saving the results')
@click.option('--model-save-freq', type=int, default=10, help='frequency of saving the model')
@click.option('--display-query', type=int, default=5, help='Number of test displayed')
@click.option('--encoder-save-path', type=str, default='feature_encoder_trained',
              help='Path to save the encoder after training')
@click.option('--network-save-path', type=str, default='relation_network_trained',
              help='Path to save the relation network after training')
@click.option('--dataset-path', type=str, default='../data',
              help='Path to the dataset(containing sub-folders of different classes)')
def entry_point(finetune: bool, feature_model: str, relation_model: str, learning_rate: int,
                start_episode: int, nbr_episode: int, class_num: int, sample_num_per_class: int,
                batch_num_per_class: int, train_result_path: str, model_save_path: str,
                result_save_freq: int, display_query: int, model_save_freq: int,
                encoder_save_path: str, network_save_path: str, dataset_path: str):
    main(finetune, feature_model, relation_model, learning_rate,
         start_episode, nbr_episode, class_num,
         sample_num_per_class,
         batch_num_per_class, train_result_path, model_save_path,
         result_save_freq, display_query, model_save_freq,
         encoder_save_path, network_save_path, dataset_path)


if __name__ == '__main__':
    entry_point()
