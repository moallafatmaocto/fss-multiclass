# FSS-multiclass : Using the Encoder - RelationNetwork method
Inspired repo from the FSS-1000 paper. This is a metric-learning solution to segment a dataset of labeled images using only few number of images per class.

## Environment setup 

1) First, set-up the conda environment with python >=3.7 
2) Activate the new envirnment 
```
conda activate new_env python=3.7
```
3) Install the required packages:
```
pip install requirements.txt
```
or 
```
conda install requirements.txt
```


## Training phase: (I advise to run it on a GPU if episodes > 20)

### The FSS-dataset

The FSS dataset is a 1000 class dataset with 10 images per class described in **the original paper**. In order to train on this dataset: 

1) Split dataset into train and test (760 for train / 240 for test):
```
python split_train_test_dataset.py
```
This function will change the structure of the dataset: i.e data --> train --> elephant --> 1.jpg

2) Train the model on this dataset on 1000 episode in the case of 5-shot 1-way
```
python relationnetencoder_entrypoint.py -N 1 -K 5 -episode 1000 --result-save-freq 100 --model-save-freq 100 -batch 5 --data-name 'FSS' --dataset-path '../data'

```
The loss that we use to train the encoder-relationnetwork is the crossentropy loss
### The Pascal-5i dataset
The pascal5i dataset is generated from the dataset PAscal VOC 2012 using **this script**. It is constituted by 4 batches (0,1,2,3) and each batch contains 5 differents classes and hundreds of images. The train and test sets for each batch are the same in this case. No need to resplit the dataset, we will use it in the same way as in the litterature and the original paper **OSLSM**.

1) Train the model by specifying the pascal batch number (0,1,2,3):
```
python relationnetencoder_entrypoint.py -N 1 -K 5 -episode 1000 --result-save-freq 100 --model-save-freq 100 -batch 5 --data-name 'pascal5i' --pascal-batch 0 --dataset-path '../data_pascal5i/'
```
## Testing phase: 
In this phase , we will evaluate the images of the test set using the meanIoU metric which calculates the average internsection over union between the gound truth mask and the predicted mask per class.

### The FSS-dataset
3) Test the model on the testing set( not the same classes as the training set, generalization to unseen classes) 
```
python test_entrypoint.py -N 1 -K 5 --data-name 'FSS' 
```
### The Pascal-5i dataset
2) Test the model on the same pascal batch and with the same parameters, in this case the training and testing classes are the same:
```
python test_entrypoint.py -N 1 -K 5 --data-name 'pascal5i' --pascal-batch 0 --test-path '../data_pascal5i' 
```

## Example 

images + results soon

