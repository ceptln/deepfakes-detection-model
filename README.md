# DeepFakes Image Classsification on Celeb-DF dataset

## About the Project

The goal of the project is to build a classifier predicting the label of the 3'679 images from the Celeb-DF dataset, with the objective to get the best possible F1 score.

## Installation

The requirements.txt file lists all Python libraries that you will need for this project. You can install these by running:

```pip install -r requirements.txt```

## Usage

The first step here is to create a data folder which you can then add to .gitignore. In this folder, you will have two sub-folders: processed and raw data.

The raw data will contain a file with all the images, a train.csv containing the image names and their label and a test.csv with one column for the image names.

The processed data folder will contain images folder under which there will be an images_train folder in which there will be two folders for each class "0" and "1". The processed data folder will also contain another folder images_test that will contain the images for the test set. These are processed images created by running the main.py file in the src/data_processing file as shown below:

```python3 src/data_processing/main.py```

The second step is to train the model by running the src/main_train.py file by running:

```python3 src/main_train.py```

You can choose between a VGG16 or a ResNet50 pretrained network by changing the input in the config file under train -> model_choice. The config file also contains other parameters related to the data_generator, the models and the train files parameters which can be changed by the user.

The best model will be saved under the file src/models.

The third step is to visualize the confusion matrix and the train, validation loss and accuracy graphs to evaluate the model. In order to do that, you can run the code below:

Loss and Accuracy for Train and Validation sets:
```python3 src/visualizations/metrics.py```

Finally, the last step is to infer the labels for the test set. In order to do that, the user should first choose the model they would like to user under the src/models file and update the corresponding path in the config file under inference -> model_path. The user should then change the file name under inference -> output_path (keeping the path). Then run the following code:

```python3 src/main_inference.py```

The results of the predictions will then be stored under prediction -> output and will contain a columnwith the image names followed by the labels predicted by the model.

## Credits

Authors: Ines Benito, Lea Chader, Jiahe Zhu, Camille Epilaton

## Tests
Tests were added under the tests folder to test the different functions used in src from loading to processing to the model. In order to run the tests, use:

```pytest```

If the tests run properly, the pipeline button above should be green.
