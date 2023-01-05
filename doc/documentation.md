## Data loader

The data loader includes the following steps:
- Parse the totality of the dataset into the following structure
    images/images_test
    images/images_train/0
    images/images_train/1
- Unify data type to uint8
- Resize images to (224, 224)

In addition, we included an optional face extraction module in the implementation. This option can be turned on by changing the parameter enable_face_extraction under data_processing in the [**config.txt**](./src/config.txt).
We use yoloface package to detect a bounding box for the facial area and save the cropped area. 
According to [this source](https://arxiv.org/pdf/2006.07084.pdf), by providing the model with more focused input data, the model should be able to better learn the difference between fake and real images.
However, we did not observe meaningful improvement when training our model on this face extraction images.

## Data generator

We prepare the loaded data such that they can be used as arguments for the fit function. 
We used tf.keras.utils.image_dataset_from_directory and opted for a simple train/validation split strategy where we have one fixed validation split.

For the test generator, we store images and their labels in lists that can be referenced using index. 

## Model:

We create two different models, the VGG16 and the ResNet under the src/train folder. The user can change a parameter in the [**config.txt**](./src/config.txt) which would allow them to choose to train a VGG16 or a ResNet (under train -> model_choice).

VGG16 and ResNet are standard cnn model for classification. We first do a data augmentation, add the base model VGG16/ResNet which is set to retrained equal to True and add additional layers such as dense layers and dropout layers to reduce overfitting.

We then train one of the models using the class_weight parameters due to the nature of our imbalanced dataset, an early stoppping as well as a csv logger to save the best model obtained in training. Please refer to the [**README.md**](./src/README.md) to understand how to train a model locally.

## Prediction:

We load the best trained model and infer the probability of each image in the test set. We choose the threshold to be 0.5. If the probability is greater than the threshold then we assign to it label 1, 0 otherwise 9note that the threshold is a parameter that is configured in the config file). We then save the predictions in a csv file in a column that containes the image_names and their predictions.