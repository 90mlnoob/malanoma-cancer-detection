# Melanoma Skin Cancer Detection

## Overview

Melanoma is the deadliest type of skin cancer, accounting for 75% of skin cancer-related deaths. Early detection is critical, as it significantly improves treatment outcomes. Traditional diagnosis involves clinical screening, dermoscopic analysis, and histopathological examination. Dermatologists use dermatoscopic images for initial assessments, achieving diagnostic accuracies of 65% to 80%. With additional evaluations by oncologists and advanced analysis, the accuracy improves to 75% to 84%.

This project aims to develop an automated classification system using image processing and deep learning techniques to classify skin lesions and assist in the early detection of melanoma.

## Problem Statement

Melanoma can be fatal if not detected early. The goal of this project is to build a convolutional neural network (CNN)-based model capable of accurately detecting melanoma from skin lesion images. Automating this process can significantly reduce the manual effort required by dermatologists and enhance diagnostic precision.

## Dataset

The dataset contains 2,357 images representing malignant and benign skin conditions. These images were obtained from the International Skin Imaging Collaboration (ISIC) and are evenly distributed across categories based on ISIC's classification standards.

To address class imbalance, the dataset was augmented using the [Augmentor Python package](https://augmentor.readthedocs.io/en/master/). This process generated additional samples for underrepresented classes, ensuring balanced representation across the dataset.

## Model Architecture

The following steps outline the design of the CNN architecture used in this project:

1. **Data Augmentation**  
   Data augmentation techniques are applied to the training data to increase its diversity. Transformations such as rotation, scaling, and flipping are implemented using the `augmentation_data` variable. This enhances the model's ability to generalize to unseen data.

2. **Normalization**  
   A `Rescaling(1./255)` layer is used to normalize pixel values of the input images to the range [0, 1]. This helps stabilize the training process and speeds up convergence.

3. **Convolutional Layers**  
   Three convolutional layers are added using the `Conv2D` function. Each layer uses a rectified linear unit (ReLU) activation function to introduce non-linearity. The layers have 16, 32, and 64 filters, respectively, with `padding='same'` to maintain spatial dimensions.

4. **Pooling Layers**  
   Max-pooling layers (`MaxPooling2D`) follow each convolutional layer to downsample feature maps, reducing spatial dimensions while retaining essential information. This helps in reducing computational complexity and mitigating overfitting.

5. **Dropout Layer**  
   A `Dropout` layer with a 0.2 dropout rate is added to prevent overfitting by randomly deactivating neurons during training.

6. **Flatten Layer**  
   The `Flatten` layer converts the 2D feature maps into a 1D vector, preparing the data for the fully connected layers.

7. **Fully Connected Layers**  
   Two dense layers are added. The first layer contains 128 neurons with ReLU activation, and the second outputs classification probabilities for each class label.

8. **Output Layer**  
   The output layer has neurons equal to the number of classes (`target_labels`). It outputs logits, which are used in the loss calculation during training.

9. **Model Compilation**  
   The model is compiled with the Adam optimizer (`optimizer='adam'`) and Sparse Categorical Crossentropy loss function (`loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`), suitable for multi-class classification. Accuracy is used as the evaluation metric (`metrics=['accuracy']`).

10. **Training**  
    The model is trained for 20 epochs initially and later with 30 epochs using the `fit` method as per the recommendation in the Case Study. The callbacks implemented is below:
    - **EarlyStopping**: Halts training if validation accuracy does not improve for five consecutive epochs (`patience=5`).

These steps ensure efficient training, minimize overfitting, and optimize the model's performance for classifying skin cancer images.

## Collaborators

Created by Saurav Suman - [@90mlnoob](https://github.com/90mlnoob)
