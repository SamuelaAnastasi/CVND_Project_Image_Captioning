# CVND Project: Image Captioning

This is the second project of the Udacity's Computer Vision Nanodegree. The project will create a neural network architecture to automatically generate captions from images.
After using the Microsoft Common Objects in COntext (MS COCO) dataset to train the network, we will test it on novel images!

### 1. The Dataset

The Microsoft Common Objects in COntext (MS COCO) dataset is a large-scale dataset for
scene understanding. The dataset is commonly used to train and benchmark object detection,
segmentation, and captioning algorithms.

### 2. CNN-LSTM Architecture

The model is build of two main parts: An Encoder and a Decoder:

On **Encoder** side we have a pretrained net (Resnet50) with weights frozen and last fully connected layer replaced by a new one to be trained and output a fixed-size image feature vector.

The **Decoder** is built by 1-layer LSTM cells and a final fully connected layer which outputs for each step the most probable word from the vocabulary and feeds it back as an input for the next time step. At step 0 an input vector with features and captions embeddings concatenated is provided to the network.

### 3. Tuning Hyperparameters and Training
To figure out how well the model is doing, we can look at how the training loss and perplexity
evolve during training and for the purposes of this project, we need to amend the
hyperparameters to improve accuracy.

The model is trained for only 3 epochs, given limited time. A hidden size and embed size of 512 is used and from the vocabulary words that appears less than 5 times are excluded reducing the number of unique words present.
After the a neural network is trained, we can then apply it to test dataset to generate new captions.
### 4. Image Captioning Inference Sampling
For the sampling we first load the saved weights for the Encoder and the Decoder, pass through the net the selected image and after generating the caption we clean it up to exclude the `<start>` and `<end>` tokens.

Although trained for a very short time the model generally performs quite well, generating captions that truthfully describe the content of the images.

![Image Captions Good Performance](https://raw.githubusercontent.com/SamuelaAnastasi/CVND_Project_Image_Captioning/master/images/inference_g.jpg)

Even when the results are not so satisfactory and the model could have performed better, it's still able to detect actual objects, features and relationships between the objects in the selected images.

![Image Captions Low Performance](https://raw.githubusercontent.com/SamuelaAnastasi/CVND_Project_Image_Captioning/master/images/inference_b.jpg)
