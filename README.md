### Image Localization using a multi-output CNN for classification & bbox regression

In this project I used a synthetic dataset based on MNIST in order to train a model for the localization task.  
The dataset can be found here https://github.com/ayulockin/synthetic_datasets/tree/master.  

Image localization is a combined task in which the final model should predict a class label, in this case numbers, as well as detect the object inside the picture with a bounding box (bbox).

I chose to use Resnet18 as a base model, strip the last fc layer from the model and use the feature vector produced by the CNN as input for two components:  
* Softmax classifier to detect class labels.
* Fully-connected NN to preform a regression that would output 4 numbers to construct a bbox.

The bbox follows the convention of (x1, y1, x2, y2) where (x1, y1) is the upper-left corner and (x2, y2) is the bottom-right corner, can be also shown as (x_min, y_min, x_max, y_max).

In order to quantify the bbox regression performance I used the IoU (intersection over union) metric, which is also used as a loss function for the regression part of the model.

Below you can observe figures of the model architecture, test samples with inference & TB graphs that were produced during training.

<p align="center">
<img src="https://github.com/matfain/Image-Localization-MNIST/assets/132890076/9e5afa27-576b-4254-a3c8-b22554ebd958" width="500" height="800">      <img src="https://github.com/matfain/Image-Localization-MNIST/assets/132890076/87aff3ca-3514-4743-ad21-93d08df21296" width="340" height="780">
<img src="https://github.com/matfain/Image-Localization-MNIST/assets/132890076/5cfafb29-128f-4321-9bd9-f430c067c3e7" width="363" height="244">      <img src="https://github.com/matfain/Image-Localization-MNIST/assets/132890076/92a1f488-0794-453a-a898-8e065e6fe591" width="363" height="244">
<img src="https://github.com/matfain/Image-Localization-MNIST/assets/132890076/2b60b9f3-cd3c-48f2-ac59-c430c3cf00ce" width="363" height="244"> 
</p>
