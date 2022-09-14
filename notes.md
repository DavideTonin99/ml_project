## Image preprocessing operations

- resize: all the images are resized to the same dimension (250x250)
- scaling: we used sklearn's StandardScaler to standardize the features by removing the mean and scaling to unit variance
  - the mean and the variance where computed over the training set, and then used to scale the test set
- cropping: since the image resizing operation does not take into account the image proportion, we considered and tested cropping the images to a squared proportion before applying the resizing process
  - after a few tests, we realized it was not a problem, so we didn't apply the cropping anymore
  - e.g: 
    - we have an 1920x1080 image representing a cherry
    - The cherry is round
    - if we apply the resizing before the cropping, the cherry will be distorted and no more round, but oval and elongated
- gray scale: we considered and tested converting the images to gray scale, but it didn't make much of a difference, even though the computation was faster due to the reduced number of channels in the images

## Validation metrics
The following validation metrics were used to assess the quality of the classification output.
Each of the following metrics comes with its mean and standard deviation (and relative standard deviation - RSC), since the algorithm output was evaluated on a repeated number of executions (about 100 for binary classification tests). 

- accuracy
- precision
- recall

## Dimensionality reduction - Principal Component Analysis
In order to reduce the feature space, we used the PCA algorithm.
We chose an 80% threshold, i.e. we kept the components that together consisted of 80% of the variance. 
To ensure the right number of components, we instantiated the PCA constructor with `n_components=0.8`.
Here it follows the variance distribution, from which follows the choice of the number of components, and a couple of screenshots of the reconstructed images after the reduction operation.
At first we implemented PCA from scratch like we did in lab lectures, but the computation was really slow, due to the number of images and their size.
So, we went with sklearn's built-in method, that is much faster.

## Classification analysis - Support Vector Machine
We performed the classification with SVM, playing around with the *kernel*, the *C* parameter and the *gamma* parameter.
The SVM input is the PCA output, i.e. the reduced feature space. 
At first, we tested it with a subset of the classes to get a feeling of the output quality.
We chose 2 classes that were really different from each other: lemon and pineapple.
With more than 2 classes, the classification results were not satisfying.

## Reasons why the SVM algorithm did not perform well:

- target classes are overlapping
  - data samples appear as valid instances of more than one class
  - mettere immagini che possono essere sia di una classe che di un'altra
- too many support vectors compared to the number of samples
  - the consequence is high chances of overfitting
  - example with binary classification
    - number of train samples: 37
    - number of support vectors: 21
      - more than half of the training samples are support vectors!

Below are the results of some tests:
mettere risultati da file txt di output