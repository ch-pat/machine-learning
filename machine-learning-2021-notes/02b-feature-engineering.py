"""
What is our role in Machine Learning?
1. Set up the problem the machine needs to solve correctly and optimized
2. Identify the relevant features
3. Find relevant data
4. Identify the best learning algorithm for the job

What can happen:
1. You have a problem but don't know what features would help with the problem 
   (and don't know how to gather data)
2. You have some data and a problem

Workflow (Data collection and preprocessing):
1. Problem analysis: what is the task? How to solve it?
2. Feature identification: which features will be useful?
3. Data collection: where to find data?
4. Feature Extraction: obtain features from _dirty_ data                       } Feature Engineering
5. Feature transformation: normalize, aggregate, discretize data appropriately } Feature Engineering

Gathering data:
Data is important; nowadays there are many sources of data that are freely accessible.
The only problem is that they are often not ready for use; that's why we need to preprocess it.
Often we gather data from multiple sources with different structures or different units of measurement
and need to really change it, scrap irrelevant parts, keep the important features and build a new dataset from it.
If the data is not coherent within itself, the learning algorithm will not work properly.

FEATURE EXTRACTION
== Text ==
When working with text we usually _tokenize_ data, as we are mostly interested in the used _words_.
We clean up (stemming/lemmatization) the different versions of same words (work, works, working... -> work)
Encode -> usually bag of words is used, each document becomes a vector [0, 1, 1, 0...]
with each 1 corresponding to a word being used in a vocabulary vector that contains _all_ words encountered.
A new idea is _embedding_ of words, where each word is a vector itself. This creates a concept of 'distance'
between words, with close words having similar meaning.

== Images ==
Can analyze images at the pixel level.
Can also use techniques such as Convolution to extract higher level features from images (such as edges).
Can be used to extrapolate many features beyond the recognition of objects! For example in house listings
brighter house pictures attract more attention, therefore one can extract the 'avg pixel brightness' feature!

== Geospacial data ==
Latitude, Longitude, interfaced with Maps databases can obtain a lot of data
Feature that can be obtained are limited only by your imagination. Distance from subways, # of stores within
radius, # of restaurants etcetera...

== Date and time data ==
Extract day of the week; is_weekend can be a bool feature, paydays, holydays.

== Time series ==
Sequential data, can be found in many places (stock market is most common)
Can extract statistics such as average, min, max. Analyzing peaks might be interesting.

== Other kinds of data ==
Data can be anywhere and anything, use your intuition to come up with useful features to extract.

FEATURE TRANSFORMATION
== Normalization ==
Certain learning algorithms may impose some limitations on the type of data they accept (decision trees
allow for any type of data, but most other algorithms strictly need numbers)
Some algorithms really screw up when using unbalanced data (features may need to be normalized to 
live in the same range)
Methods such as clustering look at the 'distance' between data points, therefore are disproportionately 
affected by unbalanced data. Normalizing will help remove this issue and should not introduce bias 
-- the different values only reflect the measurement used, not any intrinsic property --
Centering is another kind of normalization that brings all points around the center of the plane; it is
done by subtracting the mean of a sample from all values.
"""
import numpy as np
def centering(arr: np.array):
    mean = arr.mean()
    return arr - mean

if __name__ == '__main__':
    print("\nCentering example:")
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(centering(arr))

"""
Scaling of real-valued features is done by dividing all values by the standard deviation of the sample
"""

def scaling(arr: np.array):
    mean = arr.mean()
    size = arr.size
    normalizing_factor = 1/ (size - 1)
    return np.sqrt(normalizing_factor * (arr - mean) ** 2)

if __name__ == '__main__':
    print("\nScaling example:")
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(scaling(arr))

"""
Change of bases
sometimes when analyzing data we need a visual representation of it -- often we want the data to look
_normal_ or close to normal.
We can quantify _skewness_ of data and use it to determine how different the distribution is to a gaussian
many algorithms are actually affected by skewness and it can help to mitigate it (regression, knn and k-means)
Skewness can be reduced by using a log transformation, or other more complex methods.

Categorical data to Numeric
As said before some algorithms do not accept categorical data; nonetheless some data is just categorical by
nature (artist name, name of song...)
Even among categorical data we may have some kind of ordering (days of week), that would be _ordinal_ data,
as opposed to _nominal_ data where no intrinsic ordering is present.

One way to deal with categorical data is to use _one-hot encoding_, which encodes a categorical data point
into a vector in which only one element is 1, and the others are 0. (it is 1 where it corresponds to
its categorical value)
"""

def one_hot_encoding(arr: np.array):
    values = sorted(set(arr))
    size = len(values)
    new_list = []
    for value in arr:
        encoded = [0 if value != values[i] else 1 for i in range(size)]
        new_list += [encoded]
    return np.array(new_list)

if __name__ == '__main__':
    print("\nOne-Hot Encoding example:")
    arr = np.array(['green', 'red', 'yellow', 'green', 'yellow'])
    print(one_hot_encoding(arr))

"""
However when the size of the sample is large (when there are many different values to encode) the one-hot
encoding might be a bad idea, as it introduces a lot of dimensions.

Label Encoding can be used in such cases, in which every categorical value is assigned a number.
Algorithms that assign weight to the values will perform poorly with this technique, as bias is introduced
towards values that are assigned high numbers.
A possible intermediate option is label binarization that limits the additional dimensions to log_2(n)
"""

def label_encoding(arr: np.array):
    values = sorted(set(arr))
    size = len(values)
    new_list = []
    for value in arr:
        encoded = values.index(value)
        new_list += [encoded]
    return np.array(new_list)

def binary_encoding(arr: np.array):
    labels = label_encoding(arr)
    width = int(max(labels)) if  int(max(labels)) == max(labels) else int(max(labels) + 1)
    new_list = []
    for value in labels:
        new_list += [[int(x) for x in np.binary_repr(value, width=width)]]
    return np.array(new_list)


if __name__ == '__main__':
    print("\nLabel Encoding example:")
    arr = np.array(['green', 'red', 'yellow', 'green', 'yellow'])
    print(label_encoding(arr))
    
    print("\nLabel Encoding example:")
    print(binary_encoding(arr))

"""
Missing Values
real data is imperfect, it contains mistakes, impossible values and missing values.
Many algorithms do not accept missing values, there are ways to handle this.
First, check how many values are missing, if they are a few you can just remove the lines where data is missing.
If they are a lot, you can't just throw away most of your data.
For numerical values you can replace the missing values with the mean of the sample (or median/mode when
appropriate). For categorical value the best you can usually do is replace missing with the most likely value 
(but this can introduce bias...)
A more advanced approach is using a regression to determine the value for that data point based on the other
features. Use a correlation matrix to determine which features will best help determine the missing value.
Another approach uses K-Nearest Neighbours, just use the most common value that the nearest data points have.

Data Augmentation
Methods that _add_ data from existing data (image transformations such as rotation, functions of other features)
use your imagination.

Category Imbalance
*Class imbalance*
 happens when some classes of a feature do not have equal representation.
If a particular class is dominant, say, appears in 90% of samples, then your learning algorithm
may just decide to always predict that class and achieve 90% accuracy. Or maybe you are mostly interested
in the less represented class anyway. In any case, class imbalance adds strong bias to our model.

We can address this problem by over/undersampling -- just pick less samples of the dominant data or create
copies of the smaller data.
This has consequences though: undersampling is a waste of precious data, while oversampling may cause 
overfitting as the copies are identical.
SMOTE (synthetic minority oversampling technique) is an oversampling tech that produces convex combinations
of neighbour instances of data, which is very powerful. You can imagine it as adding points along a line
of existing sample points.

*Anomaly Detection*
We often assume that the distribution of data follows a normal distribution -- outliers, especially
when few, represent anomalies very often. We can simply ignore such samples (unless we were specifically
looking for them, such as in fraud detection)

*Cost-Sensitive learning*
We may address imbalance by penalizing more harshly wrong predictions on low-populated classes, thus
prompting the algorithm to "pay more attention" to these classes.

FEATURE SELECTION
Too many? Too little? 
We can always come up with a lot of features, but we may not need all of them, this is task dependent.
When you want to distinguish nature images and city images, you might not need to recognize shapes: pixel
colors might be enough.

_Potentially_ useful features can be hundreds, but this is often too much. Adding _correlated_ feature might
even lower model performance.
Training time gets longer as features increase, so we really only want to pick useful ones, but this
also gets harder as the number of features increases!
A need for automated feature selection arises.

Exhaustive search is mostly not practically doable, so we won't delve in it. (small note on dimensionality
reduction: feature selection simply removes features. Dimensionality reduction transforms the feature space
into a lower dimensional one without sacrificing the feature involved)

Filter Methods
Select features based on a performance measure, such as information, distance, consistency, similarity.
Filters can look at a single feature (univariate) or be multivariate.
Examples: Information Gain, Relief (updates _quality_ of a feature by comparing distance between instances),
Correlation (Spearman, x-squared)

Wrapper Methods
Evaluate performance of a subset of features on an ML algorithm. Pick the best subset for the actual training.
This procedure is much slower than filtering due to the need of running multiple times learning algorithms.

Similarly we have _Sequential Feature selection_ to extract the smallest subset of features with maximum
performance. This method can't be used for datasets with too many features due to exponential increase in
run time, but it can be applied heuristically in such cases. 
The idea is to start with the single feature that leads to the best performance, then add one feature from
the remaining one and choose the one that leads to the highest performance... and so on and so forth (greedy).
When no improvement is achieved at the next step, we stop.

Embedded methods
These methods perform feature selection _during_ the learning process; they are model dependent.
They are kind of recursive, they produce a model and then determine which features are least important using
their own model. Ex: Regularization and Lasso (L1)
"""