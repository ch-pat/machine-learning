"""
In traditional programming, you provide the data and the program (or function) and obtain an output.
In Machine Learning, you provide the data and the output and obtain a program (function, or model).

== Design cycle ==
1. Data preparation - includes collecting and cleaning data.
2. Feature engineering - Feature extraction, selection and construction into a final dataset of feature 
3. Model generation - model selection and hyperparameter optimization to get best results
4. Model evaluation - determine best model

== Data ==
Cleaning and preprocessing
Get rid of errors and remove redundancies
Rename, rescale for efficiency, discretize, aggregate before using for training

== Feature Selection / Engineering ==
The feature space might be enormous. When considering, for example, a dataset of words, it might contain
millions of different words. _Dimensionality Reduction_ might be needed: one could group sets of words with
positive meaning, for example. This also helps some less used words be properly represented, while their
meaning might be lost otherwise due to not enough examples in the dataset.

== Model Selection ==
There are 4 categories of machine learning algorithms:
- Supervised learning (or inductive)
  - training data contains output labels
  - lead to a program that can predict the correct label for new, unseen input
- Unsupervised learning
  - no expected output in the data
  - divides input into clusters; new input is categorized as one or more of these clusters
- Semi-unsupervised
  - a few output labels in the data
- Reinforcement learning
  - Model strives to maximize a 'score', given based on the actions of the model
  - given a task to accomplish, devises a strategy to complete it.

The resulting prediction can also be of different types.
When the output is _discrete_ it's called _classification_
  - can be binary or multi-class
  - resulting function may not be linear
when the output is continuous, it's called _regression_
  - resulting function may not be linear

== Supervised ML ==
We have various supervised models
- Discrete classifiers
  - Decision trees
  - Decision forests
  - Support Vector Machines (SVM)
- Continuous classifiers (Regression)
  - Neural Networks
- Probability estimators (when the output is a probability)
  - Naive Bayes
- Many Ensemble methods that combine the results of multiple models

== Unsupervised ML ==
Generally, the model learns the similarity in groups of data and divides it in clusters
For example it might group people based on their traits. Naturally it is more complex than 
supervised learning (what even is 'similar'?)
- Convolutional Neural Networks
- Rule Learning
- Clustering

== Reinforcement Learning ==
Learn a policy or strategy to act in an environment.
Learn it through pure trial and error, the 'agent' learns through rewards for 'better' results
- Q-learning

== Learning ==
Learning can be seen as an optimization problem: learn a function that _minimizes error_ or _maximizes reward_
Set hyper-parameters correctly to optimize the learning algorithm; hyper-parameters are settings
of the learning algorithm itself

An example of parameters optimization in a decision tree model.
We have two avenues of optimization: and the values for splits in the tree. Therefore hyper-parameters, set
before the learning process begins, determine the structure of our model. They also have significant
effect on performance.

Methods for hyper-parameter optimization are Grid Search and Random Search
these methods are sometimes unfeasible, especially when the number of hyper-parameters is large

== Evaluation ==
It's impossible to obtain perfect predictors, but we certainly want to distinguish a good predictor from
a bad one.

The most commonly used way of determing 'how good' a model is relies on splitting the dataset into a 
_training set_ and _test set_. The model learns from the training set and its results are judged on
input taken from the test set.
We apply appropriate measures on the results and determine which are the most important for our case.
Such measures are Accuracy, Precision, Recall, AUC...

Other more complex methods are based on techniques such as cross-validation or random sub-sampling.

We are satisfied when our model achieves good statistical measures.
"""