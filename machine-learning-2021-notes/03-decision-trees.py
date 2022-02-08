"""
SUPERVISED LEARNING
We have a dataset D of instances represented by feature vectors X: (x1..xn) and for each X we have an unknown
value y of the function f(x) that will be learned.
When f(x) is discrete, the model is called a _classifier_
when f(x) is continuous, we call the model a _regression_

A simple classifier is the Decision Tree, which outputs discrete labels (a similar algorithm that leads to 
continuous values is called a regression tree).

== Decision Trees ==
The model output takes the form of a tree structure where each node is a _test_ on feature values, with one
branch for each possible value of the feature. Leaf nodes represent a 'decision'.
Interestingly, a tree can be rewritten in first order login in Disjunctive Normal Form.
When considering the feature space, a decision tree basically divides the feature space in areas between 
'decision boundaries'.

How to construct a Decision Tree (DTree)
Begin with considering one feature at the top node, in the children nodes corresponding to values of that feature
(for example, for feature color the children can be 'green', 'blue', 'yellow'...).
Therefore at each level of the DTree we have a split based on the values of each feature.
The goal of a good DTree algorithm is to find _the best split_

Pseudocode
DTree(examples, features):
    if all examples are one category: return leaf node with that category
    else if features is empty: return leaf node with most frequent category
    else: pick a feature f and create node R for it
        for value v in f:
            ex_i = [ex for ex in examples with value v for feature f]
            add an edge from R with label v
            if ex_i is empty:
                edge goes to leaf node with most common category in examples
            else:
                edge goes to DTree(ex_i, features - f)
    return the tree starting at R

This pseudocode tells us how to determine the class when we are out of examples, but as said before
the order of the splits is imortant for performance. How do we decide that?

The best split is the one that leads to the *smallest tree* (application of Occam's Razor)
Sadly, finding the minimal DTree is NP-hard. We are content with greedy algorithms then.
At each split we pick the feature that creates the 'purest' subsets of examples, that is those mostly
dominated by a particular value --> makes them closer to being leaf nodes.
A popular method is based on information gain.

We can interpret the above 'pureness' as low entropy.
"""

import math
def binary_entropy(examples: list):
    cases = tuple(set(examples))
    probability_1 = len([i for i in examples if i == cases[0]]) / len(examples)
    probability_2 = 1
    if len(cases) > 1:
        probability_2 = len([i for i in examples if i == cases[1]]) / len(examples)
    return -probability_2 * math.log(probability_2, 2) -probability_1 * math.log(probability_1, 2)

if __name__ == "__main__":
    arr = [1, 0, 0, 0, 0, 1, 0]
    print(f"Binary entropy for examples {arr}: ", binary_entropy(arr))
    arr = [0, 0, 0, 0, 0, 0, 0]
    print(f"Binary entropy for examples {arr}: ", binary_entropy(arr))
    arr = [0, 0, 0, 0, 1, 1, 1, 1]
    print(f"Binary entropy for examples {arr}: ", binary_entropy(arr))
    

"""
As we can see, when all examples belong to one category, the entropy of the set of examples is 0.
The maximum entropy of 1 is reached when examples are evenly split.

Entropy is closely related to information; in fact it can be seen as the _number of bits_ required to encode
the class of an example in D on average. It reflects the 'disorder' of classification for the set given.
"""

def set_entropy(examples: list):
    cases = list(set(examples))
    total = len(examples)
    probabilities = []
    for case in cases:
        probability = len([ex for ex in examples if ex == case]) / total
        probabilities += [probability]
    logs = [p * math.log(p, 2) for p in probabilities]
    return -sum(logs)

if __name__ == "__main__":
    arr = [1, 2, 2, 3, 2, 3, 2, 2, 1, 3, 3, 3]
    print(f"General entropy for examples {arr}: ", set_entropy(arr))

"""
Note that while binary entropy is always a number in [0, 1], general entropy is in the range [0, log(k)]
where k is the number of categories.

Information Gain
The information gain of a feature is the expected reduction in entropy resulting from a split on this feature.
We will want to make splits that maximize information gain at each step (greedy approach).
"""

def information_gain(examples: list, subset: list):
    """For the purpose of these notes, subset will be a list of subsets of examples that
    have different values v for the feature on which we perform the split"""
    full_entropy = set_entropy(examples)
    reduction = 0
    for case_subset in subset:
        reduction += (len(case_subset) / len(examples)) * set_entropy(case_subset)
    return full_entropy - reduction

import random
if __name__ == "__main__":
    arr = [1, 2, 2, 3, 2, 3, 2, 2, 1, 3, 3, 3, 1, 2, 2, 3, 3, 3, 1, 1, 2, 2, 1, 2, 1, 3, 3, 1]
    subset = [random.sample(arr, random.randint(1, 3)) for _ in range(random.randint(1, 5))]

    print(f"Information gain for examples {arr} and subset {subset}: ", information_gain(arr, subset))

"""
Now that we can quantify the 'best' split, we can include it in the pseudocode for DTree to make
the best split at each step

Note that in the calculation of 'reduction' we have the term (len(case_subset) / len(examples))
This is called _cover_ or _support_. This is used to 'weight' the particular entropy gain of that subset
of examples.
*Confidence* is closely related to support: it is the fraction of confidence samples that actually
correctly label the input (remember that when we are out of features and the values for the last feature
still contain different results, we are forced to label all examples with the most likely outcome,
therefore some examples are 'mislabeled' in the end. A low amount of mislabeled examples produces
a high value for confidence).

Issues of Decision Trees
We can extend DTrees to support real valued features by splitting them into ranges of values.
Efficient in data mining and with large amounts of data.
Methods ready-to-use for missing and noisy data.
Regardless, output is always discrete (regression trees needed for real values)

Regression Trees
Regression trees work by splitting the feature space into (multi-dimensional) rectangles (aka boxes). Each box
is assigned the mean value of values within the box and will be given as output. They can be any shape in theory
but it is easier to reason with rectangles.

The goal is to find boxes that minimize the _residual sum of squares_ (RSS error). Once again, we can't
exhaustively check every possible partition of the feature space, therefore we use a greedy approach called
_recursive binary splitting_, which splits all examples within an area into two sub-areas (corresponding to
the two children nodes of the resulting tree).
(The resulting splits kind of look like fibonacci rectangles)

###########
#       # #
#       # #
#       # #
###########

###########
#       # #
######### #
#       # #
###########

###########
#       # #
######### #
#   #   # #
###########

Overfitting
A tree that best classifies learning data may not generalize well on unseen data.
The definition of an _overfitting_ *hypothesis* is an hypothesis h such that there exists a different 
hypothesis h' where h performs better than h' on training data but worse on unseen data.

This may happen due to poor data or a lack of examples for a specific, existing and reliable trend.
In case of a regression, underfitting corresponds to a low-polynomial curve while 
overfitting is a high-polynomial curve.

How to avoid overfitting
For DTree we have pruning methods:
- pre-pruning: stop growing the tree when data becomes scarce (towards the end the last few features
  may only be needed to categorize a few data points -- those datapoints may well be outliers and reduce
  the final correctness)
- post-pruning: as above, but first build the full tree and then remove branches with unsufficient data
we label the remaining leaf nodes affected by pruning with the majority class.

Reduced error pruning
A post-pruning approach: idea is to split data into training and validation; build a tree from the learning
data, attempt to prune a branch of the tree and record the resulting accuracy. Repeat until the accuracy of the
tree on the validation sets starts decreasing. If a branch lowered accuracy, prune it permanently.
Risks: we might throw away data; if we are on a plateau on the learning curve we might keep going and
wasting a lot of data.
"""