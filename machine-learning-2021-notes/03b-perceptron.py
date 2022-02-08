"""
Perceptrons
A simple classifier and a _building block_ of Neural Networks.
Perceptrons learn an _algebraic function_ rather than a boolean function (DTree), and is also linear.
(Neural Networks combine perceptrons into a non-linear function often)
Input and output can be real valued -> can be a classifier or a regressor.

Neural Networks
Similar to a biological neural system (made for learning); allow for a lot of parallel computing;
Behaviour of a Neural Network is an emergent property of many simple units.
Difficult to interpret results...

Compared to biological neurons, computers are orders of magnitude faster. Nonetheless the human brain
can perform recognition tasks at incredible speeds.
Idea: human brain must take advantage of massive parallel computation (10^11 neurons in human brain,
with 10^4 connections _each_)

Perceptron (single neuron model)
We model a neuron as a graph with (biological) cells as nodes
We define the _net_ of a node as net = sum(wi, xi), wi weights, xi feature values of instances X
The neuron outputs 1 or 0 depending on net and a threshold.

Each perceptron learns a linear decision boundary -- to be precise the perceptron learns
the values for the weights wi such that the learned function f(x) correctly classifies each (or most) instance
to learn the correct weights we use an iterative update algorithm (to minimize error at each step)

Pseudocode
Set all weights to random values
until all instances are correctly classified:
    for instance in instances:
        compute net(instance)
        update weights as such: w_i = w_i + n * (y_j - o_j) * x_ij

y_j - o_j is the _error_
therefore w_i will keep changing value until error is zero (ie: output is _correct_)
n is the learning rate and determines how much w_i changes at each step

Note however that perceptrons can't learn everything; some data is not _linearly separable_
it might need a polynomial decision boundary.
Luckily, if the data _is_ linealry separable, then the perceptron is guaranteed to converge (which also means
a set of weights exists that is compatible with the data)

Note that the iterative process of updating weights corresponds to an optimization problem
more specifically it is the minimization of classification error.
In the space of weights, this is a Gradient Descent (or hill-climbing) that reaches a local minimum 
(the model of a single neuron is well-behaved and has a single minimum).

Pros and cons of Perceptrons:
They converge fairly quickly in practice, when dealing with linearly separable data.
They are not appropriate when data is not linearly separable, therefore they are insufficient for many tasks.
This can be solved with multi-layer perceptrons.
"""