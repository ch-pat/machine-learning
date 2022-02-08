"""
EVALUATION
There are no guarantees on the correctness of the results of a machine learning model, and
take into consideration the fact that, despite training on known data, what we really need is for the model
to be able to make correct predictions on unseen data.
Therefore we need some way to be able to evaluate how a model performs. 
We call 'hypotheses" the alternative models we could use for the task at hand (ie: when pruning a dtree,
the different variations are all different hypotheses).
What we need to be able to differentiate is _what hypothesis best predicts reality_.

1. What performance metrics should we use?
First of all, performance can _only_ be evaluated if we have some way to know the _ground truth_ (some data that 
we didn't use for training or an human expert to evaluate the results)
Any performance measure is a function of the errors made by the model.
Which performance measure to use also depends if we are learning a regressor or a classifier
    classifiers are binary (either the prediction is wrong or it is correct)
    regressors take into account the _distance_ between predicted value and correct value


Reality and hypothesis
Independently of what kind of predictor we are learning, either a continuous f(x) or discrete c(x), the predictor
will be a function that at best _approximates_ the "real" predictor function that correctly predicts for 
every case in existence.
Obviously our dataset will only be a subset of all real cases -- in other words it only contains a subset of points
of the "real" function and our model sort of tries to "fill in the blanks"

Therefore we can say that learning a model is the same as learning a function h(x) (hypothesis) that best approximates
the unknown real function.
This means that we want to minimize the error of our h(x) on the learning set of points (our train data).

PERFORMANCE OF CLASSIFIERS
Classifiers
For classifiers we are often interested in the type of error (the direction of the error, in some sense)
As such we usually make use of the 'confusion matrix', which (in binary classification) contains 4 cells:
    true positive TP
    true negative TN
    false positive FP
    false negative FN

These statistics can be further summed up in specific performance measures
    Precision = TP / (TP + FP)                              
        # Rate of true positives over all model positives
        goes to 1 when false positives go to 0 (perfect recognition of positives; can 'accept' false negatives)
    
    Recall = TP / (TP + FN)       (True positive rate)                          
        # Rate of true positives recognized over all real positives
        goes to 1 when false negatives go to 0 (perfect recognition of negatives; can 'accept' false positives)
    
    F-Score = 2* Precision * Recall / (Precision + Recall)  
        # Harmonic mean of precision and recall
        goes to 1 when both precision and recall are 1; gives good idea of overall accuracy
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)              
        # Overall correct guesses

    Error Rate = Classification Error = 1 - Accuracy
        # Percentage error

    Less relevant measures:
    
    Specificity = True Negative Rate = TN / (TN + FP)
        # Rate of true negatives over all model negatives

    False Positive Rate = FP / (TN + FP)
        # Rate of wrongly classified negatives over all real negatives

    False Negative Rate = FN / (FN + TP)
        # Rate of wrongly classified positives over all real positives


    All the above are for binary classification, but the definition can be easily extended to multiclass classification
    Ex: we know we have 251 spam emails; our classifier finds 233 spam emails
        of the 233 spam emails, we know 200 are spam (correct), 30 are normal and 3 urgent (other classes)
        The precision of our classifier _over the spam class_ is 200 / (200 + 30 + 3)

        We can then use this definition of precision over a class to calculate an overall precision:
        With n classes:
        Precision = (precision_1 + precision_2 + precision_3... + precision_n) / n

    ROC Curve (Receiving Operator Characteristic curve)
    A graphical plot for binary classifiers is obtained when plotting 
    Recall (TPR - True positive rate) over FPR (False Positive Rate)
    Ideally we want a plot with the most area under the ROC curve; the worst ROC curve is the one that bisects the plane
    (corresponding to a random assignment in 50% probability binary events)
    Therefore the higher the area under the curve, the better the model recognizes the classes it needs to recognize.
    Usually we want to fine tune a model (changing its hyperparameters) and compare the different 
    Areas Under ROC curve (AUROC) to determine the better version of the model.

    ROC Curve is very useful in determining a "Confidence Range" (or conversely, "Area of Uncertainty") 
    for our model -- an area where we are pretty confident in our prediction as opposed to 
    an area where we are less certain about the result. An AUROC of 1 represents perfect separation of cases,
    in other words complete confidence in the result.


    Precision-Recall Curve
    Similarly to AUROC, we can plot Precision against Recall at various tunings of hyperparameters to determine
    the performance of a model for binary classification.
    The area under precision-recall curve (AUPR) is better suited to imbalanced classes problems compared to AUROC.

    ROC vs PR
    ROC does not help much in high class imbalance cases: imagine a system where we want to predict fraud from users.
    Most users will not be fraudulent (negative), with a positive minority, but the minority is exactly what we're looking for
    ROC curve would still show the model as very good even when it fails to identify that positive minority
        in detail: a few real negatives fail to be recognized (we are mostly interested in correct positives)
        -> we have a few false negatives in a vast sea of true negatives
        -> FPR = FP / (FP + TN) = little / (little + many) -> we might wrongly identify an user as fraudulent
                                                              and not notice due to lots of TN
    
    Precision-Recall curve is highly sensitive to false positives, in particular precision drops sharply with more FP:
        Precision = TP / (TP + FP) -> especially sensitive in this low TP scenario
    
"""