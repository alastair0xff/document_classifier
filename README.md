# A simple document classifier
A simple example of a document classifier model, for Trellis.Law's Data Science
Case Study.

Given a corpus of already-classified documents, we build a model to classify
new unseen documents, into one of ten predetermined classes: technology, sport,
space, politics, medical, historical, graphics, food, entertainment and
business.  Additionally, the model is also designed to fall back to an 'other'
class for when it can't make predictions about any of the other classes.  The
model is hosted inside a simple Flask service to be accessed from a REST
endpoint

# Architecture

The model is a straightforward Naive Bayes classifier from `scikit-learn`,
trained on a matrix of TF-IDF features.  Words common to a document class and
not do others provide the signal.  Preprocessing is done to clean out non-word
text (spaces, punctuation, etc) and to lemmatize the words and remove stop
words.

Rather than just asking the classifier for a class, we instead get per-class
probabilities.  Intuitively speaking, if the model is confident in it's class
prediction, the probability for that class will be much higher than the rest.
However, to allow for a non-prediction when it's not confident, we also check
for if the maximum probability is not much higher than the rest of the
probabilities, e.g. the mean of all probabilities.  The cutoff threshold for
this is determined empirically, based on the data.  We do not have a corpus of
documents from an 'other' class, and building such a corpus probably doesn't
make sense: incorporating a representation of all possible documents that
aren't in these ten classes does not sound like a reasonable task.

# Performance

Without including logic for the 'other' class, we get an accuracy of about
0.937, with details provided in the Jupyter notebook.  Once the 'other' logic
is added, we get a small performance hit, dropping by 0.0194, with further
details provided below.

The model has generally very good performance as shown by the F1 scores.  By
including the 'other' class, we do capture the few test examples that aren't
part of the regular class list, but we do get a number of false positives; 
a handful of documents are misclassified as 'other', as show by its' low
precision.

```
Accuracy: 0.9174757281553398
Difference from previous model: -0.0194
               precision    recall  f1-score   support

     business       1.00      0.91      0.95        22
entertainment       1.00      0.92      0.96        24
         food       1.00      1.00      1.00        14
     graphics       1.00      0.75      0.86        24
   historical       1.00      0.95      0.97        19
      medical       1.00      1.00      1.00        18
        other       0.32      1.00      0.48         6
     politics       0.83      1.00      0.91        15
        space       1.00      0.77      0.87        22
        sport       1.00      1.00      1.00        19
  technologie       0.96      0.96      0.96        23

     accuracy                           0.92       206
    macro avg       0.92      0.93      0.91       206
 weighted avg       0.96      0.92      0.93       206

[[20  0  0  0  0  0  0  2  0  0  0]
 [ 0 22  0  0  0  0  1  0  0  0  1]
 [ 0  0 14  0  0  0  0  0  0  0  0]
 [ 0  0  0 18  0  0  6  0  0  0  0]
 [ 0  0  0  0 18  0  1  0  0  0  0]
 [ 0  0  0  0  0 18  0  0  0  0  0]
 [ 0  0  0  0  0  0  6  0  0  0  0]
 [ 0  0  0  0  0  0  0 15  0  0  0]
 [ 0  0  0  0  0  0  5  0 17  0  0]
 [ 0  0  0  0  0  0  0  0  0 19  0]
 [ 0  0  0  0  0  0  0  1  0  0 22]]
```

# Hosting

The model is hosted in a simple Flask container that is started by running
```
$ python trellis.py
```

There is a separate script, `test_service.py` which iterates through all
provided documents and builds a set of performance statistics similar to the
above.  It runs across the entire dataset, not just the 20% test set, so the
performance details differ, but only slightly.


