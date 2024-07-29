#!/usr/bin/env python

# Test harness for final service
#
# This just iterates on all input files and sends them to the service.
# It computes performance metrics in the same way as in the notebook where the model
# is trained, and the scores are very similar (it tests the entire dataset, rather than
# the 20% test set)
#
#     Final accuracy metrics for the service:
#     Accuracy: 0.9821073558648111
#                    precision    recall  f1-score   support
#     
#          business       1.00      0.98      0.99       100
#     entertainment       1.00      0.98      0.99       100
#              food       1.00      1.00      1.00       100
#          graphics       1.00      0.94      0.97       100
#        historical       1.00      0.99      0.99       100
#           medical       1.00      1.00      1.00       100
#             other       0.32      1.00      0.48         6
#          politics       0.96      1.00      0.98       100
#             space       1.00      0.94      0.97       100
#             sport       1.00      1.00      1.00       100
#       technologie       0.99      0.99      0.99       100
#     
#          accuracy                           0.98      1006
#         macro avg       0.93      0.98      0.94      1006
#      weighted avg       0.99      0.98      0.99      1006
#     
#     [[ 98   0   0   0   0   0   0   2   0   0   0]
#      [  0  98   0   0   0   0   1   0   0   0   1]
#      [  0   0 100   0   0   0   0   0   0   0   0]
#      [  0   0   0  94   0   0   6   0   0   0   0]
#      [  0   0   0   0  99   0   1   0   0   0   0]
#      [  0   0   0   0   0 100   0   0   0   0   0]
#      [  0   0   0   0   0   0   6   0   0   0   0]
#      [  0   0   0   0   0   0   0 100   0   0   0]
#      [  0   0   0   0   0   0   5   1  94   0   0]
#      [  0   0   0   0   0   0   0   0   0 100   0]
#      [  0   0   0   0   0   0   0   1   0   0  99]]


import os
import json
import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def classify_document(file_path):
	url = 'http://127.0.0.1:5000/classify_document'
	with open(file_path, 'r', encoding='utf-8') as f:
		file = { "document_text": f.read() }
		files = { "file": json.dumps(file) }
	response = requests.post(url, files=files)
	response.raise_for_status()
	return response.json()

# walk tree of provided data, assembling predictions and ground truth
y = []
y_pred = []
root_path = "Data"
subdirs = sorted(os.listdir(root_path))
for subdir in subdirs:
	subdir_path = os.path.join(root_path, subdir)
	files = sorted(os.listdir(subdir_path))
	for file in files:
		file_path = os.path.join(subdir_path, file)
		result = classify_document(file_path)
		label = result.get('label', 'unknown')
		y.append(subdir)
		y_pred.append(label)

print("\n\nFinal accuracy metrics for the service:")
accuracy = accuracy_score(y, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))

