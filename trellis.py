import os
import sys
import json
import joblib
import re
from flask import Flask, request, jsonify
import numpy as np
import spacy

app = Flask(__name__)

#
# Load configuration, and with it, the model itself.
# Also load tooling for Spacy-related NLP
#

data_dir = "output"
config_name = os.path.join(data_dir, "model_config.json")
if not os.path.isfile(config_name):
	print(f"Configuration file '{config_name}' not found", file=sys.stderr)
	sys.exit(1)
with open(config_name, 'r', encoding='utf-8') as f:
	config = json.load(f)

def load_joblib(filename:str):
	try:
		model = joblib.load(filename)
	except: # likely FileNotFoundError, but catch all exceptions
		print(f"Cannot load file '{filename}'", file=sys.stderr)
		sys.exit(1)
	return model
	
model_file = config.get('model_file')
vectorizer_file = config.get('vectorizer_file')
other_threshold = config.get('other_threshold')

if not model_file:
	print("Model filename not in configuration", file=sys.stderr)
	sys.exit(1)
if not vectorizer_file:
	print("Model filename not in configuration", file=sys.stderr)
	sys.exit(1)
if not other_threshold:
	print("Threshold for nferring 'other' documents not in configuration", file=sys.stderr)
	sys.exit(1)

nb_classifier = load_joblib(os.path.join(data_dir, model_file))
tfidf_vectorizer = load_joblib(os.path.join(data_dir, vectorizer_file))

nlp = spacy.load('en_core_web_sm')


def clean_text(doc:str) -> str:
	""" Prep text for feature building
		remove numbers, punctuation
		remove stop words
		lemmatization

		TODO: this is identical to the code in the training notebook.  We should
		refactor this into a shared module
	"""
	doc = re.sub(r'\d+', '', doc)  # remove numbers
	doc = re.sub(r'[^\s\w]', '', doc)   # keep only non-punctuation, whitespace
	doc = doc.replace('\n', '')  # some newlines accidentally escaped as a literal '\n'
	
	# let spaCy work on stop words, lemmatization
	doc_spacy = nlp(doc)

	non_stop = [token.lemma_ for token in doc_spacy if not token.is_stop]
	doc = ' '.join(non_stop)
	return doc

def make_prediction(doc_text:str) -> str:
	""" Given raw document text, make a class prediction.
		TODO: this is identical to the code in the training notebook.  We should
		refactor this into a shared module
	"""
	doc_text = [doc_text]										# vectorize, convert bare string to one row of string
	X_tfidf = tfidf_vectorizer.transform(doc_text)
	y_pred_proba = nb_classifier.predict_proba(X_tfidf)
	y_pred_proba = y_pred_proba[0]								# only want the one
	if np.max(y_pred_proba) - np.mean(y_pred_proba) > other_threshold:
		label = max(zip(y_pred_proba, nb_classifier.classes_))[1]
	else:
		label = 'other'
	return label

@app.route('/classify_document', methods=['POST'])
def classify():
	if 'file' not in request.files:
		return jsonify({'error': 'File not provided'}), 400		# 400 Bad Request

	file = request.files['file']
	if not file:
		ret = { "message": "Classification unsuccessful" }
		return jsonify(ret), 400

	try:
		doc = json.loads(file.read().decode('utf-8'))
		doc_text = doc['document_text']
	except:
		ret = { "message": "Cannot parse JSON input" }
		return jsonify(ret), 400
	
	# else proceed normally
	doc_text = clean_text(doc_text)

	label = make_prediction(doc_text)

	ret = { "message": "Classification successfully",
			"label": label }
	return jsonify(ret)


if __name__ == '__main__':
	app.run(debug=True)
