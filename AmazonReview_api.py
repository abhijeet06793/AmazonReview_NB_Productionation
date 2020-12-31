from flask import Flask, request
import joblib
from joblib import load
import re
from nltk.stem import wordnet, WordNetLemmatizer, LancasterStemmer, PorterStemmer
import json

app = Flask(__name__)

def processing_of_query_point(comment):

	word_lem = WordNetLemmatizer()
	pst_stem = PorterStemmer()
	text = ("").join(char for char in comment if char not in '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')  ##string.punctuations
	text = (" ").join(word for word in text.split() if len(word)>2) # removal of words less than 3 characters
	text = re.sub(r"[^A-Za-z]+", " ", text) #removal of non english words
	text = re.sub(r"\s+", r" ", text) #Removal of multiple spaces
	text = (" ").join(word_lem.lemmatize(word) for word in text.split()) ##Lemmatization
	processed_comment = (" ").join(pst_stem.stem(word) for word in text.split())   ##Stemming
    
	return processed_comment
	

@app.route("/comment", methods=['POST'])
def class_prediction():
	reqobj = request.get_json()
	#print(reqobj['comment'])
	
	x = processing_of_query_point(reqobj['comment'])
	bow = load(r'G:\Model Deployment\BOW_Amazon.joblib')
	bow_comment = bow.transform([x])
	clf = load(r'G:\Model Deployment\Naive_Bayes_Amazon.joblib') # loading my model
	clf_class = clf.predict(bow_comment)
	clf_prob = clf.predict_proba(bow_comment)

	print("The prob of comment being positive is:",str(clf_prob[0][1]))
	print("The predicted class of the query point is:",str(clf_class[0]))
	
	dic = {"class":str(clf_class[0]), "prob":str(clf_prob[0][1])}
	json_obj = json.dumps(dic)
	return json_obj
    
	
if __name__ == '__main__':
    app.run(debug=True)
	
