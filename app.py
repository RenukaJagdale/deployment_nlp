from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open('nlp_model_xgb.pkl','rb'))
cv=pickle.load(open('tweet_tranform_new.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('first_page.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
	    message = request.form['message']
	    data = [message]
	    vect = cv.transform(data).toarray()
	    my_prediction = clf.predict(vect)
    return render_template('second_page.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run()
