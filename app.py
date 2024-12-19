import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):

    text = text.lower() # lowercase
    text = nltk.word_tokenize(text) #toenization of sentence into each word

    y=[]
    # sort the words to remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text =y[:] #cloning y into text. simple copying of array is not possible as it is immutable
    y.clear()
    # remove stopwords and punctuation marks
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # lemmatize into root word
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    

    # join the individual words
    return ' '.join(y)

tfidf= pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")


if st.button('Predict'):
    # 1. preprocess
    transformed_sms=transform_text(input_sms)
    # 2. vectorize
    vector_input= tfidf.transform([transformed_sms])
    # 3.predict
    result=model.predict(vector_input)[0]
    # 4. dispaly
    if result==1:
        st.header("Spam")
    else:
        st.header("Not spam")


