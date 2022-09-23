import sys

 # import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report
import nltk
import pickle

nltk.download('punkt')
nltk.download('stopwords')

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM messages',engine)
    X = df['message'] 
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = list(np.array(Y.columns))
    return X,Y,category_names 
   


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    token=word_tokenize(text)
    
    stemmer=PorterStemmer()
    
    stop_words=stopwords.words('english')
    
    normelized=[stemmer.stem(w) for w in token if w not in stop_words]
    
    return normelized


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('clf',MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {'clf__estimator__n_estimators': [ 50,100],
             'clf__estimator__max_depth': [20,50],
             'clf__estimator__min_samples_split': [2, 4]}
    
    cv = GridSearchCV(pipeline,param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    predicted_l=model.predict(X_test)
    score=[]
    for i,column in enumerate(Y_test.columns):
        accuracy = accuracy_score(Y_test.loc[:,column].values,predicted_l[:,i])
        precision =precision_score(Y_test.loc[:,column].values,predicted_l[:,i],average='micro')
        recall = recall_score(Y_test.loc[:,column].values,predicted_l[:,i],average='micro')
        f1 = f1_score(Y_test.loc[:,column].values,predicted_l[:,i],average='micro')
        score.append({'Accuracy':accuracy, 'f1 score':f1,'Precision':precision, 'Recall':recall})
    col_names = list(Y_test.columns.values)
    result=pd.DataFrame(score,index = col_names)
    print("Result for Each Category")
    print(result)
    print("Overall Evaluation Result")
    print(result.mean())


def save_model(model, model_filepath):
    model = model.best_estimator_
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()