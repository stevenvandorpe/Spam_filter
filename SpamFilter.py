## Imports and data loading

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


## Split train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
													
													
## Distribution of labels

def answer_one():
    res = len(spam_data[spam_data['target'] == 1]) / len(spam_data['target']) *100
    return res
	

## Count Vectorizer
	
from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    xtrainvectorized = CountVectorizer().fit(X_train)
    xtestvectorized = CountVectorizer().fit(X_test)
    res = xtrainvectorized.get_feature_names()
    return sorted(res, key=len, reverse=True)[0]

	
## Naive Bayes Classifier on Count Vector

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    xtrainvectorized = CountVectorizer().fit(X_train)
    xtestvectorized = CountVectorizer().fit(X_test)
    xtraintransformed = xtrainvectorized.transform(X_train)
    xtesttransformed = xtrainvectorized.transform(X_test)
    clf = MultinomialNB(alpha=0.1)
    model = clf.fit(xtraintransformed, y_train)
    return roc_auc_score(y_test, clf.predict(xtesttransformed))
	
	
## tdidf Vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    xtrainvectorized = TfidfVectorizer().fit(X_train)
    xtestvectorized = TfidfVectorizer().fit(X_test)
    xtraintransformed = xtrainvectorized.transform(X_train)
    xtesttransformed = xtrainvectorized.transform(X_test)
    features = np.array(xtrainvectorized.get_feature_names())
    coo = (xtraintransformed.tocoo(copy=False))
    frame = pd.DataFrame({'document': coo.row, 'feature': coo.col, 'data': coo.data}
                 )[['document', 'feature', 'data']].reset_index(drop=True)
    frame = frame.drop(['document'], axis=1).groupby('feature').max().reset_index().sort_values(['data', 'feature'])
    frame['feat_name'] = features[frame['feature']]
    smallest = frame[:20]
    smal = pd.Series(data=smallest['data'].values, index=smallest['feat_name'])
    smallest = frame[-20:]
    big = pd.Series(data=smallest['data'].values, index=smallest['feat_name'])
    frame = pd.DataFrame(big, columns=['data']).reset_index().sort_values(['data','feat_name'], ascending=[False, True]).set_index(['feat_name'])
    return (smal, frame)
	
	
## tdidf Vectorizer with document frequency of a least 3

def answer_five():
    xtrainvectorized = TfidfVectorizer(min_df=3).fit(X_train)
    xtestvectorized = TfidfVectorizer(min_df=3).fit(X_test)
    xtraintransformed = xtrainvectorized.transform(X_train)
    xtesttransformed = xtrainvectorized.transform(X_test)
    clf = MultinomialNB(alpha=0.1)
    model = clf.fit(xtraintransformed, y_train)
    return roc_auc_score(y_test, clf.predict(xtesttransformed))
	
	
## Average Document Length

def answer_six():
    df = spam_data[spam_data['target'] == 1]
        
    to_add = []
    for x in df['text']:
        to_add.append(len(x))
    yes = np.mean(to_add)
    
    df = spam_data[spam_data['target'] == 0]
    
    to_add = []
    for x in df['text']:
        to_add.append(len(x))
    no = np.mean(to_add)
    
    return (no, yes)
	

## Add feature

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
	
	
## tdidf Vectorizer with document frequency of a least 5 + document length

from sklearn.svm import SVC

def answer_seven():
    xtrainvectorized = TfidfVectorizer(min_df=5).fit(X_train)
    xtestvectorized = TfidfVectorizer(min_df=5).fit(X_test)
    xtraintransformed = xtrainvectorized.transform(X_train)
    xtesttransformed = xtrainvectorized.transform(X_test)
    
    to_add = []
    for x in X_train.values:
        to_add.append(len(x))
    final_feature_matrix = add_feature(xtraintransformed, to_add)
    
    to_add = []
    for x in X_test.values:
        to_add.append(len(x))
    final_feature_matrix_test = add_feature(xtesttransformed, to_add)
    
    clf = SVC(C=10000)
    model = clf.fit(final_feature_matrix, y_train)
    
    return roc_auc_score(y_test, clf.predict(final_feature_matrix_test))
	
	
## Average number of digits per document

def answer_eight():  
    df = spam_data[spam_data['target'] == 1]
        
    to_add = []
    for x in df['text']:
        to_add.append(sum(c.isdigit() for c in x))
    yes = np.mean(to_add)
    
    df = spam_data[spam_data['target'] == 0]
    
    to_add = []
    for x in df['text']:
        to_add.append(sum(c.isdigit() for c in x))
    no = np.mean(to_add)

    return (no, yes)
	
	
## tdidf Vectorizer with document frequency of a least 5 
## + document length
## + number of digits per document
## + word n-grams = 3

from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    xtrainvectorized = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    xtestvectorized = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_test)
    xtraintransformed = xtrainvectorized.transform(X_train)
    xtesttransformed = xtrainvectorized.transform(X_test)
    
    to_add = []
    for x in X_train.values:
        to_add.append(len(x))
    final_feature_matrix = add_feature(xtraintransformed, to_add)
    
    to_add = []
    for x in X_train.values:
        to_add.append(sum(c.isdigit() for c in x))
    final_feature_matrix = add_feature(xtraintransformed, to_add)
    
    to_add = []
    for x in X_test.values:
        to_add.append(len(x))
    final_feature_matrix_test = add_feature(xtesttransformed, to_add)
    
    to_add = []
    for x in X_test.values:
        to_add.append(sum(c.isdigit() for c in x))
    final_feature_matrix_test = add_feature(xtesttransformed, to_add)
    
    clf = LogisticRegression(C=100)
    model = clf.fit(final_feature_matrix, y_train)
    
    return roc_auc_score(y_test, clf.predict(final_feature_matrix_test))