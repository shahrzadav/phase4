# -*- coding: cp1256 -*-
from nltk import word_tokenize
import codecs
from nltk import stem
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import time

start = time.time()

"""opening text file and split each words in verses"""
file1=codecs.open("quran-simple.txt",'r',encoding='utf-8')
#TEST_FILE = codecs.open('shz.txt','r', encoding='utf-8')
#p = re.compile(unicode('^????', 'utf-8'), re.U)
text=file1.read()
words=[]
for line in file1:
    print line
    word = line.split()
    print word
    for w in word:
        for i in range(len(w)):
            words[i].append(w)
    
#print words
"""stemm verses"""
for i in range(len(words)):
    for j in range(len(100)):
        stemmer = stem.ISRIStemmer()
        stemmer.stem(words[i][j])

"""deleting stopwords and making tfidf vector and using kmeans algorithm on
verses and geting output """
important_word=[]
for i in range(len(words)):
    for j in range(len(100)):
        transformer = TfidfTransformer()
        trainVectorizerArray = vectorizer.fit_transform().toarray()
        testVectorizerArray = vectorizer.transform(words[i][j]).toarray()
        if words[i][j] not in stopwords.words('arabic'): #getting rid of stopwords
            del words[i][j]
        vectorizer = TfidfVectorizer(lowercase=False, max_df=0.8)
        fs_train = vectorizer.fit_transform(word[i][j])
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(fs_train)
        predict1 = kmeans.predict(fs_train)


print predict1
end = time.time()
print end - start
