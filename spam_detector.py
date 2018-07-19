import sys
import pandas as pd
import _pickle as cPickle
from sklearn.feature_extraction.text import TfidfVectorizer


input_data = sys.argv[1]
clf = cPickle.load(open('spam_detector.pkl','rb'))
data = list(pd.read_table(input_data))
vect = cPickle.load(open('TfidfVectorizer.pkl','rb'))


ans = clf.predict(vect.transform(data))[0]
if(ans==1):
	print("spam")
else:
	print("ham")	

