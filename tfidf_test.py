import nltk
#nltk.download('wordnet')
from nltk.corpus import gutenberg
from nltk.stem import WordNetLemmatizer
#from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
import string

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(word) for word in tokens if word not in string.punctuation and "'" not in word]

# Initialize a TfidfVectorizer object using the pre-computed tf-idf scores from NLTK
tfidf = TfidfVectorizer(tokenizer=tokenize)

#print(len(reuters.raw().split(" ")))
tfidf_matrix = tfidf.fit_transform([gutenberg.raw()])
print(tfidf_matrix.get_shape())
vocab = tfidf.vocabulary_
# Testing stuff
#print(tfidf_matrix[0, vocab["person"]])
#print(tfidf_matrix[0, vocab["layer"]])
#print(tfidf_matrix[0, vocab["bitcoin"]])



