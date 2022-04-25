import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    regular = r'[\*+\#+\№\"\-+\+\=—!+\?+\&\^\.+“\;\,+\>+\(\)\/+\”:\\+]'
    regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)'
    text = re.sub(regular, '', text)
    text = re.sub(regular_url, r'', text)
    text = re.sub(r'(\d+\s\d+)|(\d+)','', text)
    text = re.sub(r'\s+', ' ', text)
    return text

preprocessed_train_text = []
preprocessed_train_title = []
preprocessed_test_text = []
preprocessed_test_title = []
stop_words = stopwords.words(['english', 'german', 'italian', 'russian',])
temmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

test = pd.read_csv('kaggle1/articles_test.csv')
train = pd.read_csv('kaggle1/articles_train.csv')

for text in train['text']:
    text = clean_text(text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    preprocessed_train_text.append(" ".join(text))
for text in train['title']:
    text = clean_text(text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    preprocessed_train_title.append(" ".join(text))
for text in test['text']:
    text = clean_text(text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    preprocessed_test_text.append(" ".join(text))
for text in test['title']:
    text = clean_text(text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    preprocessed_test_title.append(" ".join(text))
train['text'], train['title'], test['text'], test['title'] = preprocessed_train_text, preprocessed_train_title,\
                                                             preprocessed_test_text, preprocessed_test_title
del[train['link'],test['link']]
train.to_csv('Kaggle1/Data_for_train.csv')
test.to_csv('Kaggle1/Data_for_test.csv')