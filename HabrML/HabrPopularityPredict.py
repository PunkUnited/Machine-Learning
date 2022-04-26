import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import datetime
from sklearn.preprocessing import MultiLabelBinarizer


stop_words = stopwords.words(['english', 'russian'])
temmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
last = datetime.datetime(2006, 9, 1)


def date_to_num(dates):
    dates_nums = []
    for date in dates:
        date = date.split(',')[0]
        dates_nums.append((datetime.datetime.strptime(date, '%Y-%m-%d') - last).days)
    return dates_nums


def clean_text(texts):
    preprocessed = []
    for text in texts:
        text = text.lower()
        regular = r'[\*+\#+\№\"\-+\+\=—!+\?+\&\^\.+“\;\,+\>+\(\)\/+\”:\\+]'
        regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)'
        text = re.sub(regular, '', text)
        text = re.sub(regular_url, r'', text)
        text = re.sub(r'(\d+\s\d+)|(\d+)', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = word_tokenize(text)
        text = [word for word in text if word not in stop_words]
        text = [lemmatizer.lemmatize(word) for word in text]
        preprocessed.append(text)
    return preprocessed


def hubs_to_category(text):
    listed = []
    for listing in text:
        one_list = []
        for one in listing:
            one_list.append(one['hubName'].strip())
        listed.append(one_list)
    trans = pd.DataFrame({'list': listed})
    value = pd.DataFrame(
        {'unique': trans.apply(lambda x: pd.Series(x['list']), axis=1).stack().reset_index(level=1, drop=True)})
    unique = []
    for val in value.value_counts()[:400].keys():
        unique.append(val[0])
    mlb = MultiLabelBinarizer(sparse_output=False, classes=unique)
    return list(mlb.fit_transform(listed))


def tags_to_category(text):
    listed = []
    for listing in text:
        one_list = []
        for one in listing:
            one_list.append(one['tagName'].strip())
        listed.append(one_list)
    trans = pd.DataFrame({'list': listed})
    value = pd.DataFrame(
        {'unique': trans.apply(lambda x: pd.Series(x['list']), axis=1).stack().reset_index(level=1, drop=True)})
    unique = []
    for val in value.value_counts()[:400].keys():
        unique.append(val[0])
    mlb = MultiLabelBinarizer(sparse_output=False, classes=unique)
    return list(mlb.fit_transform(listed))


import pandas as pd
import numpy as np


Habr = pd.read_json('habr.json')
Habr = Habr[:50]
del[Habr['articleLink'], Habr['author']]


Habr['text'] = clean_text(Habr['text'])
# Habr['title'] = clean_text(Habr['title'])
# Habr['publicationTime'] = date_to_num(Habr['publicationTime'])
# Habr['hubs'] = hubs_to_category(Habr['hubs'])
# Habr['tags'] = tags_to_category(Habr['tags'])
print(Habr['text'][0])


# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
#
#
# x_train, x_test, y_train, y_test = train_test_split(np.hstack(Habr['text']), Habr['claps'], test_size=0.3)
#
#
# vectorizer = TfidfVectorizer(min_df=4)
# vec = vectorizer.fit_transform(x_train['text']).toarray()
# vec2 = vectorizer.transform(x_test['text']).toarray()
# pca = PCA(n_components=499)
# vec = pca.fit_transform(vec)
# vec2 = pca.transform(vec2)
#
#
# vectorizer2 = TfidfVectorizer(min_df=4)
# vectorizer2.fit(x_train['title'])
# vec3 = vectorizer2.transform(x_train['title']).toarray()
# vec4 = vectorizer2.transform(x_test['title']).toarray()
# pca3 = PCA(n_components=499)
# vec3 = pca3.fit_transform(vec3)
# vec4 = pca3.transform(vec4)
#
#
# from sklearn.metrics import mean_absolute_error as MAE
# # import optuna
#
#
# # def objective(trial):
# #     params = {
# #   'learning_rate':trial.suggest_float('learning_rate', 0.1, 1, step=0.05),
# #   'max_depth':trial.suggest_int('max_depth', 2, 5),
# #   'min_samples_split':trial.suggest_int('min_samples_split', 2, 5),
# #     'n_estimators':trial.suggest_int('n_estimators',100,300, step = 100),
# #   'min_samples_leaf':trial.suggest_int('min_samples_leaf', 2, 5)}
# #     x_tr, x_te, y_tr, y_te = train_test_split(np.hstack([vec, vec3]), y_train, test_size=0.3)
# #     model1 = GradientBoostingRegressor(**params)
# #     model1.fit(x_tr, y_tr)
# #     Predict_Boosting_Scaled = model1.predict(x_te)
# #     return MAE(y_te, Predict_Boosting_Scaled)
#
#
# # study = optuna.create_study()
# # study.optimize(objective, n_trials=15)
#
#
# # print(study.best_params)
#
#
# from sklearn.ensemble import GradientBoostingRegressor
#
#
# clf = make_pipeline(StandardScaler, GradientBoostingRegressor())
#
#
# rfc = GradientBoostingRegressor(**{'learning_rate': 0.25,
#                                    'max_depth': 5,
#                                    'min_samples_split': 3,
#                                    'n_estimators': 100,
#                                    'min_samples_leaf': 2})
# rfc.fit(np.hstack([vec, vec3]), y_train)
# Predict = rfc.predict(np.hstack([vec2, vec4]))
# print(MAE(y_test, Predict))
# # output = pd.DataFrame({'id': [i for i in range(3756, 4256)], 'claps': Predict})
# # output['claps'] = round(output['claps'].apply(lambda x: max(0, x)), 0)
# # output.to_csv('answer2.csv', index=False)
