import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
# import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_absolute_error as MAPE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler,GradientBoostingRegressor())
loss = ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
min_samples_leaf = [1, 2, 4, 6, 8]
min_samples_split = [2, 4, 6, 10]
max_features = ['auto', 'sqrt', 'log2', None]
hyperparameter_grid = {'learning_rate': 0.25, 'max_depth': 5, 'min_samples_split': 3, 'n_estimators': 100, 'min_samples_leaf': 2}


test = pd.read_csv('Data_for_test.csv')
train = pd.read_csv('Data_for_train.csv')

vectorizer = TfidfVectorizer(min_df=4)
vectorizer.fit(pd.concat([train['text'],test['text']]))
vec = vectorizer.transform(train['text']).toarray()
vec2 = vectorizer.transform(test['text']).toarray()
pca = PCA(n_components=499)
vec = pca.fit_transform(vec)
vec2 = pca.transform(vec2)

vectorizer2 = TfidfVectorizer(min_df=4)
vectorizer2.fit(pd.concat([train['title'],test['title']]))
vec3 = vectorizer2.transform(train['title']).toarray()
vec4 = vectorizer2.transform(test['title']).toarray()
pca3 = PCA(n_components=499)
vec3 = pca3.fit_transform(vec3)
vec4 = pca3.transform(vec4)
#
#
# def objective(trial):
#     params = {
#   'learning_rate':trial.suggest_float('learning_rate', 0.1, 1, step=0.05),
#   'max_depth':trial.suggest_int('max_depth', 2, 5),
#   'min_samples_split':trial.suggest_int('min_samples_split', 2, 5),
#     'n_estimators':trial.suggest_int('n_estimators',100,300, step = 100),
#   'min_samples_leaf':trial.suggest_int('min_samples_leaf', 2, 5)}
#     x_tr, x_te, y_tr, y_te = train_test_split(np.hstack([vec, vec3]), train['claps'], test_size=0.3)
#     model1 = GradientBoostingRegressor(**params)
#     model1.fit(x_tr, y_tr)
#     Predict_Boosting_Scaled = model1.predict(x_te)
#     return MAPE(y_te, Predict_Boosting_Scaled)
#
#
# study = optuna.create_study()
# study.optimize(objective, n_trials=15)
#
# print(study.best_params)
#
rfc = GradientBoostingRegressor(**{'learning_rate': 0.25, 'max_depth': 5, 'min_samples_split': 3, 'n_estimators': 100, 'min_samples_leaf': 2})
rfc.fit(np.hstack([vec, vec3]), train['claps'])
y_pred = rfc.predict(np.hstack([vec2, vec4]))
output = pd.DataFrame({'id': test['id'], 'claps': y_pred})
output['claps'] = round(output['claps'].apply(lambda x: max(0, x)), 0)
output.to_csv('answer2.csv', index=False)