import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import optuna
test = pd.read_csv('TrueTest.csv')
train = pd.read_csv('TrueTrain.csv')
vyb = pd.concat([train, test])
TFTest = pd.read_csv('TFTest.csv')
TFTrain = pd.read_csv('TFTrain.csv')
TFvyb = pd.concat([TFTrain, TFTest])
Valuetest = pd.read_csv('ValueTest.csv')
Valuetrain = pd.read_csv('ValueTrain.csv')
vybValue = pd.concat([Valuetrain, Valuetest])
LargeTest = pd.read_csv('LargeTest.csv')
LargeTrain = pd.read_csv('LargeTrain.csv')
vybLarge = pd.concat([LargeTrain, LargeTest])
y = pd.read_csv('yTrain.csv')
for x in vybLarge.keys():
    vybLarge[x] = vybLarge[x].astype('category')
    vybLarge[x] = vybLarge[x].cat.codes
x = [i for i in vyb.keys()]
vyb = pd.get_dummies(vyb, columns=x, prefix_sep="_", drop_first=True)

vyb = pd.concat([vyb, vybValue, TFvyb], axis=1)
vyb = vyb.fillna(0)
test = vyb[1050815:]
train = vyb[:1050815]
pca = PCA(n_components=50)
train = pca.fit_transform(train)
test = pca.transform(test)
del[vyb]


def objective(trial):
    params = {
  'bootstrap':trial.suggest_int('bootstrap', 0, 1),
  'max_features':trial.suggest_int('max_features', 1, 3, step=1),
  'max_depth':trial.suggest_int('max_depth', 1, 7, step=1),
  'min_samples_split':trial.suggest_int('min_samples_split', 2, 7, step=1),
  'n_estimators':trial.suggest_int('n_estimators', 100, 1000, step=100),
  'min_samples_leaf':trial.suggest_float('min_samples_leaf', 0.1, 0.5, step=0.1)
             }
    x_tr, x_te, y_tr, y_te = train_test_split(train, y, test_size=0.2)
    model1 = RandomForestRegressor(**params)
    model1.fit(x_tr, y_tr)
    predict = model1.predict(x_te)
    return MAE(y_te, predict)


study = optuna.create_study()
study.optimize(objective, n_trials=15)

model2 = RandomForestRegressor()
model2.fit(train, y)
y_pred = model2.predict(test)

output = pd.DataFrame({'id': [i for i in range(1050815, 1090815)], 'mean_salary': y_pred})
output.to_csv('answer2.csv', index=False)