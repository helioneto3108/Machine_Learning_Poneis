# %%
import pandas as pd

from sklearn import model_selection
from sklearn import ensemble
from sklearn import pipeline

from feature_engine import imputation

import scikitplot as skplt
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('../data/dados_pontos.csv', sep = ';')
df

# %%
features = df.columns[3:-1].to_list()
target = 'flActive'
# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size = 0.2,
                                                                    stratify = df[target],
                                                                    random_state = 42)

print("Tx resposta train: ", y_train.mean())
print("Tx resposta test: ", y_test.mean())
# %%
X_train.isna().sum()
# %%
# Pipeline
# Pegando maior numerod e recorrencia
max_recorrencia = df['avgRecorrencia'].max()

input_max_recorrencia = imputation.ArbitraryNumberImputer(variables = ['avgRecorrencia'],
                                                          arbitrary_number = max_recorrencia)

random_forest = ensemble.RandomForestClassifier(random_state = 42)

parms = {
    "n_estimators": [200, 300, 400, 500],
    "min_samples_leaf": [10, 20, 50, 100]
        }

grid = model_selection.GridSearchCV(random_forest, 
                                    param_grid = parms,
                                    scoring = 'roc_auc',
                                    n_jobs = -1)

model = pipeline.Pipeline([
    ('imput', input_max_recorrencia),
    ('model', grid)
])
# %%
# Treinando o modelo
model.fit(X_train, y_train)
# %%
y_test_proba = model.predict_proba(X_test)
y_test_proba

# %%
# criando DataFrame com a prob de ser 1 (ativo)
df_ativo = pd.DataFrame({
    "flActive": y_test,
    "Prob_modelo": y_test_proba[:, 1]
})
df_ativo.to_excel('../data/dados_Ks.xlsx', index = False)
# %%
plt.figure(dpi = 600)

skplt.metrics.plot_ks_statistic(y_test, y_test_proba)
plt.show()
# %%
