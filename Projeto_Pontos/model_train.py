# %%
import pandas as pd

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn import tree, linear_model, ensemble, naive_bayes

from feature_engine import imputation

# %%
df = pd.read_csv("../data/dados_pontos.csv", sep=";")
df

# %%
features = df.columns.tolist()[3:-1]
features  # Matriz de colunas

# %%
target = "flActive"  # Vetor de resposta

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df[features], df[target], test_size=0.2, random_state=42, stratify=df[target]
)

print("Tx Resposta treino: ", y_train.mean())
print("Tx Resposta Teste: ", y_test.mean())

# %%
# Descobrindo colunas que tem valores nulos
X_train.isna().sum().T

# %%
max_avgRecorrencia = X_train["avgRecorrencia"].max()
max_avgRecorrencia

# %%
# Criando as imputações para os dados faltantes
imputacao_max = imputation.ArbitraryNumberImputer(
    variables=["avgRecorrencia"], 
    arbitrary_number=max_avgRecorrencia
)
imputacao_max

# %%
# Fazendo o modelo e setando o hiperparametros
model = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=50)
model

# %%
# Criando meu pipeline para aplicar isso nos novos dados
meu_pipeline = pipeline.Pipeline([("input_max", imputacao_max), ("model", model)])
meu_pipeline

# %%
# Treinando meu modelo atraves do pipeline
meu_pipeline.fit(X_train, y_train)

# %%
# Fazendo as predições
# Dados treinos
y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]

# Dados Testes
y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)[:,1]

# %%
# Métricas 
# Acurácia
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)

print('Acurácia base train: ', acc_train)
print('Acurácia base test: ', acc_test)

# %%
# Curva Roc
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print('AUC base train: ', auc_train)
print('AUC base test: ', auc_test)

# %%
# Fazendo grid search
model = ensemble.RandomForestClassifier()

meu_pipeline = pipeline.Pipeline([
    ("input_max", imputacao_max), 
    ("model", model)
    ])

params = {
    "model__n_estimators": [100,150,250,500], # Numeros de arvores no randomforest
    "model__min_samples_leaf": [10,20,30,50,100],
}

grid = model_selection.GridSearchCV(meu_pipeline,
                                    param_grid = params,
                                    n_jobs = -1, # Quantos processadores da maquina vc quer que utilize, -1 = todos
                                    scoring = "roc_auc")
# Curva roc é melhor pois ela da a probabilidade

grid.fit(X_train, y_train)

# %%
# Ver os resultados -> foram 20, (4x5)
# Porém em cada modelo eu realizo um validaçao cruzada 3 vezes
# Ou seja foram 60 modelos. (20x3) > 60
pd.DataFrame(grid.cv_results_)
# Melhor modelo é o setimo -> 20 e 250
# %%
# Vendo os melhores parametros
grid.best_params_

# %%
# Fazendo as predições para o modelo com grid
# Dados treinos
y_train_predict = grid.predict(X_train)
y_train_proba = grid.predict_proba(X_train)[:,1]

# Dados Testes
y_test_predict = grid.predict(X_test)
y_test_proba = grid.predict_proba(X_test)[:,1]

# %%
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)

print('Acurácia base train: ', acc_train)
print('Acurácia base test: ', acc_test)

# %%
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print('AUC base train: ', auc_train)
print('AUC base test: ', auc_test)

# %%
# Otimizando mais o proceso, pois no meu pipeline estou tendo mais gastos
# de processamento, pois além do modelo estou fazendo grid dos inputs

# Pipeline otimizado
model = ensemble.RandomForestClassifier()

params = {
    "n_estimators": [100,150,250,500], # Numeros de arvores no randomforest
    "min_samples_leaf": [10,20,30,50,100],
}


grid = model_selection.GridSearchCV(model,
                                    param_grid = params,
                                    n_jobs = -1,
                                    scoring = "roc_auc")

meu_pipeline = pipeline.Pipeline([
    ("input_max", imputacao_max), 
    ("model", grid)
    ])

meu_pipeline.fit(X_train, y_train) # Muito menos custoso

# %%
# Fazendo as predições para o modelo com grid menos custos
# Dados treinos
y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]

# Dados Testes
y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)[:,1]

# %%
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)

print('Acurácia base train: ', acc_train)
print('Acurácia base test: ', acc_test)

# %%
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print('AUC base train: ', auc_train)
print('AUC base test: ', auc_test)

# %%
