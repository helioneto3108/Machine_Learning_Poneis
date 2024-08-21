# %%
import pandas as pd

# %%
df = pd.read_csv("../data/dados_pontos.csv", sep=";")
df

# %%
# Separando o modelo entre treino e teste
from sklearn import model_selection

features = df.columns[3:-1]
targert = "flActive"

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df[features], df[targert], test_size=0.2, random_state=42, stratify=df[targert]
)

# O X é representado por letra maíuscula, pois na matematica matriz é
# Letra maíuscula. O vetor é representado por letra minuscula

print("Tx Resposta Treino: ", y_train.mean())
print("Tx Resposta Teste: ", y_test.mean())
# Mostrar que está estratificado, nesse caso nem precisava
# Pois minha target não é um evento raro

# %%
# Verificando missing na minha tabela de treino
X_train.isna().sum()
# %%
# Preenchendo os missing
input_missing_recorrencia = X_train["avgRecorrencia"].max()
# %%
X_train["avgRecorrencia"].fillna(input_missing_recorrencia)

X_test["avgRecorrencia"].fillna(input_missing_recorrencia)
# Na base de teste eu nunca mexo. Nunca pego info dela.
# Ela é so para eu testar o model. Ela é como se fosse um dado novo
# %%
# Treinando o modelo de arvore
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=100, random_state=42)
arvore.fit(X_train, y_train)
# max_leaf_node é a quantidade de valores minima em um no

# %%
# Fazendo as métricas
from sklearn import metrics

# Base de treino - Previsão
tree_pred_train = arvore.predict(X_train)
acc_tree_train = metrics.accuracy_score(y_train, tree_pred_train)
print("Árovre Train ACC: ", acc_tree_train)

# Base de teste - Previsão
tree_pred_test = arvore.predict(X_test)
acc_tree_test = metrics.accuracy_score(y_test, tree_pred_test)
print("Árovre Test ACC: ", acc_tree_test)

# Base de treino - Curva Roc
tree_proba_train = arvore.predict_proba(X_train)[:, 1]
auc_tree_train = metrics.roc_auc_score(y_train, tree_proba_train)
print("Árovre Train AUC: ", auc_tree_train)

# Base de teste - Curva Roc
tree_proba_test = arvore.predict_proba(X_test)[:, 1]
auc_tree_test = metrics.roc_auc_score(y_test, tree_proba_test)
print("Árovre Test AUC: ", auc_tree_test)

# Agora é so ajustar as métricas lá encima na arvore para ver o
# modelo com o maior desempenho
# %%
arvore.predict_proba(X_train)  # Para visualizar o que é isso

# %%
