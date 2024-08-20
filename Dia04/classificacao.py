# %%
import pandas as pd

# %%
df = pd.read_excel("../data/dados_cerveja_nota.xlsx")
df

# %%
# Criando uma coluna para saber se foi aprovado ou não
df["Aprovado"] = df["nota"] >= 5
df

# %%
from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)

# Aqui o modelo aprende
reg.fit(df[["cerveja"]], df["Aprovado"])
# Lembrando que no argumento X eu devo passar uma matriz de pelo
# Menos uma coluna. No Y um vetor que são as respostas.
# X = lista de variaveis explicativas. Y = Variavel resposta

# Aqui o modelo preve
reg_predict = reg.predict(df[["cerveja"]])
reg_predict

# %%
from sklearn import metrics

# Acuracia do meu modelo
reg_acc = metrics.accuracy_score(df["Aprovado"], reg_predict)
reg_acc  # Meu modelo está acertando 86% dos meus dados

# %%
# Precisão do modelo
reg_prec = metrics.precision_score(df["Aprovado"], reg_predict)
print(reg_prec)

# %%
# Matriz de confusão
reg_conf = metrics.confusion_matrix(df["Aprovado"], reg_predict)
reg_conf = pd.DataFrame(reg_conf, index=["True", "False"], columns=["True", "False"])
reg_conf

# A acuracia vem da matriz de

# %%
# Fazendo para arvore
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)

arvore.fit(df[["cerveja"]], df["Aprovado"])

arvore_predict = arvore.predict(df[["cerveja"]])
arvore_predict

# %%
arvore_acc = metrics.accuracy_score(df["Aprovado"], arvore_predict)
arvore_acc
# A arvore teve um desempenho melhor que a regressão logistica
# Ao olharmos a acuracia

# %%
arvore_prec = metrics.precision_score(df["Aprovado"], arvore_predict)
print(arvore_prec)

# %%
# Fazendo para naive
from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()

nb.fit(df[["cerveja"]], df["Aprovado"])

nb_predict = nb.predict(df[["cerveja"]])
nb_predict
# %%
nb_acc = metrics.accuracy_score(df["Aprovado"], nb_predict)
nb_acc

# %%
# Naive bayes teve mesmo desempenho em acuracia que a regressão logistica
# Se quiser ver mais afundo fazer matriz de decisão para cada.
