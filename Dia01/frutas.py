# %%
import pandas as pd

# %%
df = pd.read_excel("../data/dados_frutas.xlsx")
df

# %%
# Fazendo na mão
filtro_redonda = df["Arredondada"] == 1
filtro_suculenta = df["Suculenta"] == 1
filtro_vermelha = df["Vermelha"] == 1
filtro_doce = df["Doce"] == 1

df[filtro_redonda & filtro_vermelha & filtro_suculenta & filtro_doce]

# %%
# Fazendo a máquina aprender
from sklearn import tree

# %%
feature = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
target = "Fruta"

# %%
X = df[feature]
X  # X são as variaveis independentes, os atributos, variaveis explicativa

# %%
Y = df[target]
Y  # Y é a variável dependete, resposta, alvo

# %%
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, Y)  # Método fit é para o algoritimo tree aprender com os dados

# %%
import matplotlib.pyplot as plt

# %%
plt.figure(dpi=600)  # Melhorar qualidade da imagem o comando dpi

tree.plot_tree(arvore, class_names=arvore.classes_, feature_names=feature, filled=True)
# Quando o nó, está colorido é que ele é puro
# Ou seja, so existe uma tipo de resposta (classe)
# Os últimos nós são chamados de folhas
# Se o nó é branco ele fica branco

# %%
arvore.predict([[0, 1, 1, 1]])  # So te da o resultado

# %%
arvore.predict_proba([[0, 1, 1, 1]])
# lista com a probabilidade de cada classe
# %%
arvore.predict([[1, 1, 1, 1]])

# %%
arvore.predict_proba([[1, 1, 1, 1]])

# %%
probas = arvore.predict_proba([[1, 1, 1, 1]])[0]
# %%
pd.Series(probas, index=arvore.classes_)
# %%
