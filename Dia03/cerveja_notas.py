# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_excel("../data/dados_cerveja_nota.xlsx")
df

# %%
# Criando gráfico
plt.plot(df["cerveja"], df["nota"], "o")  # 'o' -> estilo do ponto
plt.grid(True)
plt.title("Relação quantidade de cerveja e nota")
plt.ylim(0, 12)
plt.xlim(0, 10)
plt.xlabel("Quantidade cerveja")
plt.ylabel("Nota")

# %%
from sklearn import linear_model

# %%
reg = linear_model.LinearRegression()
reg.fit(df[["cerveja"]], df["nota"])
# Tenho que colocar dois [] quando estou passando o argumento
# De features no fit, pois ele esperá uma matriz, ou um dataframe

# %%
# Serie ou coluna
df["cerveja"]

# %%
# dataframe ou matriz
df[["cerveja"]]

# %%
# Parametros encontrados
a, b = reg.intercept_, reg.coef_[0]
print(f"constate = {a} e parametro 1 = {b}")
# Coloco o .coef_[0], pois ele me retorna a lista com todos os
# Parametros, se tivesse 2 parametros ele me retonaria dois

# %%
# Criando a linha no gráfico
X = df[["cerveja"]].drop_duplicates()

y_estimado = reg.predict(X)
y_estimado
# %%
# Plotando ela junto do gráfico
plt.plot(df["cerveja"], df["nota"], "o")  # 'o' -> estilo do ponto
plt.plot(X, y_estimado, "-")
plt.grid(True)
plt.title("Relação quantidade de cerveja e nota")
plt.ylim(0, 12)
plt.xlim(0, 10)
plt.xlabel("Quantidade cerveja")
plt.ylabel("Nota")
plt.show()  # Para mostrar
# %%
# Modelo de arvore
from sklearn import tree as tree

arvore = tree.DecisionTreeRegressor(max_depth=2)
arvore.fit(df[["cerveja"]], df["nota"])

y_estimado_arvore = arvore.predict(X)

# %%
# Plotando gráfico completo
plt.figure(dpi=500)
plt.plot(df["cerveja"], df["nota"], "o")  # 'o' -> estilo do ponto
plt.plot(X, y_estimado, "-")
plt.plot(X, y_estimado_arvore, "-")
plt.grid(True)
plt.title("Relação quantidade de cerveja e nota")
plt.ylim(0, 12)
plt.xlim(0, 10)
plt.xlabel("Quantidade cerveja")
plt.ylabel("Nota")
plt.legend(["Notas", "Regressão Linear", "Árvore de Decisão"])
plt.show()

# %%
plt.figure(dpi=600)
tree.plot_tree(arvore, filled=True)

# %%
