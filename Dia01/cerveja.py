# %%
import pandas as pd

# %%
df = pd.read_excel("../data/dados_cerveja.xlsx")
df

# %%
features = ["temperatura", "copo", "espuma", "cor"]
target = "classe"

# %%
x = df[features]
y = df[target]

# %%
x = x.replace({"mud": 1, "pint": 0, "sim": 1, "não": 0, "escura": 1, "clara": 0})
x
# Não é o jeito mais recomendado, so foi feito para fins didáticos

# %%
from sklearn import tree

# %%
arvore = tree.DecisionTreeClassifier()
arvore.fit(x, y)

# %%
import matplotlib.pyplot as plt

# %%
plt.figure(dpi=600)

tree.plot_tree(arvore, class_names=arvore.classes_, feature_names=features, filled=True)
# filled = true é para colorir as caixas
# %%
