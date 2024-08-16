# %%
import pandas as pd

# %%
df = pd.read_parquet("../data/dados_clones.parquet")
df

# %%
# Análise descritiva
# Fazendo analise sobre a media das massas e estratura para ver se
# Diferencia entre os aptos e os defeituosos
df.groupby(["Status "])[["Massa(em kilos)", "Estatura(cm)"]].mean()

# %%
# Fazendo isso, pois com true e false eu consigo fazer operações
# Pois o False é igual a zero e o True igual a um
df["Status_bool"] = df["Status "] == "Apto"
df

# %%
df.groupby(["Distância Ombro a ombro"])["Status_bool"].mean()

# %%
df.groupby(["Tamanho do crânio"])["Status_bool"].mean()

# %%
df.groupby(["Tamanho dos pés"])["Status_bool"].mean()
# Percebemos que a taxa de aptos é quase a mesma para todos os tipos
# Em todas as variáveis explicativas, Atribuitos

# %%
df.groupby(["General Jedi encarregado"])["Status_bool"].mean()
# Aqui podemos perceber que o Yoda e a Shaak Ti tem taxas de aptos
# Bem menores que os outros

# %%
features = [
    "Massa(em kilos)",
    "Estatura(cm)",
    "Distância Ombro a ombro",
    "Tamanho do crânio",
    "Tamanho dos pés",
]

cat_features = df[features].select_dtypes(include="object").columns.to_list()
cat_features  # Pegando os atributos categoricos para transformá-los
# Para numérico, para ser possível realizar a modelagem

# %%
X = df[features]
X

# %%
# Transformação de categoria para numérico
from feature_engine import encoding

onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)

X = onehot.transform(X)
X

# %%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X, df["Status "])

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=600)

tree.plot_tree(
    arvore, class_names=arvore.classes_, feature_names=X.columns, filled=True
)

# Esse processo é muito bom para entender meus dados

# %%
