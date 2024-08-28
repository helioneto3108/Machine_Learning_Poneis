# %%
import pandas as pd
# %%
model = pd.read_pickle('modelo_rf.pkl')
model
# %%
df = pd.read_csv("../data/dados_pontos.csv", sep = ';')
df
# %%
X = df[model['features']]
predict_proba = model['model'].predict_proba(X)
predict_proba
# %%
df['prob_active'] = predict_proba[:, 1]
df[['Name', 'prob_active']]
# %%
