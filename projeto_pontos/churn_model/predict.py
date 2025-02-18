# %%

import pandas as pd

model = pd.read_pickle("modelo_rf.pkl")
model
# %%

df = pd.read_csv('../data/dados_pontos.csv',sep=';')
df.head()
# %%

X = df[model['features']]
predict_proba = model['model'].predict_proba(X)[:,1]
predict_proba

df['prob_active'] = predict_proba
df[['Name','prob_active']]
# %%
model
# %%
