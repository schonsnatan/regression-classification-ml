# %%

import pandas as pd
import os
# %%

model = pd.read_pickle("modelo_rf.pkl")
model
# %%

df = pd.read_csv('../../data/dados_pontos.csv',sep=';')
df.head()
# %%

dt = '2024-05-08'

with open('etl_model.sql', 'r') as open_file:
    query = open_file.read()

query = query.format(date=dt)

print(query)
# %%

X = df[model['features']]
predict_proba = model['model'].predict_proba(X)[:,1]
predict_proba

df['prob_active'] = predict_proba
df[['Name','prob_active']]
# %%
model
# %%
os.chdir('../projeto_pontos/churn_model')# %%

# %%
print(os.listdir())  # Lista os arquivos e pastas no diretório atual
# %%
