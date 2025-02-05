# %%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import tree

df = pd.read_excel("../data/dados_cerveja_nota.xlsx")
df

# %%
plt.plot(df['cerveja'],df['nota'], 'o')
plt.grid(True)
plt.title('Relação Nota vs Cerveja')
plt.ylim(0,11)
plt.xlim(0,11)
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.show()

# %%
# MODELO DE REGRESSÃO

reg = linear_model.LinearRegression()

# dois colchetes para passar uma matriz e não como vetor (Series)
reg.fit(df[['cerveja']], df['nota'])

# %%
# coeficiente é uma lista (caso fosse uma regressao multipla)
a, b = reg.intercept_, + reg.coef_[0]
print(f'a={a}; b={b}')

# %%
X = df[['cerveja']].drop_duplicates()
y_estimado = reg.predict(X)
y_estimado
# %%
plt.plot(df['cerveja'],df['nota'], 'o')
plt.plot(X,y_estimado,'-')
plt.grid(True)
plt.title('Relação Nota vs Cerveja')
plt.ylim(0,11)
plt.xlim(0,11)
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.show()
# %%

# MODELO DE ÁRVORE DE DECISÃO

# max_depth é o hiperparametro, profundidade da arvore (3 já é overfitting)
arvore = tree.DecisionTreeRegressor(max_depth=2)
arvore.fit(df[['cerveja']],df['nota'])

y_estimado_arvore = arvore.predict(X)
y_estimado_arvore

# %%

# visualização dos modelos lineares

plt.plot(df['cerveja'],df['nota'], 'o')
plt.plot(X,y_estimado,'-')
plt.plot(X,y_estimado_arvore,'-')
plt.grid(True)
plt.title('Relação Nota vs Cerveja')
plt.ylim(0,11)
plt.xlim(0,11)
plt.xlabel("Cerveja")
plt.ylabel("Nota")
plt.legend(['Pontos','Regressão','Árore'])
plt.show()
# %%
