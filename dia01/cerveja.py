# %%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_excel("../data/dados_cerveja.xlsx")
df

# %%
## Como podemos fazer a máquina aprender?

features = ['temperatura','copo','espuma','cor']
target = 'classe'

X = df[features]
y = df[target]

# %%

# alterando para numéricos
X = X.replace({
    'mud':1,'pint':0,
    'sim':1,'não':0,
    'escura':1,'clara':0,
})
X
# %%

arvore = tree.DecisionTreeClassifier()
arvore.fit(X,y)

plt.figure(dpi=600)

tree.plot_tree(arvore,
               class_names = arvore.classes_.tolist(),
               feature_names = features,
               filled=True)
# %%
probas = arvore.predict_proba([[-5,1,0,1]])[0]
pd.Series(probas, index=arvore.classes_)
# %%
