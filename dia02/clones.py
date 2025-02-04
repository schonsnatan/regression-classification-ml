# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from feature_engine import encoding


df = pd.read_parquet("../data/dados_clones.parquet")
df
# %%
## Como podemos descobrir onde está o problema?

'''
Nesse caso podemos usar a correlação entre cada característica
e o target, que seria nosso "status" para descobrir o que está
influenciando
'''
# <Estatística descritiva>

# Calculando a média da estratura e massa para clones aptos e defeituosos
df.groupby(['Status '])[['Estatura(cm)','Massa(em kilos)']].mean()

# %%

df['Status_bool'] = df['Status '] == 'Apto'
df

# %%

# Nenhuma característica do clone impacta significativamente em estar apto ou não
df.groupby(['Distância Ombro a ombro'])['Status_bool'].mean()
df.groupby(['Tamanho do crânio'])['Status_bool'].mean()
df.groupby(['Tamanho dos pés'])['Status_bool'].mean()	

# %%

# (Será?) Acionar o time de "people" para melhorar a liderança de Yoda e Shaak Ti
df.groupby(['General Jedi encarregado'])['Status_bool'].mean()
# %%

features = [
            'Estatura(cm)',
            'Massa(em kilos)',
            'Distância Ombro a ombro',
            'Tamanho do crânio',
            'Tamanho dos pés'
            ]

cat_features = ['Distância Ombro a ombro',
                'Tamanho do crânio',
                'Tamanho dos pés']

X = df[features]

# %%

'''
Tenemos que transformar as variáveis categóricas em numéricas.
Para isso usamos (one hot encoding)

ONE HOT ENCODER

'''

onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)

X = onehot.transform(X)

# %%

'''
O problema real são os clones que são iguais ou mais leves que 
83kg e que possuem estatura entre 180.555 e 180.245 cm, caso o 
clone possuir  essas características a chance dele ser defeituoso
é de 72%! Max depth define a profundidade da árvore
'''
arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X,df['Status '])


plt.figure(dpi=600)

tree.plot_tree(arvore, 
               class_names=arvore.classes_, 
               feature_names=X.columns,
               filled=True,
               )

# %%
