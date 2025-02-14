# %%
import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../data/dados_pontos.csv",
                 sep=";")
df
# %%

features = df.columns[3:-1]
target = 'flActive'

# stratify serve para equalizar o target de teste e treino (mesma medida de target)
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

print('Tx Resposta Treino',y_train.mean())
print('Tx Resposta Treino',y_test.mean())


# %%
X_train.isna().sum().T
input_avgRecorrencia = X_train['avgRecorrencia'].max()

X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(input_avgRecorrencia)
# só trabalha com a base de treino
X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(input_avgRecorrencia)
# %%
from sklearn import tree
from sklearn import metrics

# treinamos o algoritmo
# min sample leaf é o número mínimo de dados que tem que ter em cada nó
arvore = tree.DecisionTreeClassifier(max_depth=10, 
                                     min_samples_leaf=50,
                                     random_state=42)
arvore.fit(X_train,y_train)

# base treino - prevendo na própria base
tree_pred_train = arvore.predict(X_train)
tree_acc_train = metrics.accuracy_score(y_train,tree_pred_train)
print('Árvore Train ACC:', tree_acc_train)

# prevendo na base de teste
tree_pred_test = arvore.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test, tree_pred_test)
print('Árvore Test ACC:', tree_acc_test)

# PREDICT PROBA

tree_proba_train = arvore.predict_proba(X_train)[:,1]
tree_auc_train = metrics.roc_auc_score(y_train,tree_proba_train)
print('Árvore Train AUC:', tree_auc_train)

# prevendo na base de teste
tree_proba_test = arvore.predict_proba(X_test)[:,1]
tree_auc_test = metrics.roc_auc_score(y_test, tree_proba_test)
print('Árvore Test AUC:', tree_auc_test)

# %%
# foi melhor do que a hipótese que tínhamos que ninguém voltaria
# como tenho um vetor de 0s e 1s a minha média é um valor entre 0s e 1s
# soma de 1s / total de ids
1 - y_test.mean()

# %%
