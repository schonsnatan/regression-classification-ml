# %%

import pandas as pd
from sklearn import metrics

df = pd.read_excel("../data/dados_cerveja_nota.xlsx")
df

# %%

df['Aprovado'] = df['nota'] >= 5

features = ['cerveja']
target = 'Aprovado'

# %%

# Testando com REGRESSÃO LOGÍSTICA
from sklearn import linear_model

reg = linear_model.LogisticRegression(penalty=None, 
                                      fit_intercept=True)
# preciso de lista de features
features = ['cerveja']
target = 'Aprovado'

# aqui o modelo aprende e é ajustado (observado)
reg.fit(df[features],df[target])

# aqui o modelo prevê (previsto)
reg_predict = reg.predict(df[features])
reg_predict

# %%

# dado real vs modelo (acuracia)
reg_acc = metrics.accuracy_score(df[target], reg_predict)
print("Acuracia Reg Log:", reg_acc)

# precisão
reg_precision = metrics.precision_score(df[target], reg_predict)
print("Precisao Reg Log:", reg_precision)

# recall
reg_recall = metrics.recall_score(df[target], reg_predict)
print("Recall Reg Log:", reg_recall)

# acuracia são os acertos da matriz de confusão
# dividos pelo total (13/15)

# aqui temos 8 acertos para True, 5 para False, 1 FP e 1 FN
reg_conf = metrics.confusion_matrix(df[target], reg_predict)
reg_conf = pd.DataFrame(reg_conf, index=['False','True'], 
                        columns=['False','True'])
reg_conf
# %%

# Testando com a ÁRVORE DE DECISÃO
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=2)
arvore.fit(df[features],df[target])

arvore_predict = arvore.predict(df[features])

# acurácia
arvore_acc = metrics.accuracy_score(df[target], arvore_predict)
print("Acuracia arvore:", arvore_acc)

# precisao
arvore_precision = metrics.precision_score(df[target], arvore_predict)
print("Precisao arvore:", arvore_precision)

# recall
arvore_recall = metrics.recall_score(df[target], arvore_predict)
print("Recall arvore:", arvore_recall)

arvore_conf = metrics.confusion_matrix(df[target], arvore_predict)
arvore_conf = pd.DataFrame(arvore_conf, index=['False','True'], 
                            columns=['False','True'])
arvore_conf
# %%

# Testando com NAIVE BAYES
from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()
nb.fit(df[features],df[target])

nb_predict = nb.predict(df[features])
nb_predict

# %%

# acurácia
nb_acc = metrics.accuracy_score(df[target],nb_predict)
print('Acurácia Naive Bayes:',nb_acc)

# precisão
nb_precision = metrics.precision_score(df[target], nb_predict)
print("Precisao Naive Bayes:", nb_precision)

# recall
nb_recall = metrics.recall_score(df[target], nb_predict)
print("Recall Naive Bayes:", nb_recall)

nb_conf = metrics.confusion_matrix(df[target], nb_predict)
nb_conf = pd.DataFrame(nb_conf, index=['False','True'], 
                        columns=['False','True'])
nb_conf

# %%
# pegando só a segunda coluna (nota - probabilidade de quem foi aprovado)
# estou definindo um ponto de corte
# depende do ponto de corte para definir o melhor modelo

nb_proba = nb.predict_proba(df[features])[:,1]
nb_predict = nb_proba > 0.2

nb_acc = metrics.accuracy_score(df[target],nb_predict)
print('Acurácia Naive Bayes:',nb_acc)

# precisão
nb_precision = metrics.precision_score(df[target], nb_predict)
print("Precisao Naive Bayes:", nb_precision)

# recall
nb_recall = metrics.recall_score(df[target], nb_predict)
print("Recall Naive Bayes:", nb_recall)

# %%

df['prob_nb'] = nb_proba
df.to_excel('../data/cerveja_notas_predict.xlsx',index=False)
# %%

#ROC CURVE

'''
roc_curve me retorna 3 arrays:
1. um array do ponto de corte
2. um array do recall
3. um array de especificidade
'''
roc_curve = metrics.roc_curve(df[target],nb_proba)
roc_curve
# %%
import matplotlib.pyplot as plt

plt.plot(roc_curve[0],roc_curve[1])
plt.grid(True)
plt.plot([0,1],[0,1],'--')
plt.show()
# %%

# area embaixo da curva
roc_auc = metrics.roc_auc_score(df[target],nb_proba)
roc_auc
# %%

df.head()
# %%
nb_proba
# %%
