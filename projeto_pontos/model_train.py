# %%
import pandas as pd

from sklearn import metrics
from sklearn import model_selection #Llibrary pra splitar o dataset
from sklearn import pipeline

from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes

import matplotlib.pyplot as plt
from feature_engine import imputation

# %%
df = pd.read_csv("../data/dados_pontos.csv",
                 sep=";")

df

# %%

features = df.columns.to_list()[3:-1]
target = 'flActive'

# splitando entre treinamento e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

print('Tx. Resposta Treino:', y_train.mean())
print('Tx. Resposta Treino:', y_test.mean())

# %%

X_train.isna().sum()
max_avgRecorrencia = X_train['avgRecorrencia'].max()

# %%
# TREINANDO O MODELO

features_imput_0 = [
    'qtdeRecencia',
    'freqDias',
    'freqTransacoes',
    'qtdListaPresença',
    'qtdChatMessage',
    'qtdTrocaPontos',
    'qtdResgatarPonei',
    'qtdPresençaStreak',
    'pctListaPresença',
    'pctChatMessage',
    'pctTrocaPontos',
    'pctResgatarPonei',
    'pctPresençaStreak',
    'qtdePontosGanhos',
    'qtdePontosGastos',
    'qtdePontosSaldo', 
]

# substitui números ausentes NaN (dados faltantes) por um número arbitrário, nesse caso 0
imputation_0 = imputation.ArbitraryNumberImputer(variables=features_imput_0,
                                                 arbitrary_number=0)

# só estou definindo as variáveis que vão receber esse número arbitrário
imputation_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'], 
                                                   arbitrary_number=max_avgRecorrencia)

# cada nó da árvore terá pelo menos 50 amostras, o que ajuda a evitar overfitting
model = tree.DecisionTreeClassifier(max_depth=4,
                                    min_samples_leaf=50)

model_rf = ensemble.RandomForestClassifier(random_state=42)

# fazendo a transformação na base de treino
meu_pipeline = pipeline.Pipeline([
    ('imput_0', imputation_0),
    ('imput_max',imputation_max),
    ('model',model_rf)
    ])

params = {
    'model__n_estimators': [100,150,250,500],
    'model__min_samples_leaf': [10,20,30,50,100],
}

grid = model_selection.GridSearchCV(meu_pipeline, param_grid=params)
grid.fit(X_train, y_train)

# %%

# PREVENDO O MODELO

# treino
y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)[:,1]

# teste
y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)[:,1]

# %%

# acurácia
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_teste = metrics.accuracy_score(y_test, y_test_predict)
print('Acurácia base ACC:', acc_train)
print('Acurácia teste ACC:', acc_teste)

# roc curve
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_teste = metrics.roc_auc_score(y_test, y_test_proba)
print('ROC Curve Train:', auc_train)
print('ROC Curve Test', auc_teste)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Gerando os pontos da curva ROC (FPR, TPR)
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# Criando o gráfico
plt.figure(figsize=(8,6))
plt.plot(fpr_train, tpr_train, label=f'Treino AUC = {auc_train:.3f}', linestyle='--', color='blue')
plt.plot(fpr_test, tpr_test, label=f'Teste AUC = {auc_teste:.3f}', linestyle='-', color='red')

# Linha de referência (modelo aleatório)
plt.plot([0,1], [0,1], linestyle='dashed', color='gray')

# Configuração dos rótulos e legenda
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC')
plt.legend()
plt.grid()
plt.show()

# %%
