#!/usr/local/bin/python
# coding: utf-8
import os, sys

###########################################
# desativa alertas do matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

###########################################

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

from time import time

#%matplotlib inline


# Define a coluna de target
def pre_processing_target_column(dataset_negociacao, dias_previsao): 
    
    dataset_negociacao.Predict = np.NaN
    
    for index, row in dataset_negociacao[:-dias_previsao].iterrows():
        
        if row["Adj Close"] < dataset_negociacao.iloc[index + (dias_previsao)]["Adj Close"]:
            dataset_negociacao.loc[index, "Predict"] = 1
        else :
            dataset_negociacao.loc[index, "Predict"] = 0


# Graficos para auxiliar a decisao de utilização de transformacao logaritmica
def pre_processing_verify_log_transformed(dtNegociacoes):
    dataset_verificacao = dtNegociacoes.copy()

    fig = pl.figure(figsize = (18,5));

    # display(dataset_verificacao.columns.values)
    for i, feature in enumerate(['Open', 'High', 'Low', 'Close', 'Adj Close', "Volume"]):
        ax = fig.add_subplot(1, 6, i+1)
        ax.hist(dataset_verificacao[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' "%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.set_ylim((0, 600))
        ax.set_yticks([0, 50, 100, 200, 300, 400, 500, 600])
        ax.set_yticklabels([0, 50, 100, 200, 300, 400, 500, ">600"])

    fig.tight_layout()
    fig.show()

    
# Transformacao logaritmica
def pre_processing_log_transformed(dtNegociacoes, features):
    skewed = features
    features_log_transformed = pd.DataFrame(data = dtNegociacoes)
    features_log_transformed[skewed] = dtNegociacoes[skewed].apply(lambda x: np.log(x + 1))
    
    return features_log_transformed
    
    
# Normalizacao
from sklearn.preprocessing import MinMaxScaler

def pre_processing_minmax_transform(features_log_transformed, features):

    scaler = MinMaxScaler()
    numerical = features

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    
    return features_log_minmax_transform
    
    
    
# Marcando target e index features e separando dados de treinamento e teste
def pre_processing_sort_training_test(features_log_minmax_transform, target_feature, index_feature):

    features_log_minmax_transform = features_log_minmax_transform.dropna()
    features_log_minmax_transform = features_log_minmax_transform.set_index(index_feature)

    target = features_log_minmax_transform[target_feature]
    features_final = features_log_minmax_transform.drop([target_feature], axis=1)

    from sklearn.model_selection import train_test_split

    # Dividir os 'atributos' e 'income' entre conjuntos de treinamento e de testes.
    X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                        target, 
                                                        test_size = 0.25, 
                                                        random_state = 0)
    
    return X_train, X_test, y_train, y_test

    
    
def pre_processing_dataset(dtNegociacoes, dias_previsao, features_trans_logaritmica
                           , features_normalizacao, target_feature, index_feature):
    
    dataset_verificacao = dtNegociacoes.copy()
    
    # aplicando transformação logaritmica
    pre_processing_target_column(dataset_verificacao, dias_previsao)
    
    features_log_transformed = pre_processing_log_transformed(dataset_verificacao, features_trans_logaritmica)
    
    features_log_minmax_transform = pre_processing_minmax_transform(
                                                features_log_transformed
                                                , features_normalizacao)
    
    X_train, X_test, y_train, y_test = pre_processing_sort_training_test(
                                                features_log_minmax_transform
                                                , target_feature
                                                , index_feature)
    
    
    return X_train, X_test, y_train, y_test








from sklearn.metrics import fbeta_score 
from sklearn.metrics import precision_score

def train_predict(learner, X_train, y_train, X_test, y_test): 
    
    results = {}
    
    # train
    start = time() # Get start time
    learner.fit(X_train, y_train)
    end = time() # Get end time
    
    # Calc train time
    #results['train_time'] = end-start
        
    # predict
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train)
    end = time() # Get end time
   
    
    # Calc pred time
    #results['pred_time'] = end-start
            
    # Calc acuracia
    results['precision_train'] = precision_score(y_train, predictions_train)
        
    # Calc acuracia
    results['precision_test'] = precision_score(y_test, predictions_test)
    
    # Calc f-score
    #results['f_train'] = fbeta_score(y_train[:], predictions_train, average=None, beta=0.5)
        
    # calc f-score
    #results['f_test'] = fbeta_score(y_test, predictions_test, average=None, beta=0.5)
       
    # Success
    print("{} trained.".format(learner.__class__.__name__))
        
    # Return the results
    return results



# define a coluna de target
def compare_models(X_train, y_train, X_test, y_test):
    
    X_train_ = X_train.copy()
    y_train_ = y_train.copy()
    X_test_ = X_test.copy()
    y_test_ = y_test.copy()
    
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import AdaBoostClassifier

    clf_A = KNeighborsClassifier()
    clf_B = LinearSVC(random_state=0)
    clf_C = AdaBoostClassifier(random_state=0)

    # Colete os resultados dos algoritmos de aprendizado
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        results[clf_name]= train_predict(clf, X_train_, y_train_, X_test_, y_test_)
    
    return results




from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer

# avaliação do tuning do modelo
def tuning_valuation(model, params, X_train, y_train, X_test, y_test, print_result):
    
    scorer =  make_scorer(fbeta_score, beta=0.5)
    grid_obj = GridSearchCV(model, params)
    grid_fit = grid_obj.fit(X_train, y_train)

    # Recuperar o estimador
    best_clf = grid_fit.best_estimator_

    # Realizar predições utilizando o modelo não otimizado e modelar
    predictions = (model.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    if print_result:
        # Reportar os scores de antes e de depois
        print("Otimização\n-------------------------------------------------")
        print("ANTES")
        print("Precisao dos dados de teste: {:.4f}".format(precision_score(y_test, predictions)))
        #print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
        print("\nDEPOIS")
        print("Precisao dos dados de teste: {:.4f}".format(precision_score(y_test, best_predictions)))
        #print("F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

    return best_clf, [precision_score(y_test, predictions), precision_score(y_test, best_predictions)]

def tuning_visualization(empresa, default_tuned_values, benchmark): 

    labels = ['Default', 'Otimizado']
    values = default_tuned_values

    fig, ax = plt.subplots(1, 1)
    ax.yaxis.grid(True)
    ax.bar(range(len(values)), values, width=0.3,align='center',color='skyblue')
    plt.axhline(y=benchmark, color='r', linestyle='-', linewidth=1)
    plt.xticks(range(len(values)), labels, size='small')
    plt.title('Comparativo otimizacao\r\n {}'.format(empresa), fontsize=20)

    for x in range(len(values)):
        plt.text(x = x -0.1 , y = values[x] - 0.04 , s = "{:.2f}".format(values[x]), size = 10)

    plt.show()


    
import matplotlib.pyplot as plt

def compare_models_visualization(compare_models_results): 

    learners = []
    precision_train = []
    precision_test = []

    for k, learner in enumerate(compare_models_results.keys()):
         learners.append(learner)
         precision_train.append(compare_models_results[learner]['precision_train'])
         precision_test.append(compare_models_results[learner]['precision_test'])


    fig, ax = plt.subplots(1, 1)
    ax.yaxis.grid(True)
    ax.bar(range(len(precision_train)), precision_train, width=0.3,align='center',color='skyblue')
    plt.axhline(y=max(precision_train), color='r', linestyle='-', linewidth=1)
    plt.xticks(range(len(precision_train)), learners, size='small')
    plt.title('Comparacao de Modelos (Treino)', fontsize=20)

    for x in range(len(precision_train)):
        plt.text(x = x -0.1 , y = precision_train[x] - 0.04 , s = "{:.2f}".format(precision_train[x]), size = 10)

    plt.show()


    fig, ax = plt.subplots(1, 1)
    ax.yaxis.grid(True)
    ax.bar(range(len(precision_test)), precision_test, width=0.3,align='center',color='skyblue')
    plt.axhline(y=max(precision_test), color='r', linestyle='-', linewidth=1)
    plt.xticks(range(len(precision_test)), learners, size='small')
    plt.title('Comparacao de Modelos (Teste)', fontsize=20)

    for x in range(len(precision_test)):
        plt.text(x = x -0.1 , y = precision_test[x] - 0.04 , s = "{:.2f}".format(precision_test[x]), size = 10)

    plt.show()

    
    
    
    
# define a coluna de Naive
def generate_naive_predictor(dtNegociacao): 
    
    dtNegociacao.Naive = np.NaN
    
    for index, row in dtNegociacao[1:].iterrows():
        
        if row["Adj Close"] > dtNegociacao.iloc[index - 1 ]["Adj Close"]:
            dtNegociacao.loc[index, "Naive"] = 1
        else :
            dtNegociacao.loc[index, "Naive"] = 0
            
            
            
# Compara Predict e Naive, 
def generate_naive_is_correct(dtNegociacao): 
    
    dtNegociacao.Naive_is_correct = np.NaN
    
    for index, row in dtNegociacao[:].iterrows():
        
        if row["Naive"] == row["Predict"] :
            dtNegociacao.loc[index, "Naive_is_correct"] = 1
        else :
            dtNegociacao.loc[index, "Naive_is_correct"] = 0
            
            
            
# avalia naive predictor
def naive_predictor_valuation(dtNegociacao): 
    dtNegociacao = dtNegociacao.dropna()

    accuracy = np.sum(dtNegociacao["Naive_is_correct"]) / len(dtNegociacao["Naive_is_correct"])
    
    precision = np.sum(dtNegociacao["Naive_is_correct"]) / (np.sum(dtNegociacao["Naive_is_correct"]) +
                                                            (dtNegociacao["Naive_is_correct"].count() - 
                                                             np.sum(dtNegociacao["Naive_is_correct"])))
    
    recall = np.sum(dtNegociacao["Naive_is_correct"]) /(np.sum(dtNegociacao["Naive_is_correct"]) + 0)
    beta_precision = 0.5

    # TODO: Calcular o F-score utilizando a fórmula acima para o beta = 0.5 e os valores corretos de precision e recall.
    fscore = (1 + (beta_precision ** 2)) * ((precision * recall)/ (((beta_precision ** 2) * precision) + recall ))

    # Exibir os resultados 
    display("Naive Predictor: [Precisao: {:.4f}".format(precision))
    
    return precision
    
    
# preenche os dados diarios de Selic
def load_selic(Indices, SELIC_inicio): 
    
    taxa_selic = SELIC_inicio
    
    for index, row in Indices[:].iterrows():
               
        if math.isnan(row["Selic"]) :
            Indices.loc[index, "Selic"] = taxa_selic
        else :
            taxa_selic = Indices.loc[index, "Selic"]    