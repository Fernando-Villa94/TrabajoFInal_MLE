#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import lightgbm as lgb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['Target'],axis=1)
    y_test = df[['Target']]
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)


# Validación desde el inicio
def main():
    df = eval_model('Pallet_val.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()


# Conclusiones: si bien el modelo tiene un resultado no tan elevado, es un buen acercamiento para obtener una 
# noción rápida de las cargas que pueden ser palletizadas (normalmente una demanda alta diaria). Este modelo 
# podría complementar para aliviar la carga de procesamiento de un modelo de optimización. 