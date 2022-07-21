#!/usr/bin/env python
# coding: utf-8

# In[9]:

# Se desarrollara un modelo predictivo LGBM para la clasificación

import pandas as pd
import lightgbm as lgb
import pickle
import os
from imblearn.over_sampling import RandomOverSampler

# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df.drop(['Target'],axis=1)
    y_train = df['Target']
    print(filename, ' cargado correctamente')
    ros = RandomOverSampler(random_state=42)
    x_ros, y_ros = ros.fit_resample(X_train, y_train)
    # Entrenamos el modelo con toda la muestra
    lgbm_mod = lgb.LGBMClassifier(n_estimators=20,max_depth=2,min_child_samples=12,reg_alpha=1)
    lgbm_mod.fit(x_ros, y_ros)
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(lgbm_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('Pallet_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()


# In[ ]:




