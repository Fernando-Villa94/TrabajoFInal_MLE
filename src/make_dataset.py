#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import os


# Leemos los archivos excel
def read_file_excel(filename):
    df = pd.read_excel(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Realizamos la transformación de datos
def data_preparation(df):
    df['Densidad']=df['Peso']/df['Volumen']
    df['PesoCargable']=df['Volumen']*1000000/166
    df['PesoRatio']=df['PesoCargable']-df['Peso']
    
    df['ProductoVxP']=np.log10(df['Peso']*df['Volumen'])

    # Se realiza los cálculos para valores acumulados de peso, volumen, volumen con operaciones(raíz cuadrada)
    # Se toma en cuenta lotes de 43 registros para hacer la acumulación (dato del negocio)   

    df['VolumenAcu']=df['Volumen'].iloc[0]
    df['PesoAcu']=df['Peso'].iloc[0]
    df['VolumenAcuRaiz']=np.power(df['Volumen'].iloc[0]*100,1/2)
    v=df['Volumen'].iloc[0]
    p=df['Peso'].iloc[0]
    vraiz=np.power(df['Volumen'].iloc[0]*100,1/2)
    k=1
    for i in range (1,len(df)):
        v=v+df['Volumen'].iloc[i]
        p=p+df['Peso'].iloc[i]
        vraiz=np.power(vraiz*df['Volumen'].iloc[i]*100,1/2)
        
        if i>=42*k+1:
            v=df['Volumen'].iloc[i]
            p=df['Peso'].iloc[i]
            vraiz=np.power(df['Volumen'].iloc[i]*100,1/2)
            k=k+1
            
        df['VolumenAcu'].iloc[i]=v
        df['PesoAcu'].iloc[i]=p
        df['VolumenAcuRaiz'].iloc[i]=vraiz
       
    df['Densidadacu']=np.log10(df['VolumenAcu']/df['PesoAcu'])
    
    # la sobredimension es un factor para considerar en aquellas cargas que por sus dimensiones no se pueden palletizar
    df['sobredimension']=[1 if (df['Volumen'].iloc[i])>10|(df['Peso'].iloc[i]>4500) else 0 for i in range(0, len(df))]


    print('Transformación de datos completa')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename),index = False)
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación

def main():
    # Matriz de Entrenamiento
    df1 = read_file_excel('DefaultPallet.xlsx')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['Peso','Volumen','Densidad','PesoRatio','VolumenAcu','PesoAcu','VolumenAcuRaiz','ProductoVxP','Densidadacu','sobredimension','Target'],'Pallet_train.csv')
    # Matriz de Validación
    df2 = read_file_excel('DefaultPallet_new.xlsx')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2,['Peso','Volumen','Densidad','PesoRatio','VolumenAcu','PesoAcu','VolumenAcuRaiz','ProductoVxP','Densidadacu','sobredimension','Target'],'Pallet_val.csv')
    # Matriz de Scoring
    df3 = read_file_excel('DefaultPallet_score.xlsx')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3,['Peso','Volumen','Densidad','PesoRatio','VolumenAcu','PesoAcu','VolumenAcuRaiz','ProductoVxP','Densidadacu','sobredimension'],'Pallet_score.csv')
    
if __name__ == "__main__":
    main()







