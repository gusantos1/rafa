# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def user_input_features(xCol):
    data = {}
    for col in xCol.columns.to_list(): #[]
        xCol[col] = xCol[col].astype(float, errors = 'raise')
        data[col] = st.sidebar.slider(col,float(xCol[col].min()),float(xCol[col].max()),float(xCol[col].mean()))  
    return pd.DataFrame(data, index=[0])

def model_ia(x,y):
    model = LinearRegression()
    model.fit(x, y)
    return model

def create_dataframe():
    data = {'Department': [],'MaritalStatus': [],'OverTime': [],'JobRole': [],'Gender': [],'EducationField': []}
    df = pd.DataFrame(data=data)
    return df

def load_data():
    binario = open('classificador7.pkl','rb')
    modelo = pickle.load(binario)
    dados = pd.read_csv('x_test.csv').drop(columns=['Unnamed: 0'])
    return dados, modelo
     

def main():
    dados, modelo = load_data()

    st.write("""
    # Predição de preços de casa
    # """)
    st.write('---')
    
    st.sidebar.header('Escolha de paramentros para Predição')

    menu = dados.filter(['Education', 'Age', 'DistanceFromHome'])
    df = user_input_features(menu)

    dados['Education'] = df['Education']
    dados['Age'] = df['Age']
    dados['DistanceFromHome'] = df['DistanceFromHome']

    st.header('Parametros especificados')
    st.write(df)
    st.write('---')
    
    st.header('Preço Previsto')
    prediction = modelo.predict(dados)
    resultado = pd.DataFrame({'Previsão':prediction[0]})
    
    st.write(resultado)
    st.write('---')

if __name__=='__main__':
    main()