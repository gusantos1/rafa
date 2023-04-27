import pandas as pd
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def user_input_features(xCol):
    data = {}
    for col in xCol.columns.to_list():
        xCol[col] = xCol[col].astype(float, errors = 'raise')
        data[col] = st.sidebar.slider(col,xCol[col].min(),xCol[col].max(),xCol[col].mean())  
    #data['bedrooms'] = st.sidebar.slider('bedrooms',xCol['bedrooms'].min(),xCol['bedrooms'].max(),xCol['bedrooms'].mean())    
    return pd.DataFrame(data, index=[0])

def model_ia(x,y):
    model = LinearRegression()
    model.fit(x, y)
    return model


def load_data():
    base_casas = pd.read_csv('house_prices.csv')
    x = base_casas[['bedrooms','bathrooms','floors','condition','sqft_living']]
    y = base_casas.iloc[:, 2]
    return  (x,y)
     


def main():
    X_casas, y_casas = load_data()

    st.write("""
    # Predição de preços de casa
    # """)
    st.write('---')

    
    model = model_ia(X_casas,y_casas)
    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Escolha de paramentros para Predição')

    df = user_input_features(X_casas)


    st.header('Parametros especificados')
    st.write(df)
    st.write('---')

    prediction = model.predict(df)

    st.header('Preço Previsto')
    st.write(prediction)
    st.write('---')

if __name__=='__main__':
    main()

