import streamlit as st
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
#import plotly.express as px
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from PIL import Image

  
st.title('Relatórios Breast Cancer Wisconsin')

st.header("Informações do conjunto de dados:")
st.write(
    """As características são computadas a partir de uma imagem digitalizada de um aspirado
    por agulha fina (PAAF) de uma massa mamária.Eles descrevem características 
    dos núcleos celulares presentes na imagem..""" )

image = Image.open('C:\Users\isadora skibinski\Downloads')
st.image(image, caption = 'This is a picture', use_column_width = True)

st.header("Conjunto de dados:")
dataframe = pd.DataFrame(np.random.randn(10, 20),
  columns = ('col %d' % i
    for i in range(20)))
st.write(dataframe)
st.header('Visualização do gráfico de área.')
st.area_chart(dataframe)
st.header('Visualização do histograma.')
st.bar_chart(dataframe)
         
st.sidebar.write("### Parâmetros")
perimeter = st.sidebar.slider("Perimeter", 40.0, 190.0, 100.0)
area= st.sidebar.slider("Area", 184.0, 4254.0, 881.0)
compactness = st.sidebar.slider("Compactness", 0.01, 1.6, 0.25)
concavity = st.sidebar.slider("Concavity", 0.0, 1.30, 0.27)

    
with open("objetos.pkl", "rb") as arquivo:
  ss, classifier = pickle.load(arquivo)

  estrutura = {'perimeter': perimeter, 'area': area, 'compactness': compactness, 'concavity': concavity}
  df = pd.DataFrame(estrutura, index=[0])
 
  st.write("### Parâmetros de Entrada")
  st.write(df)
  
  df_sample = df.head()
  df_sample
    
  #df = ss.transform(df)
  #st.write(df)
  
  #predicao = classifier.predict(df)
  #st.write(f"A classe é: **{predicao[0]}**")
  
  #predicao = classifier.predict_proba(df)
  #predicao = pd.DataFrame(predicao)
  #predicao.rename({
     #'M' : 0,
     #'B' : 1
  #}, axis=1, inplace=True)
  
  #st.write("Probabilidades")
  #st.write(predicao)
    
   
  st.header("Informações do atributo:")
  st.write( """ a. perímetro(soma dos tamanhos dos lados da figura)""" )
  st.write( """ b. área (medida total que uma figura ocupa no plano)""" )
  st.write( """ c. compacidade (perímetro^2 / área - 1,0)""" )
  st.write( """ d. concavidade (severidade das porções côncavas do contorno).""" )

  

    

 
