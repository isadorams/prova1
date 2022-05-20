import streamlit as st
from sklearn.neighbors import KNeighborsClassifier  
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score
from PIL import Image
#from gsheetsdb import connect

  
st.title('Relatórios Breast Cancer Wisconsin')
st.header("Informações do conjunto de dados:")
st.write(
    """As características são computadas a partir de uma imagem digitalizada de um aspirado
    por agulha fina (PAAF) de uma massa mamária.Eles descrevem características 
    dos núcleos celulares presentes na imagem..""" )

IMAGE_URL = "https://miro.medium.com/max/1400/1*yjsLGG-U9km84AvWLLmK8A.png"
st.image(IMAGE_URL, caption="Imagem da célula cancerosa")

st.header("Conjunto de dados:")
IMAGE_URL = "https://miro.medium.com/max/1400/1*51Hm0b9RlgnPVQLariliRw.png"
st.image(IMAGE_URL, caption="amostra dos dados")

st.sidebar.write("### Parâmetros") #barra lateral com interatividade
radius0 = st.sidebar.slider("0radius", 7.0, 29.0, 15.0)
texture0 = st.sidebar.slider("0texture", 10.0, 40.0, 20.0)
smoothness0 = st.sidebar.slider("0smoothness", 0.06, 1.0, 1.0)
concave_points0 = st.sidebar.slider("0concave_points", 0.1, 1.0, 1.0)
symmetry0 = st.sidebar.slider("0symmetry", 0.12, 1.0, 1.0)
fractal_dimension0 = st.sidebar.slider("0fractal_dimension", 0.05, 1.0, 1.0)

radius1 = st.sidebar.slider("Radius1", 1.0, 3.0, 1.0)
texture1 = st.sidebar.slider("Texture1", 1.0, 5.0, 2.0)
smoothness1 = st.sidebar.slider("smoothness1", 0.0, 1.0, 1.0)
concave_points1 = st.sidebar.slider("concave_points1", 0.0, 1.0, 1.0)
symmetry1 = st.sidebar.slider("symmetry1", 0.0, 1.0, 1.0)
fractal_dimension1 = st.sidebar.slider("fractal_dimension1", 0.0, 1.0, 1.0)

radius2 = st.sidebar.slider("radius2", 8.0, 37.0, 17.0)
texture2 = st.sidebar.slider("texture2", 12.0, 50.0, 26.0)
smoothness2 = st.sidebar.slider("smoothness2", 0.0, 1.0, 1.0)
concave_points2 = st.sidebar.slider("concave_points2", 0.0, 1.0, 1.0)
symmetry2 = st.sidebar.slider("symmetry0", 0.2, 1.0, 1.0)
fractal_dimension2 = st.sidebar.slider("fractal_dimension2", 0.0, 1.0, 1.0)

with open("objetos.pkl", "rb") as arquivo:
  ss, classifier = pickle.load(arquivo)
  
  #df = pd.read_csv('wdbc.csv', names = colunas)
 

  estrutura = {'radius0': [radius0], 'texture0': [texture0], 'smoothness0': [smoothness0], 'concave_points0':[concave_points0], 'symmetry0':[symmetry0], 
               'fractal_dimension0': [fractal_dimension0], 
               'radius1': [radius1], 'texture1': [texture1], 'smoothness1': [smoothness1], 'concave_points1': [concave_points1],
               'symmetry1': [symmetry1], 'fractal_dimension1': [fractal_dimension1], 
               'radius2': [radius2],'texture2':[texture2], 'smoothness2': [smoothness2], 'concave_points2':[concave_points2], 'symmetry2':[symmetry2], 'fractal_dimension2': [fractal_dimension2]}
  df = pd.DataFrame(estrutura, index=[0])
 
  st.write("### Parâmetros de Entrada")
  st.write(df)
    
  df = ss.transform(df)
  st.write(df)
 """
  predicao = classifier.predict([df.values])
  st.write("**A classe desse cancer é:**")
  st.write(predicao)
  
  predicao = classifier.predict_proba(df)
  predicao = pd.DataFrame(predicao)
  predicao.rename({
     0: "M",
     1: "B",
     2: "inválido"
  }, axis=1, inplace=True)
  
  st.write("Probabilidades")
  st.write(predicao)
  """
  dataframe = pd.DataFrame(np.random.randn(10, 20), #dataframe como uma tabela interativa
  columns = ('col %d' % i
    for i in range(20)))
  st.write(dataframe)
  st.header('Visualização do gráfico de área.')
  st.area_chart(dataframe)
  #st.header('Visualização do histograma.')
  #st.bar_chart(dataframe)
    
   
  st.write("### Informações do atributo:")
  st.write( """ a. perímetro(soma dos tamanhos dos lados da figura)""" )
  st.write( """ b. área (medida total que uma figura ocupa no plano)""" )
  st.write( """ c. compacidade (perímetro^2 / área - 1,0)""" )
  st.write( """ d. concavidade (severidade das porções côncavas do contorno).""" )


         


    

 
