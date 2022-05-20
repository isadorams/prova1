import streamlit as st
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
0radius = st.sidebar.slider("Radius",7.0,29.0,4.3, 0.1)
0texture = st.sidebar.slider("Texture", 11.0, 50.0, 26.0, 0.1)
0smoothness = st.sidebar.slider("Smoothness", 0.06, 0.22, 0.14, 0.1)
0concave_points = st.sidebar.slider("Concave", 0.0, 0.30, 0.11, 0.1)
0symmetry = st.sidebar.slider("Symmetry", 0.14, 0.66, 0.30, 0.1)
0fractal_dimension = st.sidebar.slider("Fractal", 0.054, 0.20, 0.08, 0.1)
1radius = st.sidebar.slider("Radius1", 7.8, 36.0, 6.3, 0.1)
1texture = st.sidebar.slider("Texture1", 11.0, 50.0, 26.0, 0.1)
1smoothness = st.sidebar.slider("Smoothness1", 0.06, 0.22, 0.14, 0.1)
1concave_points = st.sidebar.slider("Compactness1", 0.01, 1.6, 0.25)
1symmetry = st.sidebar.slider("Concavity1", 0.0, 1.30, 0.27)
1fractal_dimension = st.sidebar.slider("Concave1", 0.0, 0.30, 0.11, 0.1)
2radius = st.sidebar.slider("Smoothness1", 0.06, 0.22, 0.14, 0.1)
2texture = st.sidebar.slider("Compactness1", 0.01, 1.6, 0.25)
2smoothness = st.sidebar.slider("Concavity1", 0.0, 1.30, 0.27)
2concave_points = st.sidebar.slider("Concave1", 0.0, 0.30, 0.11, 0.1)
2symmetry = st.sidebar.slider("Compactness1", 0.01, 1.6, 0.25)
2smoothness = st.sidebar.slider("Concavity1", 0.0, 1.30, 0.27)
2concave_points = st.sidebar.slider("Concave1", 0.0, 0.30, 0.11, 0.1)
2fractal_dimension = st.sidebar.slider("Concave1", 0.0, 0.30, 0.11, 0.1)

with open("objetos.pkl", "rb") as arquivo:
  ss, classifier = pickle.load(arquivo)
  
  #df = pd.read_csv('wdbc.csv', names = colunas)
 

  estrutura = {'0radius': [0radius], '0texture': [0texture], '0smoothness': [0smoothness], '0concave_points':[0concave_points], '0symmetry':[0symmetry], 
               '0fractal_dimension': [0fractal_dimension], '1radius': [1radius], '1texture': [1texture], '1smoothness': [1smoothness],
               '1concave_points': [1concave_points], '1symmetry': [1symmetry], '1fractal_dimension': [1fractal_dimension], '2radius': [2radius], 
               '2texture':[2texture], '2smoothness': [2smoothness], '2concave_points':[2concave_points],'2symmetry':[2symmetry], '2fractal_dimension': [2fractal_dimension], }
  df = pd.DataFrame(estrutura, index=[0])
 
  st.write("### Parâmetros de Entrada")
  st.write(df)
    
  df = ss.transform(df)
  st.write(df)
  
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


         


    

 
