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
0radius = st.sidebar.slider("0radius", 7.0, 29.0, 15.0)
0texture = st.sidebar.slider("0texture", 10.0, 40.0, 20.0)
0smoothness = st.sidebar.slider("0smoothness", 0.06, 1.0, 1.0)
0concave_points = st.sidebar.slider("0concave_points", 1.0, 1.0, 1.0)
0symmetry = st.sidebar.slider("0symmetry", 1.0, 1.0, 1.0)
0fractal_dimension = st.sidebar.slider("0fractal_dimension", 1.0, 1.0, 1.0)

1radius = st.sidebar.slider("Radius1", 1.0, 3.0, 1.0)
1texture = st.sidebar.slider("Texture1", 1.0, 5.0, 2.0)
1smoothness = st.sidebar.slider("Smoothness1", 1.0, 1.0, 1.0)
1concave_points = st.sidebar.slider("Compactness1", 1.0, 1.0, 1.0)
1symmetry = st.sidebar.slider("Concavity1", 1.0, 1.0, 1.0)
1fractal_dimension = st.sidebar.slider("Concave1", 1.0, 1.0, 1.0)

2radius = st.sidebar.slider("Smoothness1", 8.0, 37.0, 17.0)
2texture = st.sidebar.slider("Compactness1", 12.0, 50.0, 26.0)
2smoothness = st.sidebar.slider("Concavity1", 1.0, 1.0, 1.0)
2concave_points = st.sidebar.slider("Concave1", 1.0, 1.0, 1.0)
2symmetry = st.sidebar.slider("Compactness1", 1.0, 1.0, 1.0)
2fractal_dimension = st.sidebar.slider("Concave1", 1.0, 1.0, 1.0)

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


         


    

 
