from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

header = st.container()
dataset = st.container
features = st.container()
modelTraining = st.container()

with header:
    st.title("Bienvenido a mi proyecto de Data Science")
    st.text("En este proyecto pondremos a prueba 5 diferentes algoritmos: Regresion Lineal, Regresion Polinomial, Clasificador Gaussiano, Clasificador de arboles de decision, Redes neuronales")
#-----------------------------S I D E   B A R   P A N E L------------------------------------------#

with st.sidebar.header('1. Upload your data file'):
    data_menu = ["CSV FILE", "XLS FILE", "XLSX FILE", "JSON FILE"]
    choice_file = st.sidebar.selectbox("Data select", data_menu)

    if choice_file == "CSV FILE":
        st.sidebar.subheader("CSV files")
        data_file = st.sidebar.file_uploader("Upload only: CSV", type=['csv'])

    elif choice_file == "XLS FILE":
        st.sidebar.subheader("XLS files")
        data_file = st.sidebar.file_uploader("Upload only: XLS", type=['xls'])

    elif choice_file == "XLSX FILE":
        st.sidebar.subheader("XLSX files")
        data_file = st.sidebar.file_uploader("Upload only: XLSX", type=['xlsx'])

    elif choice_file == "JSON FILE":
        st.sidebar.subheader("JSON files")
        data_file = st.sidebar.file_uploader("Upload only: JSON", type=['json'])


with st.sidebar.header('2. Set Parameters'):
    algorithm_name = st.sidebar.selectbox("Select Algorithm", ("Regresion Lineal", "Regresion Polinomial", "Clasificador Gaussiano", "Clasificador de arboles de desicion", "Redes neuronales"))
    
    # Definir los parametros para cada uno de los algoritmos
    if algorithm_name == "Regresion Lineal":
        parameter_data = st.sidebar.text_input('Que caracteristica debe usarse como entrada explicativa? : X')
        parameter_target = st.sidebar.text_input('Cual es la variable de respuesta objetiva? : Y')
    elif algorithm_name == "Regresion Polinomial":
        parameter_data = st.sidebar.text_input('Que caracteristica debe usarse como entrada explicativa? : X')
        parameter_target = st.sidebar.text_input('Cual es la variable de respuesta objetiva? : Y')
        parameter_degree = st.sidebar.number_input("Pick a degree", 1, 3, 2)
    elif algorithm_name == "Clasificador Gaussiano":
        max_depth =st.sidebar.slider("max_depth", 2, 15)

#-----------------------------A L G O R I T M O S------------------------------------------#
def build_linear_regression(df):
    if parameter_data == "" or parameter_target == "":
        st.info('No ha ingresado parametros')
        return None

    try:
        x = np.asarray(df[parameter_data]).reshape(-1,1)
        y = df[parameter_target]

        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)

        st.write("Interseccion (b)", regr.intercept_)
        st.write("Pendiente (m)", regr.coef_)

        st.markdown('**Prediccion**')
        input_feature = st.number_input('Establezca parametro de prediccion', None, None, 1000)

        st.write("Prediccion", regr.predict([[input_feature]]))

        fig = plt.figure(figsize = (10, 4))
        plt.scatter(x, y, color='black')
        plt.plot(x, y_pred, color='blue', linewidth=3)

        plt.ylabel("variable dependiente Y")
        plt.xlabel("Variable independiente X")

        st.subheader("Vistazo a la grafica")
        st.pyplot(fig)
    except:
        st.info('Parametros no reconocidos por la data')

def build_polynomial_regression(df):
    if parameter_data == "" or parameter_target == "":
        st.info('No ha ingresado parametros')
        return None

    try:
        # data
        x = np.asarray(df[parameter_data])[:,np.newaxis]
        y = np.asarray(df[parameter_target])[:,np.newaxis]

        # regression transform
        polynomial_features = PolynomialFeatures(degree= parameter_degree)
        x_transform = polynomial_features.fit_transform(x)

        # fit the model
        regr = linear_model.LinearRegression()
        regr.fit(x_transform, y)
        y_pred = regr.predict(x_transform)

        st.write("Interseccion (b)", regr.intercept_)
        st.write("Pendiente (m)", regr.coef_)

        st.markdown('**Prediccion**')
        input_feature = st.number_input('Establezca parametro de prediccion', None, None, 1000)

        x_new_min = 0.0
        x_new_max = input_feature

        x_new = np.linspace(x_new_min, x_new_max, input_feature)
        x_new = x_new[:,np.newaxis]

        x_new_transform = polynomial_features.fit_transform(x_new)
        y_pred = regr.predict(x_new_transform)

        st.write("Prediccion", y_pred)

        fig = plt.figure(figsize = (10, 4))
        plt.scatter(x, y, color='black')
        plt.plot(x_new, y_pred, color='coral', linewidth=3)
        plt.grid()
        # Estableciendo limites para los ejes de la grafica
        lim_min_x = st.number_input('Establezca limite maximo para el eje X', None, None, x_new_min)
        lim_max_x = st.number_input('Establezca limite maximo para el eje X', None, None, x_new_max)
        lim_min_y = st.number_input('Establezca limite minimo para el eje Y', None, None, 0)
        lim_max_y = st.number_input('Establezca limite maximo para el eje Y', None, None, 400000)
        plt.xlim(lim_min_x, lim_max_x)
        plt.ylim(lim_min_y, lim_max_y)
        title = 'Degree = {};'.format(parameter_degree)
        plt.title('Prediction for Polynomial Regression\n ' + title, fontsize=10)

        plt.ylabel("variable dependiente Y")
        plt.xlabel("Variable independiente X")

        st.subheader("Vistazo a la grafica")
        st.pyplot(fig)
    except:
        st.info('Parametros no reconocidos por la data')

#-----------------------------M A I N   P A N E L------------------------------------------#
if data_file is not None:

    # Eleccion del tipo de archivo
    if data_file.type == "text/csv":
        df = pd.read_csv(data_file)
        st.markdown('**Vistazo a los datos**')
        st.write(df) 

    elif data_file.type == "application/vnd.ms-excel":
        st.write("Estoy en un excel")
    elif data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        st.write("Estoy en un xlsx")
    elif data_file.type == "application/json":
        st.write("Estoy en un json")

    # Eleccion del algoritmo
    if algorithm_name == "Regresion Lineal":
        build_linear_regression(df)
    elif algorithm_name == "Regresion Polinomial":
        build_polynomial_regression(df)
    elif algorithm_name == "Clasificador Gaussiano":
        st.write("Esperando a gauss")

else:
    st.info('Esperando a que se cargue un archivo')