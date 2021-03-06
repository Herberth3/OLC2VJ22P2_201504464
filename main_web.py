from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

header = st.container()

st.markdown(
    """
    <style>
    .main {
        background-color: #FCAD22;
    }
    <\style>
    """,
    unsafe_allow_html=True
)

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
        parameter_encoder = st.sidebar.checkbox('Implementar Encoder')
        parameter_target = st.sidebar.text_input('Ingrese el nombre de la columna objetivo : Y')
    elif algorithm_name == "Clasificador de arboles de desicion":
        parameter_encoder = st.sidebar.checkbox('Implementar Encoder')
        parameter_target = st.sidebar.text_input('Ingrese el nombre de la columna objetivo : Y')
        parameter_depth_tree = st.sidebar.number_input('Profundidad de las ramas en el arbol', 0, None, 5)
        parameter_name_tree = st.sidebar.text_input('Ingrese un nombre para su Arbol de decision:', "Mi arbol de decision")
    elif algorithm_name == "Redes neuronales":
        parameter_encoder = st.sidebar.checkbox('Implementar Encoder')
        parameter_target = st.sidebar.text_input('Ingrese el nombre de la columna objetivo : Y')
        parameter_capa1 = st.sidebar.number_input('Capa1, Hidden Layer Size', 0,None, 100)
        parameter_capa2 = st.sidebar.number_input('Capa2, Hidden Layer Size', 0,None, 100)
        parameter_capa3 = st.sidebar.number_input('Capa3, Hidden Layer Size', 0,None, 100)
        parameter_max_iter = st.sidebar.number_input('Max, iteraciones', 0,None, 1000)


#-----------------------------A L G O R I T M O S------------------------------------------#
# -- Regresion Lineal -- #
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

        # Mostrando la funcion de tendencia
        st.markdown('**La funcion para regresion lineal es: y = mx + b**')
        st.write("y = " + str(regr.coef_[0]) + "x + (" + str(regr.intercept_) + ")")

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

# -- Regresion Polinomial -- #
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

        # Mostrando la funcion de tendencia
        st.markdown('**Funcion de tendencia polinomial:**')
        coeficientes = ""
        grado = 0
        for coef in regr.coef_[0]:
            coeficientes = coeficientes +"("+ str(coef) +")"+ "x^" + str(grado) + " + "
            grado = grado + 1

        st.write(coeficientes +"("+ str(regr.intercept_[0]) + ")")

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

# -- Clasificador Gaussiano -- #
def build_gaussian_model(df):
    if parameter_target == "":
        st.info('No ha ingresado parametros')
        return None

    # Parametro que se solicita para realizar la prediccion
    # Este es una secuencia de numeros separados por coma, de lo contrario no se muestra el resultado
    st.markdown('**Prediccion**')
    input_feature = st.text_input('Establezca un array de prediccion separado por comas','Ex: 45,1,8,5,6')
    input_array = input_feature.split(',')

    try:
        # Columnas implementando Encoder
        if parameter_encoder:

            list_col = df.drop(columns=parameter_target)
            # Creating labelEncoder
            le = LabelEncoder()
            for col_name in list_col:
                df[col_name] = le.fit_transform(df[col_name])

            df[parameter_target] = le.fit_transform(df[parameter_target])
            st.write("Nueva Tabla con Encoder")
            st.write(df)

        # Validacion para que los parametros de localizacion en la tabla sean correctos
        y = df[[parameter_target]]

        x = df.drop(columns=parameter_target)

        # Entrenamiento y prueba de los campos
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=None, random_state=None)

        # Modelo a utilizar y poner a prueba la prediccion que este tomo en el entrenamiento
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        st.write("Prediccion del entrenamiento:")
        st.write(y_pred)

        # Array con los datos a predecir
        desired_array = [int(numeric_string) for numeric_string in input_array]
        # Se crea un nuevo valor para la X que se enviara para una nueva prediccion
        x_new_val = np.array(desired_array)
        y_pred = model.predict([x_new_val])

        st.write("Prediccion", y_pred)
    except:
        st.info('Parametros no reconocidos por la data')

# -- Clasificador de arboles de desicion -- #
def build_decision_tree(df):
    if parameter_target == "":
        st.info('No ha ingresado parametros')
        return None

    try:
        # Validacion para que los parametros de localizacion en la tabla sean correctos

        # Columnas implementando Encoder
        
        #x = df.drop(columns=parameter_target)

        # Columnas implementando Encoder
        #if parameter_encoder:
            #lista_col_x = []
            #for col_name in x:
                #list_c = df[col_name].values.tolist()
                #lista_col_x.append(list_c)

            #lista_col_y = df[parameter_target].values.tolist()

            # Creating labelEncoder
            #le = LabelEncoder()
            #lista_encoder = []
            #for elem in lista_col_x:
                #encoder = le.fit_transform(elem)
                #lista_encoder.append(encoder)

            #label = le.fit_transform(lista_col_y)

            # Combinig attributes into single listof tuples
            #features=list(zip(*lista_encoder))
        #else:
            # Columnas sin Encoder
            #lista_col_x = []
            #for col_name in x:
                #list_c = df[col_name]
                #lista_col_x.append(list_c)

            #label = df[parameter_target]

            # Combinig attributes into single listof tuples
            #features=list(zip(*lista_col_x))

        # fit the model
        #clf = DecisionTreeClassifier().fit(features, label)
        
        if parameter_encoder:

            list_col = df.drop(columns=parameter_target)
            # Creating labelEncoder
            le = LabelEncoder()
            for col_name in list_col:
                df[col_name] = le.fit_transform(df[col_name])
            #df[parameter_target] = le.fit_transform(df[parameter_target])
            st.write("Nueva Tabla con Encoder")
            st.write(df)

        # Validacion para que los parametros de localizacion en la tabla sean correctos
        y = df[parameter_target]
        x = df.drop(columns=parameter_target)
        # Entrenamiento y prueba de los campos
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=None, random_state=None)

        print(x.columns.values)
        print(y.values)

        clf = DecisionTreeClassifier(max_depth=parameter_depth_tree)
        clf_fit = clf.fit(x_train, y_train)
        st.markdown("**"+ parameter_name_tree +"**")
        fig = plt.figure(figsize=(20,10))
        #plot_tree(clf, filled=True)
        plot_tree(clf_fit, feature_names=list(x.columns.values), class_names=list(y.values), filled=True)
        st.pyplot(fig)
        
    except:
        st.info('Parametros no reconocidos por la data')

# -- Clasificador Gaussiano -- #
def build_red_neuronal(df):
    if parameter_target == "":
        st.info('No ha ingresado parametros')
        return None

    # Parametro que se solicita para realizar la prediccion
    # Este es una secuencia de numeros separados por coma, de lo contrario no se muestra el resultado
    st.markdown('**Prediccion**')
    input_feature = st.text_input('Establezca un array de prediccion separado por comas','Ex: 45,1,8,5,6')
    input_array = input_feature.split(',')

    try:
        # Columnas implementando Encoder
        if parameter_encoder:

            list_col = df.drop(columns=parameter_target)
            # Creating labelEncoder
            le = LabelEncoder()
            for col_name in list_col:
                df[col_name] = le.fit_transform(df[col_name])

            #df[parameter_target] = le.fit_transform(df[parameter_target])
            st.write("Nueva Tabla con Encoder")
            st.write(df)

        # Validacion para que los parametros de localizacion en la tabla sean correctos
        y = df[parameter_target]

        x = df.drop(columns=parameter_target)

        # Entrenamiento y prueba de los campos
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=None, random_state=None)

        mlp = MLPClassifier(hidden_layer_sizes=(parameter_capa1, parameter_capa2, parameter_capa3,), max_iter=parameter_max_iter)

        scores = cross_val_score(mlp, x, y)
        st.write("Score:")
        st.write(scores.mean())
        st.write("Std. Desviacion")
        st.write(scores.std())

        mlp.fit(x_train, y_train)

        # Array con los datos a predecir
        desired_array = [int(numeric_string) for numeric_string in input_array]
        # Se crea un nuevo valor para la X que se enviara para una nueva prediccion
        x_new_val = np.array(desired_array)
        y_pred = mlp.predict([x_new_val])

        st.write("Prediccion", y_pred)

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
        build_gaussian_model(df)
    elif algorithm_name == "Clasificador de arboles de desicion":
        build_decision_tree(df)
    elif algorithm_name == "Redes neuronales":
        build_red_neuronal(df)

else:
    st.info('Esperando a que se cargue un archivo')