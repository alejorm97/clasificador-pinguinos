import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Clasificador de Pingüinos",
    page_icon="🐧",
    layout="wide",
)

st.title("Clasificador de Especies de Pingüinos 🐧")
st.write(
    """
    Esta aplicación utiliza un modelo de **Random Forest** para predecir la especie a la que
    pertenece un pingüino a partir de sus características físicas.
    """
)

# Control de acceso por contraseña
clave_acceso = st.text_input("Ingrese la contraseña para continuar:", type="password")
if clave_acceso != "streamlit_212":
    st.info("Por favor, ingrese la contraseña correcta (streamlit_212) para desplegar los controles.")
    st.stop()


@st.cache_resource
def cargar_artefactos():
    with open("random_forest_penguin.pickle", "rb") as rf_archivo:
        pipeline = pickle.load(rf_archivo)
    with open("output_penguin.pickle", "rb") as salida_archivo:
        especies = pickle.load(salida_archivo)
    return pipeline, especies


@st.cache_data
def cargar_datos_base():
    df = sns.load_dataset("penguins")
    return df.dropna()


rf_pipeline, especies_unicas = cargar_artefactos()
pinguinos_df = cargar_datos_base()

st.subheader("Ingrese las características del pingüino:")

with st.form("formulario_pinguino"):
    col1, col2 = st.columns(2)

    with col1:
        isla = st.selectbox("Isla de origen", options=["Biscoe", "Dream", "Torgerson"])
        sexo = st.selectbox("Sexo", options=["Male", "Female"])
        longitud_pico = st.number_input("Longitud del pico (mm)", min_value=0.0, value=40.0)

    with col2:
        profundidad_pico = st.number_input("Profundidad del pico (mm)", min_value=0.0, value=18.0)
        longitud_aleta = st.number_input("Longitud de la aleta (mm)", min_value=0.0, value=200.0)
        masa_corporal = st.number_input("Masa corporal (g)", min_value=0.0, value=4000.0)

    boton_enviar = st.form_submit_button("Predecir Especie", use_container_width=True)

if boton_enviar:
    datos_usuario_df = pd.DataFrame(
        [
            {
                "island": isla,
                "bill_length_mm": longitud_pico,
                "bill_depth_mm": profundidad_pico,
                "flipper_length_mm": longitud_aleta,
                "body_mass_g": masa_corporal,
                "sex": sexo,
            }
        ]
    )

    prediccion_indice = rf_pipeline.predict(datos_usuario_df)[0]
    especie_predicha = especies_unicas[prediccion_indice]

    st.success(f"**Resultado de la predicción:** El pingüino pertenece a la especie **{especie_predicha}**.")

    tab1, tab2 = st.tabs(["Importancia de Características", "Distribución de Datos"])

    with tab1:
        st.write("### Importancia relativa de cada variable en la decisión del modelo")
        st.image("feature_importance.png")

    with tab2:
        st.write("### Posición del pingüino evaluado respecto a la población base")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        sns.histplot(data=pinguinos_df, x="bill_length_mm", hue="species", ax=axes[0], kde=True)
        axes[0].axvline(longitud_pico, color="black", linestyle="--", linewidth=2)
        axes[0].set_title("Longitud del Pico (mm)")

        sns.histplot(data=pinguinos_df, x="bill_depth_mm", hue="species", ax=axes[1], kde=True)
        axes[1].axvline(profundidad_pico, color="black", linestyle="--", linewidth=2)
        axes[1].set_title("Profundidad del Pico (mm)")

        sns.histplot(data=pinguinos_df, x="flipper_length_mm", hue="species", ax=axes[2], kde=True)
        axes[2].axvline(longitud_aleta, color="black", linestyle="--", linewidth=2)
        axes[2].set_title("Longitud de la Aleta (mm)")

        plt.tight_layout()
        st.pyplot(fig)
