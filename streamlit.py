import streamlit as st
import pandas as pd
import plotly.express as px
import glob 
import os
import pickle
from utils import load_data
from sklearn.preprocessing import StandardScaler 

# --- Configuration Initiale ---
st.set_page_config(
    page_title="Projet ML - Analyse et Prédiction",
    layout="wide"
)

dataset = None
dataset_name = None
df_type = None

# --- Barre Latérale ---
with st.sidebar:
    st.markdown('<h1 style="font-size: 2.5em;">Configuration Globale</h1>', unsafe_allow_html=True)
    
    dataset_choice = st.radio(
        "Choisir le Dataset :",
        ["AmesHousing", "Adult", "WDBC"]
    )
    clas_datasets = ['Adult', 'WDBC']
    reg_datasets = ['AmesHousing']
    try:
        dataset, dataset_name = load_data(dataset_choice)
    
    except Exception:
        st.warning('Erreur de chargement du dataset.')

    if dataset_choice in reg_datasets:
            df_type = 'regression'
    elif dataset_choice in clas_datasets:
        df_type = 'classification'
        
# --- Contenu Principal (Onglets) ---
tab1, tab2, tab3 = st.tabs(["EDA & Pré-traitement", "Modélisation & Performance", "🚀 Prédiction"])


# --- Onglet 1 : Exploration des Données ---
with tab1:
    st.header("1. Exploration et Nettoyage des Données")
    
    if dataset is not None:
        st.subheader("Aperçu du Dataset")
        st.dataframe(dataset.head()) 
        
        st.subheader("Statistiques & Qualité des Données")
        col_stats, col_missing = st.columns(2)
        
        with col_stats:
            st.write("Description des Variables Numériques:")
            st.dataframe(dataset.describe().T) 
        
        with col_missing:
            st.write("Valeurs Manquantes:")
            missing_data = dataset.isnull().sum().sort_values(ascending=False)
            st.dataframe(missing_data[missing_data > 0]) 
            
        st.subheader("Distributions Interactives")
        feature_to_plot = st.selectbox("Choisir une Caractéristique", dataset.columns)

        fig = px.histogram(dataset, x=feature_to_plot, marginal="box")
        st.plotly_chart(fig, use_container_width=True) 


# --- Onglet 2 : Modélisation et Performance ---
with tab2:
    st.header("2. Entraînement et Évaluation des Modèles")
    st.subheader("Comparaison des Modèles Entraînés")

    if dataset_name: 
        try:
            df_performance = pd.read_csv(f'./models/{dataset_name}/performance_summary_{dataset_name}.csv')
            
            if df_type == 'classification':
                sort_by = 'Accuracy'
                ascending = False 
                title = "Accuracy par Modèle"
            elif df_type == 'regression':
                sort_by = 'MSE'
                ascending = True 
                title = "MSE par Modèle (Plus Bas est Mieux)"
            else:
                sort_by = df_performance.columns[1] 
                ascending = False
                title = "Performance par Modèle (Générique)"

            st.dataframe(df_performance.sort_values(by=sort_by, ascending=ascending), use_container_width=True)
            
            fig_comp = px.bar(df_performance, x='Model', y=sort_by, title=title)
            st.plotly_chart(fig_comp, use_container_width=True)

        except FileNotFoundError:
            st.error(f"Fichier de performance non trouvé pour '{dataset_name}'.")
        except Exception as e:
            st.error(f"Erreur lors du traitement des données de performance: {e}")

    else:
        st.info("Veuillez sélectionner un jeu de données dans la barre latérale pour voir les performances.")


# --- Onglet 3 : Prédiction ---
with tab3:
    st.header("3. Prédiction en Temps Réel")
    best_model = None
    
    if dataset_name and dataset is not None:
        models_dir = f'./models/{dataset_name}'
        
        best_model_files = glob.glob(os.path.join(models_dir, '*.pkl'))

        if best_model_files:
            best_model_file = best_model_files[0] 
            
            try:
                with open(best_model_file, 'rb') as f:
                    best_model = pickle.load(f)
                st.info(f"Modèle chargé pour la prédiction: **{os.path.basename(best_model_file)}**")
            except Exception as e:
                st.error(f"Erreur de chargement du modèle: {e}")

        else:
            st.warning("Aucun modèle sérialisé (.pkl) trouvé pour ce dataset.")

        if best_model is not None:
            st.subheader("Uploader les données à Prédire")
            input_file = st.file_uploader("Fichier CSV à prédire (mêmes colonnes que les données d'entraînement)", type=["csv"])
            
            if input_file is not None and st.button("Faire la Prédiction"):
                try:
                    input_data_df = pd.read_csv(input_file)
                    processed_input = input_data_df 
                    
                    predictions = best_model.predict(processed_input)

                    if df_type == 'regression':
                        sc = StandardScaler() 
                        predictions = sc.inverse_transform(predictions.reshape(-1, 1))[:,0] 
                        
                    results_df = pd.DataFrame(predictions, columns=['Prédiction'])
                    st.success("Prédictions générées avec succès!")
                    st.dataframe(results_df)

                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Télécharger les Prédictions (CSV)",
                        data=csv_export,
                        file_name=f'predictions_{dataset_name}.csv',
                        mime='text/csv',
                    )
                
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: Détail: {e}")