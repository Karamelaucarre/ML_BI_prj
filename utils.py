import pandas as pd
import streamlit as st

@st.cache_data
def load_data(dataset_name, df_type = 'original'):
    # Chargez les données en fonction de la sélection
    if dataset_name == "AmesHousing":
        return pd.read_csv(f'./data/{df_type}/AmesHousing.csv'), 'ameshousing'
    elif dataset_name == "Adult":
        return pd.read_csv(f'./data/{df_type}/adult.csv'), 'adult'
    elif dataset_name == "WDBC":
        return pd.read_csv(f'./data/{df_type}/wdbc.csv'), 'wdbc'
    return None, None # Pour 'Uploader un CSV' initial