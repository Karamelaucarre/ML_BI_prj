import numpy as np
import pandas as pd
import pickle
import os 

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)

# --- Fonctions Auxiliaires (Internes) ---

def _calculate_metrics(y_true, y_pred, task_type='classification'):
    metrics = {}
    
    if task_type == 'classification':
        acc = accuracy_score(y_true, y_pred)
        metrics = {
            'Accuracy': acc,
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        key_metric = ('Accuracy', acc, True)
        
    elif task_type == 'regression':
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2_Score': r2_score(y_true, y_pred)
        }
        key_metric = ('MSE', mse, False)
        
    else:
        raise ValueError("task_type doit être 'classification' ou 'regression'.")
        
    return metrics, key_metric


def _save_model(model, name, save_path, metric_info=None):
    os.makedirs(save_path, exist_ok=True)
    
    if metric_info:
        metric_name, metric_value = metric_info
        filename = f'{name}_{metric_name}_{metric_value:.4f}.pkl'
    else:
        filename = f'model_{name}.pkl'
        
    file_path = os.path.join(save_path, filename)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   Modèle '{name}' sauvegardé dans {file_path}")
    except Exception as e:
        print(f"   Erreur de sauvegarde du modèle '{name}' : {e}")


def _save_performance_df(df_results, save_path, model_name):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'performance_summary_{model_name}.csv')
    
    try:
        df_results.to_csv(file_path, index=False)
        print(f"\nRésumé des performances sauvegardé dans {file_path}")
    except Exception as e:
        print(f"Erreur de sauvegarde du résumé de performance : {e}")


# --- Fonction Principale ---

def train_save_models(models: dict, X, y, save_file_name: str, task_type='classification', save='all'):
    results = []
    
    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    
    # Initialisation des variables de suivi pour le meilleur modèle
    if task_type == 'classification':
        best_metric_value = -1.0
        metric_key_name = 'Accuracy'
        is_maximization = True
    elif task_type == 'regression':
        best_metric_value = np.inf
        metric_key_name = 'MSE'
        is_maximization = False
    else:
        raise ValueError("task_type doit être 'classification' ou 'regression'.")

    best_model_object = None
    best_model_name = None
    
    save_path = f'./models/{save_file_name}' 
    
    print(f"Début de l'entraînement pour la tâche : {task_type.upper()}")

    # Boucle d'entraînement et d'évaluation
    for name, model in models.items():
        print(f'-> {name} Training ...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calcul des métriques
        metrics_dict, (key_name, key_value, key_max) = _calculate_metrics(y_test, y_pred, task_type)
        
        metrics_dict['Model'] = name
        results.append(metrics_dict)
        
        print(f"  {key_name}: {key_value:.4f}")

        # Mise à jour du meilleur modèle
        if (is_maximization and key_value > best_metric_value) or \
           (not is_maximization and key_value < best_metric_value):
            
            best_metric_value = key_value
            best_model_object = model
            best_model_name = name
            
        # Sauvegarde de tous les modèles
        if save == 'all':
            _save_model(model, name, save_path)

    # Sauvegarde du MEILLEUR modèle (hors de la boucle)
    if save == 'only-best' and best_model_name:
        print(f"\nSauvegarde du MEILLEUR modèle: {best_model_name} ({metric_key_name}: {best_metric_value:.4f})")
        
        metric_info = (metric_key_name, best_metric_value)
        _save_model(best_model_object, best_model_name, save_path, metric_info)

    # SAUVEGARDE DU RÉSUMÉ DE PERFORMANCE GLOBAL
    df_results = pd.DataFrame(results)
    _save_performance_df(df_results, save_path, save_file_name) 

    return df_results