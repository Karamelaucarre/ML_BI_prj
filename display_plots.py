from sklearn.metrics import confusion_matrix, roc_curve, auc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --- Fonctions de Plotting (Internes) ---

def _plot_confusion_matrix(y_true, y_pred, df_name):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    class_names = [0, 1] 
    
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                cbar=False,
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.ylabel('Valeur Réelle (True Label)')
    plt.xlabel('Valeur Prédite (Predicted Label)')
    plt.title(f'Matrice de Confusion ({df_name})')

def _plot_roc_curve(model, X_test, y_true, df_name):
    
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print("ATTENTION: Modèle sans méthode predict_proba ou decision_function. ROC/AUC non tracé.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Courbe ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(f'Courbe ROC et AUC ({df_name})')
    plt.legend(loc="lower right")

def _plot_feature_importance(model, X_test, df_name):
    
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        importances = importances.nlargest(15)

        plt.figure(figsize=(10, 6))
        importances.sort_values().plot(kind='barh', color='teal')
        plt.title(f'Top 15 - Importance des Features ({df_name} / {type(model).__name__})')
        plt.xlabel('Importance Relative')
    else:
        print(f"ATTENTION: {type(model).__name__} ne supporte pas feature_importances_.")

# --- Fonction Principale ---

def plot_model_diagnostics(model_path, X_test, y_test, df_name):
    """
    Charge un modèle sérialisé et génère les visualisations de performance.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print(f"Modèle '{type(model).__name__}' chargé avec succès.")
        
    except FileNotFoundError:
        print(f"Erreur: Fichier modèle non trouvé à l'emplacement: {model_path}")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return

    # Prédictions
    X_input = X_test if isinstance(X_test, np.ndarray) else X_test.values
    
    y_pred = model.predict(X_input)

    # Génération des Graphiques
    print("\n--- Visualisation des Performances ---")
    
    _plot_confusion_matrix(y_test, y_pred, df_name)
    _plot_roc_curve(model, X_input, y_test, df_name)
    
    if isinstance(X_test, pd.DataFrame):
        _plot_feature_importance(model, X_test, df_name)
    else:
        print("ATTENTION: Le tracé de l'importance des features nécessite X_test en tant que DataFrame.")

    plt.show()