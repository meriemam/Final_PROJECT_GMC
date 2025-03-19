import pandas as pd
import numpy as np
import joblib
import streamlit as st
import seaborn as sns
import time  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve

   ##### Chargement et analyse du jeu de données #####
df=pd.read_csv('framingham.csv')
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
   ##### Nettoyage et traitement de la base de données #####
# Gestion des des valeurs manquantes
def handle_missing_values(df):
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed
df = handle_missing_values(df)
print(df.isnull().sum())
#Suppression des doublons
df = df.drop_duplicates()
print(df.duplicated().sum())
#Suppression des colonnes vides:
df = df.loc[:, ~df.isna().all()]
#La visualisation et la gestion des valeurs aberrantes
numeric_cols = [col for col in df.select_dtypes(include=["number"]).columns if col != 'TenYearCHD']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()
def remove_outliers(df, numeric_cols):
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask]
df = remove_outliers(df, numeric_cols)
df = handle_missing_values(df)
#Suppression des colonnes à variance nulle
df = df.loc[:, df.var() != 0]
  ##### Modélisation ######
# Séparation des caractéristiques et de la variable cible
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]
#Gestion du déséquilibre des classes avec SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name="TenYearCHD")], axis=1)
# La matrice de corrélation
matrice_correlation = df.corr()
print("Matrice de Corrélation :")
print(matrice_correlation)
# Visualisation de la matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(matrice_correlation, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Matrice de Corrélation')
plt.show()
#Tester la multicolinéarité
X_const = add_constant(X)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print(vif_data)
plt.figure(figsize=(8, 6))
sns.barplot(x='VIF', y='Variable', data=vif_data.sort_values('VIF', ascending=False), palette='viridis')
plt.title("Variance Inflation Factor (VIF) pour chaque variable")
plt.xlabel("VIF")
plt.ylabel("Variable")
plt.show()
##### Modélisation Logistique #####
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ajustement des hyperparamètres avec GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

log_reg = LogisticRegression(random_state=42)
grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print("Meilleurs paramètres : ", grid_search.best_params_)
print("Meilleur score F1 : ", grid_search.best_score_)
print("Meilleur score AUC : ", grid_search.best_score_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

##### ÉVALUATION DU MODÉLE ####

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Visualisation de la matrice de confusion
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.title("Matrice de Confusion")
plt.ylabel("Réel")
plt.xlabel("Prédit")
plt.show()
# Visualisation du ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Courbe ROC")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend(loc="lower right")
plt.show()
# Enregistrer le modèle et le scaler après l'entraînement
joblib.dump(best_model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

### deploiement du projet  sur Streamlit ###

# Charger le modèle et le scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Fonction pour faire la prédiction
def make_prediction(input_data):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]  # Probabilité de la classe positive
    return prediction, prediction_prob

# Fonction pour afficher l'interface utilisateur et récupérer les données
def user_input_features():
    st.title('Prédiction du Risque de Maladies Cardiaques 💖')
    
    features = {
        'male': st.selectbox('Sexe biologique (0: Féminin, 1: Masculin) 👨👩', [0, 1]),
        'age': st.number_input('Âge du patient en années 🧑‍🦳', 20, 100, 50),
        'education': st.selectbox("Niveau d'éducation atteint (1: élémentaire à 7: universitaire)", range(1, 8)),
        'currentSmoker': st.selectbox('Le patient est-il fumeur ? (0: Non, 1: Oui) 🚬', [0, 1]),
        'cigsPerDay': st.number_input('Nombre de cigarettes fumées par jour 🚬', 0, 100, 0),
        'prevalentHyp': st.selectbox("Présence d'hypertension artérielle ? (0: Non, 1: Oui) 💔", [0, 1]),
        'totChol': st.number_input('Taux de cholestérol total en mg/dL 🧪', 100, 600, 200),
        'sysBP': st.number_input('Pression artérielle systolique (mmHg) 💉', 80, 200, 120),
        'diaBP': st.number_input('Pression artérielle diastolique (mmHg) 💉', 40, 150, 80),
        'BMI': st.number_input('Indice de Masse Corporelle (IMC) ⚖️', 10.0, 60.0, 25.0),
        'heartRate': st.number_input('Fréquence cardiaque (battements par minute) ❤️', 40, 200, 70),
        'glucose': st.number_input('Taux de glucose sanguin en mg/dL 🩸', 40, 300, 100)
    }
    
    return pd.DataFrame([features])

# Afficher les résultats de la prédiction
user_input = user_input_features()

if st.button('Faire la prédiction 🧠'):
    # Faire la prédiction avec les données d'entrée
    prediction, prediction_prob = make_prediction(user_input)

    # Afficher les résultats
    if prediction[0] == 0:
        st.success("Le patient n'a pas de risque élevé de maladie cardiaque. ✅")
    else:
        st.warning("Le patient présente un risque élevé de maladie cardiaque. ⚠️")

    # Afficher la probabilité de la classe positive (risque élevé)
    st.write(f"Probabilité de risque élevé : {prediction_prob[0]:.4f}")

    # Optionnel : Afficher la matrice de confusion pour les prédictions (si tu veux inclure cette partie)
    # Cette partie est plus pertinente pour des tests sur un ensemble de données de validation
    y_test = np.array([prediction[0]])  # Cela devrait être remplacé par les vraies cibles de validation
    conf_matrix = confusion_matrix(y_test, prediction)
    st.write("Matrice de confusion :")
    st.write(conf_matrix)

    # Visualiser la matrice de confusion
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title("Matrice de Confusion")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    st.pyplot()