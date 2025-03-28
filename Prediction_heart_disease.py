import pandas as pd
import seaborn as sns
import joblib
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import os
# Chargement des données
df = pd.read_csv('framingham.csv')
df.info()
df.head()
df.describe()
# Gestion des valeurs manquantes
df.fillna(df.median(), inplace=True)
#Suppression des colonnes vides:
df = df.loc[:, ~df.isna().all()]
# Suppression des doublons
df.drop_duplicates(inplace=True)
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
#Suppression des colonnes à variance nulle
df = df.loc[:, df.var() != 0]

# Séparation des features et de la cible
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Gestion du déséquilibre avec SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

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
# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', C=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "SVM": SVC(kernel="linear", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

## Entraînement et évaluation des modèles
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = []
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else 0

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob) if y_pred_prob is not None else (None, None, None)

        fpr_dict[name], tpr_dict[name] = fpr, tpr
        auc_dict[name] = auc

        results.append({
            "Modèle": name, "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1 Score": f1, "AUC": auc
        })
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"])
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.title(f"Matrice de confusion - {name}")
        plt.show()

    return pd.DataFrame(results), fpr_dict, tpr_dict, auc_dict

# Exécution de l'évaluation
results_df, fpr_dict, tpr_dict, auc_dict = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test)

# Sélection du meilleur modèle
best_model_row = results_df.sort_values(by="AUC", ascending=False).iloc[0]
best_model_name = best_model_row["Modèle"]
best_model = models[best_model_name]

# Sauvegarde du meilleur modèle
joblib.dump(best_model, "best_model.pkl")
# Sauvegarder le scaler
joblib.dump(scaler, "scaler.pkl")
print(f"🏆 Meilleur modèle : {best_model_name} sauvegardé sous 'best_model.pkl'.")

# Visualisation des scores AUC
plt.figure(figsize=(10, 5))
sns.barplot(x="Modèle", y="AUC", data=results_df.sort_values(by="AUC", ascending=False), palette="Blues_r")
plt.xticks(rotation=45)
plt.title("Comparaison des modèles selon l'AUC")
plt.show()

# Visualisation des courbes ROC pour tous les modèles
plt.figure(figsize=(10, 8))
for name in models.keys():
    if fpr_dict[name] is not None and tpr_dict[name] is not None:
        plt.plot(fpr_dict[name], tpr_dict[name], label=f'{name} (AUC = {auc_dict[name]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonale aléatoire
plt.title("Courbes ROC pour tous les modèles")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.legend(loc="lower right")
plt.show()

# Courbe ROC pour le meilleur modèle
y_pred_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_prob_best)
roc_auc_best = auc(fpr_best, tpr_best)

plt.figure(figsize=(10, 8))
plt.plot(fpr_best, tpr_best, label=f'{best_model_name} (AUC = {roc_auc_best:.2f})', color='red')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonale aléatoire
plt.title(f"Courbe ROC pour le meilleur modèle : {best_model_name}")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.legend(loc="lower right")
plt.show()
# Matrice de confusion pour le meilleur modèle
y_pred_best = best_model.predict(X_test_scaled)
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"])
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title(f"Matrice de confusion - {best_model_name}")
plt.show()
### deploiement du projet  sur Streamlit ###

# Charger le modèle et le scaler
# Informations sur l'application (sidebar avec Markdown)
st.sidebar.title('Prédiction des Maladies Cardiaques 🫀')

# Affichage des emojis dans la barre latérale
st.image("Images/Image1.jpg", caption="Prediction Heart disease",use_container_width =True)
#st.image("Images/Image2.jpg", caption="Illustration Représentative", use_container_width=True)
st.sidebar.markdown("""
Bienvenue dans notre application dédiée à la **prédiction du risque de maladies cardiaques** ! 🫀💓

### 🚨 **Définition des maladies cardiaques**
Les maladies cardiaques désignent un ensemble de troubles affectant le cœur et les vaisseaux sanguins, notamment :
- **Infarctus du myocarde** 💔
- **Angine de poitrine** 💥
- **Insuffisance cardiaque** 🫣

### 🌟 **Causes des maladies cardiaques** 🌟
Elles peuvent être causées par divers facteurs, tels que :
- **Hypertension artérielle** ⬆️ : Augmente la charge de travail du cœur.
- **Hypercholestérolémie** 🧴 : Accumulation de plaques dans les artères.
- **Tabagisme** 🚬 : Endommage les vaisseaux sanguins et réduit l'apport en oxygène.
- **Obésité et sédentarité** ⚖️ : Risques liés à un mode de vie inactif.
- **Diabète** 🍩 : Risque accru de complications cardiaques.

### 🧐 **Objectif de l'étude**
L'objectif est de construire un modèle qui prédit le risque de maladies cardiaques basé sur les caractéristiques médicales et de mode de vie. Cela peut aider les professionnels de santé à détecter les risques chez les patients et à prendre des mesures préventives.

### 🔧 **Modèles utilisés** :
Nous avons utilisé plusieurs modèles de Machine Learning pour cette tâche :
1. **Régression Logistique** 🔄 : Pour des prédictions simples mais efficaces.
2. **Random Forest** 🌲🌲 : Pour des performances robustes avec des arbres de décision multiples.
3. **XGBoost** 🚀 : Un modèle puissant basé sur le boosting d'arbres de décision.
4. **SVM (Support Vector Machine)** 🧑‍💻 : Pour les problèmes de classification avec des frontières de décision.
5. **KNN (K-Nearest Neighbors)** 👯‍♂️ : Basé sur les voisins les plus proches pour classifier les patients.
6. **Naive Bayes** 📊 : Un modèle probabiliste simple mais efficace.

Tous ces modèles ont été évalués avec des critères comme la **précision** (accuracy), le **rappel** (recall), la **précision** (precision), le **score F1**, et l'**AUC** (Area Under the Curve).

### ⚠️ **Attention :** 
Les résultats de la prédiction sont une estimation basée sur des données médicales. Il est important de consulter un professionnel de santé pour un diagnostic exact. 🩺

""")
def user_input_features():
    st.title('Prédiction du Risque de Maladies Cardiaques 💖')
 # Dictionary to map education levels to numerical values
    education_map = {
        "Élémentaire": 1,
        "Secondaire": 2,
        "Collège": 3,
        "Lycée": 4,
        "Université": 5,
        "Post-universitaire": 6,
        "Doctorat": 7
    }
    features = {
        'male': st.selectbox('Sexe biologique (Masculin, Féminin)', ['Féminin', 'Masculin']),
        'age': st.number_input('Âge du patient en années 🧑‍🦳', 20, 100, 50),
        'education': st.selectbox(
            "Niveau d'éducation atteint",
            list(education_map.keys())
        ),
        'currentSmoker': st.selectbox('Le patient est-il fumeur ? (Non, Oui) 🚬', ['Non', 'Oui']),
    }

    # Ajouter la logique conditionnelle pour le nombre de cigarettes fumées
    if 'currentSmoker' in features:  # Vérifier que la clé 'currentSmoker' est présente
        if features['currentSmoker'] == 'Oui':
            features['cigsPerDay'] = st.number_input('Nombre de cigarettes fumées par jour 🚬', 0, 100, 0)
        else:
            features['cigsPerDay'] = 0

    additional_features = {
        'prevalentHyp': st.selectbox("Présence d'hypertension artérielle ? (Non, Oui) 💔", ['Non', 'Oui']),
        'totChol': st.number_input('Taux de cholestérol total en mg/dL 🧪', 100, 600, 200),
        'sysBP': st.number_input('Pression artérielle systolique (mmHg) 💉', 80, 200, 120),
        'diaBP': st.number_input('Pression artérielle diastolique (mmHg) 💉', 40, 150, 80),
        'BMI': st.number_input('Indice de Masse Corporelle (IMC) ⚖️', 10.0, 60.0, 25.0),
        'heartRate': st.number_input('Fréquence cardiaque (battements par minute) ❤️', 40, 200, 70),
        'glucose': st.number_input('Taux de glucose sanguin en mg/dL 🩸', 40, 300, 100)
    }

    features.update(additional_features)

    # Convertir les colonnes catégorielles en numériques
    features['male'] = 1 if features['male'] == 'Masculin' else 0
    features['currentSmoker'] = 1 if features['currentSmoker'] == 'Oui' else 0
    features['prevalentHyp'] = 1 if features['prevalentHyp'] == 'Oui' else 0
    features['education'] = education_map[features['education']]    
    return pd.DataFrame([features])

user_input = user_input_features()


if st.button('Faire la prédiction 🧠'):
    
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')

    input_data_scaled = scaler.transform(user_input)
    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]  # Probabilité de la classe positive

    if prediction[0] == 0:
        st.success("Le patient n'a pas de risque élevé de maladie cardiaque. ✅")
    else:
        st.warning("Le patient présente un risque élevé de maladie cardiaque. ⚠️")

    st.write(f"Probabilité de risque élevé : {prediction_prob[0]:.4f}")


