# Application Streamlit complète - Analyse et Prédiction Iris
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import datasets

# Imports pour le Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)
import pickle
import json

# Configuration de la page
st.set_page_config(
    page_title="Iris ML Dashboard",
    page_icon="🌸",
    layout="wide"
)

# Titre principal
st.title("🎯 Tableau de Bord Complet - Dataset Iris avec ML")
st.markdown("---")

# Chargement des données
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    df['Species'] = df['Species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    return df

# Fonction pour entraîner et sauvegarder le modèle
@st.cache_resource
def train_and_cache_model():
    df = load_data()
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraîner plusieurs modèles
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(probability=True),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    
    return trained_models, scaler, X_test_scaled, y_test

df = load_data()
trained_models, scaler, X_test, y_test = train_and_cache_model()

# Sidebar pour la navigation
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio(
    "Choisir une section:",
    ["📊 Aperçu des données", 
     "📈 Distribution des espèces", 
     "📉 Histogrammes",
     "📦 Boxplots",
     "🔵 Nuages de points",
     "🔗 Corrélations",
     "🎯 Analyses avancées",
     "🤖 Machine Learning - KNN",
     "🔧 Optimisation & Comparaison",
     "🎯 Dashboard de Prédiction"]
)

# Section 1: Aperçu des données
if section == "📊 Aperçu des données":
    st.header("Aperçu des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Premières lignes du dataset")
        st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Informations sur le dataset")
        st.write(f"**Nombre de lignes :** {df.shape[0]}")
        st.write(f"**Nombre de colonnes :** {df.shape[1]}")
        st.write(f"**Colonnes :** {', '.join(df.columns)}")
    
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe())

# Section 2: Distribution des espèces
elif section == "📈 Distribution des espèces":
    st.header("Distribution des espèces d'iris")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Effectifs par espèce (Barres)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='Species', data=df, ax=ax, palette='viridis')
        ax.set_title('Effectifs par espèce')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Répartition en secteurs")
        fig, ax = plt.subplots(figsize=(8, 5))
        df['Species'].value_counts().plot.pie(
            autopct='%1.1f%%', 
            ax=ax, 
            colors=['#66b3ff','#99ff99','#ff9999']
        )
        ax.set_title('Répartition en secteurs')
        ax.set_ylabel('')
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Nombre d'échantillons par espèce")
    st.dataframe(df['Species'].value_counts())

# Section 3: Histogrammes
elif section == "📉 Histogrammes":
    st.header("Histogrammes des variables quantitatives")
    
    variable = st.selectbox(
        "Sélectionner une variable:",
        ['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']
    )
    
    bins = st.slider("Nombre de bins:", min_value=5, max_value=30, value=10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[variable], bins=bins, color='steelblue', edgecolor='black')
    ax.set_title(f"Histogramme de {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Effectif")
    st.pyplot(fig)
    plt.close()

# Section 4: Boxplots
elif section == "📦 Boxplots":
    st.header("Boîtes à moustaches (Boxplots)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Longueur des pétales par espèce")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Species', y='PetalLength', ax=ax, palette='Set2')
        ax.set_title('Boxplot de la longueur des pétales par espèce')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Largeur des sépales par espèce")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Species', y='SepalWidth', ax=ax, palette='Set2')
        ax.set_title('Boxplot de la largeur des sépales par espèce')
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Boxplot personnalisé")
    variable_y = st.selectbox(
        "Choisir une variable à analyser:",
        ['PetalLength', 'PetalWidth', 'SepalLength', 'SepalWidth']
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='Species', y=variable_y, ax=ax, palette='coolwarm')
    ax.set_title(f'Boxplot de {variable_y} par espèce')
    st.pyplot(fig)
    plt.close()

# Section 5: Nuages de points
elif section == "🔵 Nuages de points":
    st.header("Nuages de points")
    
    st.subheader("Sépales : Longueur vs Largeur")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='SepalLength', y='SepalWidth', hue='Species', style='Species', s=100, ax=ax)
    ax.set_title('Nuage de points de la longueur et largeur des sépales par espèce')
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Pétales : Longueur vs Largeur")
    fig, ax = plt.subplots(figsize=(10, 6))
    for esp in df["Species"].unique():
        sous_df = df[df["Species"] == esp]
        ax.scatter(sous_df["PetalLength"], sous_df["PetalWidth"], label=esp, s=100, alpha=0.7)
    ax.set_title("Nuage de points pétales avec distinction par espèce")
    ax.set_xlabel("Longueur du pétale (cm)")
    ax.set_ylabel("Largeur du pétale (cm)")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# Section 6: Corrélations
elif section == "🔗 Corrélations":
    st.header("Corrélations entre variables quantitatives")
    
    correlation = df.drop("Species", axis=1).corr()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Matrice de corrélation")
        st.dataframe(correlation.style.background_gradient(cmap='coolwarm', axis=None))
    
    with col2:
        st.subheader("Heatmap des corrélations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
        ax.set_title('Heatmap des corrélations')
        st.pyplot(fig)
        plt.close()

# Section 7: Analyses avancées
elif section == "🎯 Analyses avancées":
    st.header("Analyses avancées")
    
    st.subheader("Pairplot - Relations entre toutes les variables")
    fig = sns.pairplot(df, hue='Species', height=2.5)
    fig.fig.suptitle('Pairplot des variables en fonction de l\'espèce', y=1.02)
    st.pyplot(fig.fig)
    plt.close()

# Section 8: Machine Learning - KNN
elif section == "🤖 Machine Learning - KNN":
    st.header("🤖 Machine Learning - K-Nearest Neighbors (KNN)")
    
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Caractéristiques (X) :**")
        st.dataframe(X.head())
    with col2:
        st.write("**Cible (y) :**")
        st.dataframe(y.head())
    
    test_size = st.slider("Taille de l'ensemble de test (%)", 10, 40, 20) / 100
    n_neighbors = st.slider("Nombre de voisins (k)", 1, 15, 3)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if st.button("🚀 Entraîner le modèle KNN"):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("🎯 Exactitude", f"{accuracy * 100:.2f}%")
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                   xticklabels=df['Species'].unique(), 
                   yticklabels=df['Species'].unique(), ax=ax)
        ax.set_title('Matrice de confusion')
        st.pyplot(fig)
        plt.close()

# Section 9: Optimisation
elif section == "🔧 Optimisation & Comparaison":
    st.header("🔧 Optimisation & Comparaison")
    
    X = df.drop('Species', axis=1)
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if st.button("🚀 Comparer tous les modèles"):
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(),
            'SVM': SVC(),
            'Neural Network': MLPClassifier(max_iter=1000)
        }
        
        results = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results.append({'Modèle': name, 'Exactitude': accuracy})
        
        results_df = pd.DataFrame(results).sort_values('Exactitude', ascending=False)
        st.dataframe(results_df.style.format({'Exactitude': '{:.2%}'}))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(results_df['Modèle'], results_df['Exactitude'], color='skyblue')
        ax.set_xlabel('Exactitude')
        ax.set_title('Comparaison des Modèles')
        st.pyplot(fig)
        plt.close()

# Section 10: Dashboard de Prédiction
elif section == "🎯 Dashboard de Prédiction":
    st.header("🎯 Tableau de Bord de Prédiction Interactive")
    
    st.markdown("""
    ### 📋 Instructions
    Utilisez les curseurs ci-dessous pour saisir les caractéristiques d'une fleur Iris.
    Le système prédi les caractéristiques d'une fleur et prédira son espèce en temps réel !
    """)
    
    # Créer deux colonnes : formulaire à gauche, résultats à droite
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Formulaire de Saisie")
        
        # Obtenir les statistiques pour les limites des sliders
        stats = df.describe()
        
        with st.form("prediction_form"):
            st.markdown("#### Caractéristiques de la Fleur")
            
            sepal_length = st.slider(
                "🌿 Longueur du Sépale (cm)",
                float(stats.loc['min', 'SepalLength']),
                float(stats.loc['max', 'SepalLength']),
                float(stats.loc['mean', 'SepalLength']),
                0.1
            )
            
            sepal_width = st.slider(
                "🌿 Largeur du Sépale (cm)",
                float(stats.loc['min', 'SepalWidth']),
                float(stats.loc['max', 'SepalWidth']),
                float(stats.loc['mean', 'SepalWidth']),
                0.1
            )
            
            petal_length = st.slider(
                "🌸 Longueur du Pétale (cm)",
                float(stats.loc['min', 'PetalLength']),
                float(stats.loc['max', 'PetalLength']),
                float(stats.loc['mean', 'PetalLength']),
                0.1
            )
            
            petal_width = st.slider(
                "🌸 Largeur du Pétale (cm)",
                float(stats.loc['min', 'PetalWidth']),
                float(stats.loc['max', 'PetalWidth']),
                float(stats.loc['mean', 'PetalWidth']),
                0.1
            )
            
            model_choice = st.selectbox(
                "🤖 Choisir le Modèle",
                list(trained_models.keys())
            )
            
            submit_button = st.form_submit_button("🔮 Prédire l'Espèce", use_container_width=True)
    
    with col2:
        st.subheader("📊 Résultats de la Prédiction")
        
        if submit_button:
            # Préparer les données
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)
            
            # Prédiction
            model = trained_models[model_choice]
            prediction = model.predict(input_scaled)[0]
            
            # Afficher la prédiction avec style
            st.markdown("### 🎯 Prédiction")
            
            # Couleurs pour chaque espèce
            species_colors = {
                'setosa': '#66b3ff',
                'versicolor': '#99ff99',
                'virginica': '#ff9999'
            }
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {species_colors[prediction]}; text-align: center;">
                <h1 style="color: white; margin: 0;">🌸 {prediction.upper()}</h1>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilités si le modèle le supporte
            if hasattr(model, 'predict_proba'):
                st.markdown("### 📈 Probabilités")
                proba = model.predict_proba(input_scaled)[0]
                species_list = ['setosa', 'versicolor', 'virginica']
                
                proba_df = pd.DataFrame({
                    'Espèce': species_list,
                    'Probabilité': proba
                }).sort_values('Probabilité', ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(proba_df['Espèce'], proba_df['Probabilité'], 
                              color=[species_colors[s] for s in proba_df['Espèce']])
                ax.set_xlabel('Probabilité')
                ax.set_xlim(0, 1)
                ax.set_title('Probabilités par Espèce')
                
                for i, (idx, row) in enumerate(proba_df.iterrows()):
                    ax.text(row['Probabilité'] + 0.02, i, f"{row['Probabilité']:.2%}", 
                           va='center')
                
                st.pyplot(fig)
                plt.close()
            
            # Comparaison avec les données du dataset
            st.markdown("### 📊 Comparaison avec le Dataset")
            
            # Filtrer les données de l'espèce prédite
            species_data = df[df['Species'] == prediction]
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Votre Sépale (L × l)", 
                         f"{sepal_length:.1f} × {sepal_width:.1f} cm")
                st.metric("Moyenne dans dataset", 
                         f"{species_data['SepalLength'].mean():.1f} × {species_data['SepalWidth'].mean():.1f} cm")
            
            with col_b:
                st.metric("Votre Pétale (L × l)", 
                         f"{petal_length:.1f} × {petal_width:.1f} cm")
                st.metric("Moyenne dans dataset", 
                         f"{species_data['PetalLength'].mean():.1f} × {species_data['PetalWidth'].mean():.1f} cm")
            
            # Visualisation : Position dans l'espace des caractéristiques
            st.markdown("### 🔵 Position dans l'Espace des Caractéristiques")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Graph 1: Sépales
            for species in df['Species'].unique():
                species_df = df[df['Species'] == species]
                axes[0].scatter(species_df['SepalLength'], species_df['SepalWidth'],
                              label=species, alpha=0.6, s=50)
            axes[0].scatter(sepal_length, sepal_width, 
                          color='red', s=200, marker='*', 
                          label='Votre fleur', edgecolors='black', linewidths=2)
            axes[0].set_xlabel('Longueur Sépale (cm)')
            axes[0].set_ylabel('Largeur Sépale (cm)')
            axes[0].set_title('Espace des Sépales')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Graph 2: Pétales
            for species in df['Species'].unique():
                species_df = df[df['Species'] == species]
                axes[1].scatter(species_df['PetalLength'], species_df['PetalWidth'],
                              label=species, alpha=0.6, s=50)
            axes[1].scatter(petal_length, petal_width, 
                          color='red', s=200, marker='*', 
                          label='Votre fleur', edgecolors='black', linewidths=2)
            axes[1].set_xlabel('Longueur Pétale (cm)')
            axes[1].set_ylabel('Largeur Pétale (cm)')
            axes[1].set_title('Espace des Pétales')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Section de comparaison de modèles pour la même entrée
    st.markdown("---")
    st.subheader("🔀 Comparaison entre Modèles")
    
    if submit_button:
        st.markdown("Voyons comment différents modèles classifient la même fleur :")
        
        comparison_results = []
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        
        for model_name, model in trained_models.items():
            pred = model.predict(input_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                max_proba = proba.max()
            else:
                max_proba = None
            
            comparison_results.append({
                'Modèle': model_name,
                'Prédiction': pred,
                'Confiance': max_proba
            })
        
        comp_df = pd.DataFrame(comparison_results)
        
        # Afficher le tableau
        if comp_df['Confiance'].notna().all():
            st.dataframe(
                comp_df.style.format({'Confiance': '{:.2%}'})
                .background_gradient(subset=['Confiance'], cmap='YlGn')
            )
        else:
            st.dataframe(comp_df)
        
        # Vérifier le consensus
        predictions = comp_df['Prédiction'].value_counts()
        if len(predictions) == 1:
            st.success(f"✅ **Consensus parfait !** Tous les modèles prédisent : **{predictions.index[0]}**")
        else:
            st.warning(f"⚠️ **Prédictions divergentes.** Majorité : **{predictions.index[0]}** ({predictions.iloc[0]}/{len(comp_df)} modèles)")
    
    # Filtres interactifs pour l'exploration
    st.markdown("---")
    st.subheader("🔍 Exploration Interactive du Dataset")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        filter_species = st.multiselect(
            "Filtrer par espèce",
            options=df['Species'].unique().tolist(),
            default=df['Species'].unique().tolist()
        )
    
    with col_f2:
        filter_sepal_min = st.number_input(
            "Longueur Sépale Min",
            value=float(df['SepalLength'].min()),
            min_value=float(df['SepalLength'].min()),
            max_value=float(df['SepalLength'].max())
        )
    
    with col_f3:
        filter_petal_min = st.number_input(
            "Longueur Pétale Min",
            value=float(df['PetalLength'].min()),
            min_value=float(df['PetalLength'].min()),
            max_value=float(df['PetalLength'].max())
        )
    
    # Appliquer les filtres
    filtered_df = df[
        (df['Species'].isin(filter_species)) &
        (df['SepalLength'] >= filter_sepal_min) &
        (df['PetalLength'] >= filter_petal_min)
    ]
    
    col_r1, col_r2 = st.columns([2, 1])
    
    with col_r1:
        st.markdown(f"**{len(filtered_df)} fleurs correspondent aux critères**")
        st.dataframe(filtered_df, height=300)
    
    with col_r2:
        st.markdown("**Distribution filtrée**")
        fig, ax = plt.subplots(figsize=(6, 4))
        filtered_df['Species'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_ylabel('')
        st.pyplot(fig)
        plt.close()

# Footer
st.markdown("---")
st.markdown("💻 Application développée avec Streamlit | 🌸 Dataset Iris | 🤖 Machine Learning")
st.markdown("✨ Dashboard interactif avec prédictions en temps réel")
