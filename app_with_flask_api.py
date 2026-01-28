# Version Streamlit avec API Flask
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets

# Configuration
st.set_page_config(page_title="Iris ML avec API", page_icon="🌸", layout="wide")

# URL de l'API (à modifier selon votre configuration)
API_URL = st.sidebar.text_input("🔗 URL de l'API", value="http://localhost:5000")

st.title("🌸 Dashboard Iris avec API Flask")
st.markdown("---")

# Charger les données pour les visualisations
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    df['Species'] = df['Species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    return df

df = load_data()

# Vérifier l'état de l'API
st.sidebar.markdown("### 🔌 État de l'API")
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API connectée")
        health_data = health_response.json()
        st.sidebar.info(f"Modèles chargés: {health_data.get('models_loaded', 'N/A')}")
    else:
        st.sidebar.error("❌ API non disponible")
except:
    st.sidebar.error("❌ Impossible de se connecter à l'API")
    st.sidebar.info("💡 Lancez d'abord l'API avec: `python api_flask.py`")

# Section principale
tab1, tab2, tab3 = st.tabs(["🎯 Prédiction Simple", "📊 Prédictions Multiples", "📈 Visualisations"])

with tab1:
    st.header("🎯 Prédiction d'Espèce via API")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Formulaire de Saisie")
        
        with st.form("api_prediction_form"):
            sepal_length = st.slider(
                "🌿 Longueur du Sépale (cm)",
                4.0, 8.0, 5.8, 0.1
            )
            
            sepal_width = st.slider(
                "🌿 Largeur du Sépale (cm)",
                2.0, 4.5, 3.0, 0.1
            )
            
            petal_length = st.slider(
                "🌸 Longueur du Pétale (cm)",
                1.0, 7.0, 4.0, 0.1
            )
            
            petal_width = st.slider(
                "🌸 Largeur du Pétale (cm)",
                0.1, 2.5, 1.3, 0.1
            )
            
            # Récupérer les modèles disponibles
            try:
                models_response = requests.get(f"{API_URL}/models", timeout=2)
                if models_response.status_code == 200:
                    available_models = models_response.json().get('available_models', ['knn'])
                else:
                    available_models = ['knn']
            except:
                available_models = ['knn']
            
            model_choice = st.selectbox("🤖 Choisir le Modèle", available_models)
            
            submit = st.form_submit_button("🔮 Envoyer à l'API", use_container_width=True)
    
    with col2:
        st.subheader("📊 Résultats de l'API")
        
        if submit:
            # Préparer la requête
            payload = {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width,
                'model': model_choice
            }
            
            try:
                # Envoyer la requête POST
                with st.spinner("🔄 Requête en cours..."):
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=5
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('success'):
                        # Afficher la prédiction
                        species = result['prediction']['species']
                        
                        species_colors = {
                            'setosa': '#66b3ff',
                            'versicolor': '#99ff99',
                            'virginica': '#ff9999'
                        }
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; 
                                    background-color: {species_colors[species]}; 
                                    text-align: center;">
                            <h1 style="color: white; margin: 0;">
                                🌸 {species.upper()}
                            </h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Afficher les probabilités si disponibles
                        if 'probabilities' in result:
                            st.markdown("### 📈 Probabilités")
                            proba = result['probabilities']
                            
                            proba_df = pd.DataFrame([
                                {'Espèce': k, 'Probabilité': v}
                                for k, v in proba.items()
                            ]).sort_values('Probabilité', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            bars = ax.barh(
                                proba_df['Espèce'], 
                                proba_df['Probabilité'],
                                color=[species_colors[s] for s in proba_df['Espèce']]
                            )
                            ax.set_xlabel('Probabilité')
                            ax.set_xlim(0, 1)
                            ax.set_title('Probabilités par Espèce')
                            
                            for i, row in proba_df.iterrows():
                                ax.text(
                                    row['Probabilité'] + 0.02, 
                                    list(proba_df['Espèce']).index(row['Espèce']), 
                                    f"{row['Probabilité']:.2%}",
                                    va='center'
                                )
                            
                            st.pyplot(fig)
                            plt.close()
                            
                            st.metric("🎯 Confiance", f"{result.get('confidence', 0):.2%}")
                        
                        # Afficher les détails de la requête
                        with st.expander("📋 Détails de la Réponse API"):
                            st.json(result)
                    else:
                        st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                else:
                    st.error(f"❌ Erreur HTTP {response.status_code}")
                    st.code(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout: L'API ne répond pas")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Erreur de connexion: Vérifiez que l'API est lancée")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

with tab2:
    st.header("📊 Prédictions Multiples")
    
    st.markdown("""
    Testez l'API avec plusieurs échantillons à la fois.
    Vous pouvez soit utiliser des exemples aléatoires, soit uploader un fichier CSV.
    """)
    
    option = st.radio(
        "Choisir la source des données:",
        ["📝 Saisie manuelle", "🎲 Exemples aléatoires", "📁 Upload CSV"]
    )
    
    samples = []
    
    if option == "🎲 Exemples aléatoires":
        n_samples = st.slider("Nombre d'échantillons", 1, 10, 5)
        if st.button("🎲 Générer des échantillons aléatoires"):
            # Générer des échantillons aléatoires depuis le dataset
            random_samples = df.sample(n=n_samples)
            samples = [
                {
                    'sepal_length': float(row['SepalLength']),
                    'sepal_width': float(row['SepalWidth']),
                    'petal_length': float(row['PetalLength']),
                    'petal_width': float(row['PetalWidth'])
                }
                for _, row in random_samples.iterrows()
            ]
            st.session_state['samples'] = samples
            st.session_state['samples_df'] = random_samples
    
    elif option == "📁 Upload CSV":
        uploaded_file = st.file_uploader("Upload fichier CSV", type=['csv'])
        if uploaded_file:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.dataframe(uploaded_df.head())
                
                if st.button("📤 Utiliser ce fichier"):
                    samples = [
                        {
                            'sepal_length': float(row[0]),
                            'sepal_width': float(row[1]),
                            'petal_length': float(row[2]),
                            'petal_width': float(row[3])
                        }
                        for _, row in uploaded_df.iterrows()
                    ]
                    st.session_state['samples'] = samples
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {e}")
    
    if 'samples' in st.session_state and len(st.session_state['samples']) > 0:
        st.success(f"✅ {len(st.session_state['samples'])} échantillons prêts")
        
        # Choisir le modèle
        try:
            models_response = requests.get(f"{API_URL}/models", timeout=2)
            available_models = models_response.json().get('available_models', ['knn'])
        except:
            available_models = ['knn']
        
        batch_model = st.selectbox("🤖 Modèle pour les prédictions", available_models, key='batch_model')
        
        if st.button("🚀 Lancer les prédictions (API)", use_container_width=True):
            payload = {
                'samples': st.session_state['samples'],
                'model': batch_model
            }
            
            try:
                with st.spinner("🔄 Prédictions en cours..."):
                    response = requests.post(
                        f"{API_URL}/predict_batch",
                        json=payload,
                        timeout=10
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('success'):
                        predictions = result['predictions']
                        
                        # Créer un DataFrame avec les résultats
                        results_data = []
                        for i, pred in enumerate(predictions):
                            sample = st.session_state['samples'][i]
                            results_data.append({
                                'Index': i,
                                'Sepal L': sample['sepal_length'],
                                'Sepal W': sample['sepal_width'],
                                'Petal L': sample['petal_length'],
                                'Petal W': sample['petal_width'],
                                'Prédiction': pred['prediction']['species'],
                                'Confiance': pred.get('confidence', None)
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        
                        st.subheader("📊 Résultats")
                        if 'Confiance' in results_df.columns and results_df['Confiance'].notna().all():
                            st.dataframe(
                                results_df.style.format({'Confiance': '{:.2%}'})
                                .background_gradient(subset=['Confiance'], cmap='YlGn')
                            )
                        else:
                            st.dataframe(results_df)
                        
                        # Graphique de distribution
                        st.subheader("📈 Distribution des Prédictions")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        results_df['Prédiction'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
                        ax.set_title('Distribution des espèces prédites')
                        ax.set_xlabel('Espèce')
                        ax.set_ylabel('Nombre')
                        st.pyplot(fig)
                        plt.close()
                        
                    else:
                        st.error(f"❌ Erreur: {result.get('error')}")
                else:
                    st.error(f"❌ Erreur HTTP {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

with tab3:
    st.header("📈 Visualisations du Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution des Espèces")
        fig, ax = plt.subplots(figsize=(8, 5))
        df['Species'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Nombre d\'échantillons par espèce')
        ax.set_xlabel('Espèce')
        ax.set_ylabel('Nombre')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Corrélations")
        fig, ax = plt.subplots(figsize=(8, 6))
        correlation = df.drop("Species", axis=1).corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Matrice de corrélation')
        st.pyplot(fig)
        plt.close()
    
    st.subheader("Nuages de Points")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for species in df['Species'].unique():
        species_df = df[df['Species'] == species]
        axes[0].scatter(species_df['SepalLength'], species_df['SepalWidth'], label=species, alpha=0.7)
    axes[0].set_xlabel('Longueur Sépale')
    axes[0].set_ylabel('Largeur Sépale')
    axes[0].set_title('Sépales')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for species in df['Species'].unique():
        species_df = df[df['Species'] == species]
        axes[1].scatter(species_df['PetalLength'], species_df['PetalWidth'], label=species, alpha=0.7)
    axes[1].set_xlabel('Longueur Pétale')
    axes[1].set_ylabel('Largeur Pétale')
    axes[1].set_title('Pétales')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("---")
st.markdown("""
💻 **Dashboard Streamlit avec API Flask** | 🌸 Dataset Iris  
📡 Communication via requêtes POST/GET | 🤖 Prédictions en temps réel
""")
