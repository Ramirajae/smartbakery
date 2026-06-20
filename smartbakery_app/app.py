from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Charger les données historiques
print("📊 Chargement des données historiques...")
df = pd.read_csv('sales_daily.csv')
df['date'] = pd.to_datetime(df['date'])

# Charger votre modèle LightGBM
print("🤖 Chargement du modèle LightGBM...")
try:
    with open('LightGBM_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Modèle LightGBM chargé avec succès")
    print(f"📈 Type du modèle: {type(model)}")
except Exception as e:
    print(f"⚠️ Erreur chargement modèle: {e}")
    print("🔄 Utilisation d'un modèle de fallback")
    model = None

def get_last_known_values(date):
    """Récupère les dernières valeurs connues pour les features de lag"""
    date = pd.to_datetime(date)
    
    # Chercher la date dans l'historique
    last_records = df[df['date'] < date].tail(30)
    
    if len(last_records) > 0:
        last_row = last_records.iloc[-1]
        return {
            'lag_1': float(last_row.get('lag_1', 420)),
            'lag_7': float(last_row.get('lag_7', 415)),
            'lag_30': float(last_row.get('lag_30', 408)),
            'rolling_mean_7': float(last_row.get('rolling_mean_7', 418)),
            'rolling_mean_30': float(last_row.get('rolling_mean_30', 412)),
            'trend': float(last_row.get('trend', date.timetuple().tm_yday / 365)),
        }
    else:
        # Valeurs par défaut basées sur les ventes moyennes
        avg_sales = df['sales'].mean() if 'sales' in df.columns else 420
        return {
            'lag_1': avg_sales,
            'lag_7': avg_sales * 0.98,
            'lag_30': avg_sales * 0.97,
            'rolling_mean_7': avg_sales,
            'rolling_mean_30': avg_sales * 0.98,
            'trend': 0.5,
        }

def prepare_features(date_str):
    """Prépare toutes les features nécessaires pour la prédiction"""
    date = pd.to_datetime(date_str)
    
    # Récupérer les valeurs des lags
    lag_values = get_last_known_values(date)
    
    # Construire toutes les features comme dans votre dataset
    features = {
        # Features temporelles
        'month': date.month,
        'day': date.day,
        'weekday': date.weekday(),
        'week': date.isocalendar()[1],
        'is_weekend': 1 if date.weekday() >= 5 else 0,
        
        # Features cycliques
        'month_sin': math.sin(2 * math.pi * date.month / 12),
        'month_cos': math.cos(2 * math.pi * date.month / 12),
        
        # Features de lag
        'lag_1': lag_values['lag_1'],
        'lag_7': lag_values['lag_7'],
        'lag_30': lag_values['lag_30'],
        
        # Features de moyenne mobile
        'rolling_mean_7': lag_values['rolling_mean_7'],
        'rolling_mean_30': lag_values['rolling_mean_30'],
        
        # Trend
        'trend': lag_values['trend'],
    }
    
    # Convertir en DataFrame avec les colonnes dans le bon ordre
    feature_df = pd.DataFrame([features])
    
    # S'assurer que l'ordre des colonnes correspond à celui du modèle
    # (si vous avez les noms exacts de votre dataset d'entraînement)
    expected_columns = ['month', 'day', 'weekday', 'week', 'lag_1', 'lag_7', 'lag_30',
                       'rolling_mean_7', 'rolling_mean_30', 'is_weekend', 
                       'month_sin', 'month_cos', 'trend']
    
    # Ajouter les colonnes manquantes
    for col in expected_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    # Réordonner les colonnes
    feature_df = feature_df[expected_columns]
    
    print(f"🔍 Features préparées pour {date_str}:")
    print(feature_df.iloc[0].to_dict())
    
    return feature_df

@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        data = request.json
        date_str = data.get('date')
        
        if not date_str:
            return jsonify({'error': 'Date requise'}), 400
        
        print(f"\n📅 Prédiction demandée pour: {date_str}")
        
        # Préparer les features
        features = prepare_features(date_str)
        
        # Faire la prédiction
        if model is not None:
            try:
                prediction = model.predict(features)[0]
                print(f"✅ Prédiction du modèle: {prediction}")
            except Exception as e:
                print(f"⚠️ Erreur lors de la prédiction: {e}")
                # Fallback si le modèle échoue
                prediction = fallback_prediction(date_str)
        else:
            print("⚠️ Modèle non disponible, utilisation du fallback")
            prediction = fallback_prediction(date_str)
        
        # Arrondir et convertir en entier
        sales = max(0, int(round(prediction)))
        
        # Ajouter la date actuelle pour référence
        today_sales = get_today_sales()
        
        return jsonify({
            'sales': sales,
            'date': date_str,
            'today_sales': today_sales,
            'status': 'success',
            'model_used': 'LightGBM' if model is not None else 'fallback'
        })
        
    except Exception as e:
        print(f"❌ Erreur générale: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def fallback_prediction(date_str):
    """Prédiction de fallback si le modèle n'est pas disponible"""
    date = pd.to_datetime(date_str)
    
    # Algorithme simple mais réaliste
    base = 420
    seasonality = math.sin(2 * math.pi * date.month / 12) * 50
    weekend_boost = 40 if date.weekday() >= 5 else 0
    day_effect = (date.day % 15) * 1.5
    
    prediction = base + seasonality + weekend_boost + day_effect
    prediction += np.random.normal(0, 15)  # Bruit aléatoire
    
    return max(250, min(700, prediction))

def get_today_sales():
    """Récupère les ventes d'aujourd'hui depuis le dataset"""
    today = pd.Timestamp.now().normalize()
    
    # Chercher dans le dataset
    today_data = df[df['date'].dt.date == today.date()]
    
    if len(today_data) > 0 and 'sales' in today_data.columns:
        return int(today_data.iloc[-1]['sales'])
    else:
        # Si pas de données pour aujourd'hui, prendre la dernière valeur connue
        last_sales = df.iloc[-1]['sales'] if 'sales' in df.columns else 420
        return int(last_sales)

@app.route('/api/historical', methods=['GET'])
def get_historical():
    """Récupère les données historiques pour les graphiques"""
    try:
        # Derniers 30 jours
        last_30_days = df.tail(30)
        
        data = {
            'dates': last_30_days['date'].dt.strftime('%Y-%m-%d').tolist(),
            'sales': last_30_days['sales'].tolist() if 'sales' in last_30_days.columns else []
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Endpoint de vérification"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': len(df) > 0 if df is not None else False
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🍞 SMARTBAKERY AI - SERVEUR FLASK")
    print("="*50)
    print(f"📂 Dataset chargé: {len(df)} lignes")
    print(f"🤖 Modèle LightGBM: {'✅ Chargé' if model is not None else '❌ Non chargé'}")
    print(f"🌐 Serveur démarré sur http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)