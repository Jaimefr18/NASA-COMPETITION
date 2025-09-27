import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import joblib
import json
from data_manager import DataManager

# Verificar disponibilidade do scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn disponível")
except ImportError as e:
    print(f"❌ Scikit-learn não disponível: {e}")
    SKLEARN_AVAILABLE = False
    # Criar classes dummy se scikit-learn não estiver disponível
    class RandomForestClassifier:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.array([0] * len(X))
        def predict_proba(self, X):
            return np.array([[0.5, 0.5]] * len(X))
        @property
        def feature_importances_(self):
            return np.array([0.1] * 10)
    
    class RandomForestRegressor:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.array([0.0] * len(X))
        @property
        def feature_importances_(self):
            return np.array([0.1] * 10)
    
    class StandardScaler:
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
    
    def train_test_split(X, y, **kwargs):
        split_idx = int(len(X) * 0.8)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def accuracy_score(y_true, y_pred):
        return 0.5
    
    def mean_squared_error(y_true, y_pred):
        return 0.1

class SatelliteBloomMLModel:
    """
    Modelo de Machine Learning para detecção e previsão de florações
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metadata = {}
        self.is_trained = False
        self.data_manager = DataManager()
        self.training_data = self.data_manager.get_training_data()

        print(f"✅ Dados de treinamento carregados: {len(self.training_data)} análises salvas")
        
    def add_training_data(self, analysis_data: Dict[str, Any]):
        """
        Adiciona dados de análise para treinamento futuro
        """
        success = self.data_manager.add_training_data(analysis_data)
        if success:
            self.training_data = self.data_manager.get_training_data()
            print(f"✅ Dados adicionados e salvos. Total: {len(self.training_data)}")
        else:
            print("❌ Erro ao salvar dados")
    
    def prepare_training_data(self, historical_data: List[Dict[str, Any]]):
        """
        Prepara os dados para treinamento a partir dos dados históricos acumulados
        """
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn não disponível para preparar dados")
            return None, None
            
        all_features = []
        all_targets = []
        
        for historical_record in historical_data:
            events = historical_record.get('events', [])
            sensor_data = historical_record.get('data', {})
            
            # Processar dados de cada sensor
            for sensor_name, data in sensor_data.items():
                if isinstance(data, list) and len(data) > 2:
                    try:
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                        
                        # Criar features para cada ponto de dados
                        for i, row in df.iterrows():
                            feature_vector = self._extract_features_from_sensor_data(df, row['date'], sensor_name)
                            if feature_vector is not None and len(feature_vector) > 0:
                                all_features.append(feature_vector)
                                event_target = self._find_event_for_date(events, row['date'], sensor_name)
                                all_targets.append(event_target)
                    except Exception as e:
                        print(f"❌ Erro processando {sensor_name}: {e}")
                        continue
        
        if len(all_features) == 0:
            return None, None
            
        return np.array(all_features), np.array(all_targets)
    
    def _extract_features_from_sensor_data(self, df: pd.DataFrame, target_date: datetime, sensor_name: str):
        """
        Extrai features relevantes dos dados do sensor
        """
        try:
            features = []
            
            # Dados dos últimos 90 dias
            start_date = target_date - timedelta(days=90)
            recent_data = df[df['date'] >= start_date]
            
            if len(recent_data) < 3:
                return None
            
            # Features de NDVI
            if 'NDVI' in df.columns:
                ndvi_values = recent_data['NDVI'].dropna()
                if len(ndvi_values) > 0:
                    features.extend([
                        float(ndvi_values.mean()),
                        float(ndvi_values.max()),
                        float(ndvi_values.std()),
                        float(ndvi_values.iloc[-1]) if len(ndvi_values) > 0 else 0.0,
                    ])
            
            # Features de clorofila
            if 'chlorophyll' in df.columns:
                chl_values = recent_data['chlorophyll'].dropna()
                if len(chl_values) > 0:
                    features.extend([
                        float(chl_values.mean()),
                        float(chl_values.max()),
                        float(chl_values.std()),
                        float(chl_values.iloc[-1]) if len(chl_values) > 0 else 0.0,
                    ])
            
            # Features de umidade do solo
            if 'soil_moisture' in df.columns:
                moisture_values = recent_data['soil_moisture'].dropna()
                if len(moisture_values) > 0:
                    features.extend([
                        float(moisture_values.mean()),
                        float(moisture_values.max()),
                        float(moisture_values.std()),
                        float(moisture_values.iloc[-1]) if len(moisture_values) > 0 else 0.0,
                    ])
            
            # Features temporais
            features.extend([
                float(target_date.month),
                float(target_date.dayofyear),
            ])
            
            return features
            
        except Exception as e:
            print(f"❌ Erro extraindo features de {sensor_name}: {e}")
            return None
    
    def _find_event_for_date(self, events: List[Dict], target_date: datetime, sensor_name: str):
        """
        Encontra evento correspondente à data para target
        """
        for event in events:
            try:
                event_start = datetime.strptime(event['start'], '%Y-%m-%d')
                event_end = datetime.strptime(event['end'], '%Y-%m-%d')
                
                if event_start <= target_date <= event_end and event['sensor'] == sensor_name:
                    return 1
            except:
                continue
        return 0
    
    def train_classification_model(self, X, y, model_name='bloom_classifier'):
        """
        Treina modelo de classificação para detecção de florações
        """
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn não disponível para treinamento")
            return None
            
        if len(X) == 0 or len(y) == 0:
            return None
        
        # Verificar se há eventos positivos
        if np.sum(y) < 5:
            print("❌ Eventos positivos insuficientes para treinamento")
            return None
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treinar Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Avaliar
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Modelo {model_name} treinado")
        print(f"📊 Acurácia: {accuracy:.3f}")
        print(f"📈 Eventos positivos: {np.sum(y)}/{len(y)}")
        
        # Salvar modelo e scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.feature_importance[model_name] = dict(zip(range(len(model.feature_importances_)), 
                                                      model.feature_importances_))
        
        self.model_metadata[model_name] = {
            'type': 'classification',
            'accuracy': float(accuracy),
            'trained_at': datetime.now().isoformat(),
            'features_count': X.shape[1],
            'classes': [int(cls) for cls in np.unique(y)]
        }
        
        self.is_trained = True
        return accuracy
    
    def train_regression_model(self, X, y, model_name='bloom_intensity'):
        """
        Treina modelo de regressão para prever intensidade das florações
        """
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn não disponível para treinamento")
            return None
            
        # Filtrar apenas casos com eventos para regressão
        event_indices = y > 0
        X_events = X[event_indices]
        y_events = y[event_indices]
        
        if len(X_events) < 10:
            print("❌ Dados insuficientes para treinar modelo de regressão")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_events, y_events, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"✅ Modelo de regressão {model_name} treinado")
        print(f"📊 MSE: {mse:.3f}")
        print(f"📈 Amostras de treino: {len(X_events)}")
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.feature_importance[model_name] = dict(zip(range(len(model.feature_importances_)), 
                                                      model.feature_importances_))
        
        self.model_metadata[model_name] = {
            'type': 'regression',
            'mse': float(mse),
            'trained_at': datetime.now().isoformat(),
            'features_count': X.shape[1],
            'training_samples': len(X_events)
        }
        
        return mse
    
    def predict_bloom_risk(self, current_sensor_data: Dict[str, Any], days_ahead: int = 7):
        """
        Prediz risco de floração baseado nos dados atuais dos sensores
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {
                    "error": "scikit-learn_not_available",
                    "message": "Scikit-learn não está disponível. Instale com: pip install scikit-learn"
                }
            
            if not self.is_trained or 'bloom_classifier' not in self.models:
                return {
                    "error": "model_not_trained",
                    "message": "Modelo não treinado. Use o endpoint /train-models primeiro."
                }
            
            features = self._extract_prediction_features(current_sensor_data, days_ahead)
            
            if features is None or len(features) == 0:
                return {"error": "no_features", "message": "Não foi possível extrair features dos dados atuais"}
            
            # Fazer predição com modelo de classificação
            scaler = self.scalers['bloom_classifier']
            model = self.models['bloom_classifier']
            
            features_scaled = scaler.transform([features])
            bloom_probability = model.predict_proba(features_scaled)[0][1]
            
            result = {
                'bloom_risk_probability': float(bloom_probability),
                'risk_level': self._classify_risk_level(bloom_probability),
                'prediction_date': (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d'),
                'features_used': len(features),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Se há risco significativo, prever intensidade
            if bloom_probability > 0.3 and 'bloom_intensity' in self.models:
                intensity_model = self.models['bloom_intensity']
                intensity_scaler = self.scalers['bloom_intensity']
                
                features_intensity_scaled = intensity_scaler.transform([features])
                predicted_intensity = intensity_model.predict(features_intensity_scaled)[0]
                
                result['predicted_intensity'] = float(predicted_intensity)
                result['intensity_level'] = self._classify_intensity_level(predicted_intensity)
            
            return result
            
        except Exception as e:
            return {"error": f"prediction_error: {str(e)}"}
    
    def _extract_prediction_features(self, sensor_data: Dict[str, Any], days_ahead: int):
        """
        Extrai features dos dados atuais dos sensores para predição
        """
        try:
            features = []
            current_date = datetime.now()
            prediction_date = current_date + timedelta(days=days_ahead)
            
            # Processar dados de cada sensor
            for sensor_name, data in sensor_data.items():
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Usar dados mais recentes (últimos 90 dias)
                    recent_data = df[df['date'] >= (current_date - timedelta(days=90))]
                    
                    if len(recent_data) < 3:
                        continue
                    
                    # Extrair features similares ao treinamento
                    if 'NDVI' in df.columns:
                        ndvi_values = recent_data['NDVI'].dropna()
                        if len(ndvi_values) > 0:
                            features.extend([
                                float(ndvi_values.mean()),
                                float(ndvi_values.max()),
                                float(ndvi_values.std()),
                                float(ndvi_values.iloc[-1]),
                            ])
                    
                    if 'chlorophyll' in df.columns:
                        chl_values = recent_data['chlorophyll'].dropna()
                        if len(chl_values) > 0:
                            features.extend([
                                float(chl_values.mean()),
                                float(chl_values.max()),
                                float(chl_values.std()),
                                float(chl_values.iloc[-1]),
                            ])
                    
                    if 'soil_moisture' in df.columns:
                        moisture_values = recent_data['soil_moisture'].dropna()
                        if len(moisture_values) > 0:
                            features.extend([
                                float(moisture_values.mean()),
                                float(moisture_values.max()),
                                float(moisture_values.std()),
                                float(moisture_values.iloc[-1]),
                            ])
            
            # Adicionar features temporais futuras
            features.extend([
                float(prediction_date.month),
                float(prediction_date.dayofyear),
            ])
            
            return features
            
        except Exception as e:
            print(f"❌ Erro extraindo features de predição: {e}")
            return None
    
    def _classify_risk_level(self, probability: float) -> str:
        """Classifica o nível de risco baseado na probabilidade"""
        if probability < 0.2:
            return "Baixo"
        elif probability < 0.5:
            return "Moderado"
        elif probability < 0.8:
            return "Alto"
        else:
            return "Muito Alto"
    
    def _classify_intensity_level(self, intensity: float) -> str:
        """Classifica o nível de intensidade"""
        if intensity < 0.1:
            return "Leve"
        elif intensity < 0.3:
            return "Moderado"
        elif intensity < 0.6:
            return "Forte"
        else:
            return "Severo"
    
    def get_model_status(self):
        """Retorna status dos modelos"""
        return {
            'is_trained': self.is_trained,
            'sklearn_available': SKLEARN_AVAILABLE,
            'trained_models': list(self.models.keys()),
            'training_data_points': len(self.training_data),
            'model_metadata': self.model_metadata
        }
    
    def save_models(self, filepath: str):
        """Salva os modelos treinados"""
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn não disponível para salvar modelos")
            return False
            
        try:
            for name, model in self.models.items():
                joblib.dump(model, f"{filepath}_{name}.joblib")
            
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, f"{filepath}_{name}_scaler.joblib")
            
            metadata = {
                'model_metadata': self.model_metadata,
                'feature_importance': self.feature_importance,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelos salvos em {filepath}")
            return True
        except Exception as e:
            print(f"❌ Erro salvando modelos: {e}")
            return False
    
    def load_models(self, filepath: str):
        """Carrega modelos salvos"""
        if not SKLEARN_AVAILABLE:
            print("❌ Scikit-learn não disponível para carregar modelos")
            return False
            
        try:
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.model_metadata = metadata['model_metadata']
            self.feature_importance = metadata['feature_importance']
            
            for name in self.model_metadata.keys():
                self.models[name] = joblib.load(f"{filepath}_{name}.joblib")
                self.scalers[name] = joblib.load(f"{filepath}_{name}_scaler.joblib")
            
            self.is_trained = True
            print("✅ Modelos carregados com sucesso")
            return True
        except Exception as e:
            print(f"❌ Erro carregando modelos: {e}")
            return False
    
    def get_data_stats(self):
        """Retorna estatísticas dos dados"""
        return self.data_manager.get_data_stats()