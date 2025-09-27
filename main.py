import os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Importar o módulo de ML
from ML import SatelliteBloomMLModel

# Importar o DataManager para salvar os dados de treinamento
from data_manager import DataManager

# =============================================================================
# CONFIGURAÇÃO EARTH ENGINE
# =============================================================================
try:
    import ee
    EE_AVAILABLE = True
    print("✅ Earth Engine importado com sucesso")
except ImportError:
    ee = None
    EE_AVAILABLE = False
    print("❌ Earth Engine não instalado")

def initialize_earth_engine():
    """Inicializa o Earth Engine de forma robusta"""
    if not EE_AVAILABLE:
        return False, "Earth Engine não disponível"

    try:
        ee.Initialize()
        print("✅ Earth Engine inicializado com sucesso")
        return True, "Inicializado com sucesso"
    except ee.EEException as e:
        print(f"🔐 Precisa autenticar: {e}")
        return False, "Precisa autenticar - execute 'earthengine authenticate'"
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False, f"Erro inesperado: {e}"

EE_WORKING, EE_MESSAGE = initialize_earth_engine()
print(f"Earth Engine Status: {EE_WORKING} - {EE_MESSAGE}")

# =============================================================================
# CONFIGURAÇÃO FASTAPI
# =============================================================================
app = FastAPI(
    title="Bloom Detection API - Multi-Sensor com GRACE e ML",
    description="API para detecção de florações com dados de satélite e Machine Learning",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# INICIALIZAÇÃO DO MODELO DE ML E DATA MANAGER
# =============================================================================
ml_model = SatelliteBloomMLModel()
print("✅ Modelo de Machine Learning inicializado")

# Inicializar DataManager
try:
    data_manager = DataManager()
    print("✅ DataManager inicializado com sucesso")
except Exception as e:
    print(f"❌ Erro ao inicializar DataManager: {e}")
    data_manager = None

# =============================================================================
# MODELOS DE DADOS
# =============================================================================
class AnalyzeRequest(BaseModel):
    lon: float = Field(-8.8383, description="Longitude")
    lat: float = Field(13.2344, description="Latitude")
    start: str = Field("2020-01-01", description="Data início YYYY-MM-DD")
    end: str = Field("2024-12-31", description="Data fim YYYY-MM-DD")
    buffer_km: float = Field(50.0, description="Raio da área em km")
    sensors: List[str] = Field(['landsat', 'modis', 'viirs', 'smap', 'grace'], description="Sensores a usar")
    grace_analysis: str = Field("water_mass", description="Tipo de análise GRACE")
    include_ml_predictions: bool = Field(True, description="Incluir predições de ML")

class TrainRequest(BaseModel):
    min_data_points: int = Field(50, description="Mínimo de pontos para treinar")
    retrain: bool = Field(False, description="Forçar retreinamento")

# =============================================================================
# CLASSE GRACE DATA (mantida do código original)
# =============================================================================
class GraceData:
    """Classe para processamento de dados GRACE/GRACE-FO"""

    @staticmethod
    def get_grace_data_real(lon: float, lat: float, start_date: str, end_date: str, analysis_type: str):
        """Obtém dados GRACE reais do Earth Engine"""
        if not EE_WORKING:
            return {
                "analysis_type": analysis_type,
                "time_series": [],
                "current_anomaly": 0,
                "trend": 0,
                "max_variation": 0,
                "confidence": 0,
                "interpretation": "Earth Engine não disponível",
                "units": "cm (equivalente em altura de água)",
                "n_observations": 0
            }
        
        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(200000)  # 200 km

            print(f"🔍 Buscando dados GRACE para: {lon}, {lat} entre {start_date} e {end_date}")
            
            grace_collections = {
                "water_mass": [
                    "NASA/GRACE/MASS_GRIDS_V04/MASCON",  
                    "NASA/GRACE-FO/MASS_GRIDS_V04/MASCON"
                ],
                "groundwater": [
                    "NASA/GRACE/MASS_GRIDS_V04/MASCON",
                    "NASA/GRACE-FO/MASS_GRIDS_V04/MASCON"
                ],
                "soil_moisture": [
                    "NASA/GRACE/MASS_GRIDS_V04/MASCON",
                    "NASA/GRACE-FO/MASS_GRIDS_V04/MASCON"
                ]
            }
            
            band_mapping = {
                "water_mass": ["lwe_thickness", "mass_anomaly", "uncertainty"],
                "groundwater": ["lwe_thickness", "mass_anomaly", "uncertainty"],
                "soil_moisture": ["lwe_thickness", "mass_anomaly", "uncertainty"]
            }
            
            grace_collection = None
            dataset_used = None
            band_used = None
            
            for dataset_id in grace_collections.get(analysis_type, []):
                try:
                    collection = ee.ImageCollection(dataset_id) \
                        .filterBounds(region) \
                        .filterDate(start_date, end_date)
                    
                    if collection.size().getInfo() > 0:
                        grace_collection = collection
                        dataset_used = dataset_id
                        print(f"✅ Encontrada coleção GRACE: {dataset_id} com {collection.size().getInfo()} imagens")
                        break
                except Exception as e:
                    print(f"❌ Erro ao acessar coleção {dataset_id}: {e}")
                    continue
            
            if not grace_collection:
                print("❌ Nenhuma coleção GRACE encontrada com dados para o período e localização")
                return {
                    "analysis_type": analysis_type,
                    "time_series": [],
                    "current_anomaly": 0,
                    "trend": 0,
                    "max_variation": 0,
                    "confidence": 0,
                    "interpretation": "Nenhum dado GRACE encontrado para o período e localização especificados",
                    "units": "cm (equivalente em altura de água)",
                    "n_observations": 0
                }
            
            first_image = grace_collection.first()
            band_names = first_image.bandNames().getInfo()
            print("📌 Bandas disponíveis:", band_names)

            for band_name in band_mapping.get(analysis_type, []):
                if band_name in band_names:
                    band_used = band_name
                    print(f"✅ Banda encontrada: {band_name}")
                    break
            
            if not band_used:
                print("❌ Nenhuma banda adequada encontrada")
                return {
                    "analysis_type": analysis_type,
                    "time_series": [],
                    "current_anomaly": 0,
                    "trend": 0,
                    "max_variation": 0,
                    "confidence": 0,
                    "interpretation": "Nenhuma banda adequada encontrada nos dados GRACE",
                    "units": "cm (equivalente em altura de água)",
                    "n_observations": 0
                }
            
            def extract_grace_values(image):
                date = ee.Date(image.get('system:time_start')).format('YYYY-MM')
                anomaly = image.select(band_used).reduceRegion(
                    reducer=ee.Reducer.mean(), 
                    geometry=region, 
                    scale=150000,
                    maxPixels=1e9
                ).get(band_used)
                
                return ee.Feature(None, {
                    'date': date, 
                    'anomaly': anomaly, 
                    'analysis_type': analysis_type,
                    'dataset': dataset_used,
                    'band': band_used
                })
            
            features = grace_collection.map(extract_grace_values)
            data = features.getInfo()
            
            records = []
            for feature in data['features']:
                props = feature['properties']
                if props['anomaly'] is not None:
                    records.append({
                        'date': props['date'],
                        'anomaly': float(props['anomaly']),
                        'analysis_type': props['analysis_type'],
                        'sensor': 'GRACE',
                        'source': 'earth_engine',
                        'dataset': props.get('dataset', 'unknown'),
                        'band': props.get('band', 'unknown')
                    })
            
            print(f"✅ GRACE real: {len(records)} registros")
            return GraceData._process_grace_records(records, analysis_type)
            
        except Exception as e:
            print(f"❌ Erro GRACE real: {e}")
            return {
                "analysis_type": analysis_type,
                "time_series": [],
                "current_anomaly": 0,
                "trend": 0,
                "max_variation": 0,
                "confidence": 0,
                "interpretation": f"Erro ao processar dados GRACE: {str(e)}",
                "units": "cm (equivalente em altura de água)",
                "n_observations": 0
            }

    @staticmethod
    def _process_grace_records(records: List[Dict], analysis_type: str):
        """Processa e analisa os registros GRACE"""
        if not records:
            return {
                "analysis_type": analysis_type,
                "time_series": [],
                "current_anomaly": 0,
                "trend": 0,
                "max_variation": 0,
                "confidence": 0.7,
                "interpretation": "Dados insuficientes",
                "units": "cm (equivalente em altura de água)",
                "n_observations": 0
            }
        
        anomalies = [r['anomaly'] for r in records]
        dates = [r['date'] for r in records]
        
        current_anomaly = anomalies[-1] if anomalies else 0
        max_variation = max(anomalies) - min(anomalies) if anomalies else 0
        
        if len(anomalies) > 1:
            x = np.arange(len(anomalies))
            trend = np.polyfit(x, anomalies, 1)[0] * 12
        else:
            trend = 0.0
        
        if current_anomaly > 15:
            interpretation = "🚨 Alta anomalia positiva - Possível aumento significativo na massa de água"
        elif current_anomaly > 8:
            interpretation = "⚠️ Anomalia positiva - Aumento moderado na massa de água"
        elif current_anomaly < -15:
            interpretation = "🚨 Alta anomalia negativa - Possível déficit hídrico severo"
        elif current_anomaly < -8:
            interpretation = "⚠️ Anomalia negativa - Déficit hídrico moderado"
        else:
            interpretation = "✅ Condições hídricas dentro da normalidade"
        
        if abs(trend) > 2:
            trend_direction = "aumentando" if trend > 0 else "diminuindo"
            interpretation += f". Tendência: {abs(trend):.1f} cm/ano ({trend_direction})"
        
        dataset_info = records[0].get('dataset', 'unknown') if records else 'unknown'
        band_info = records[0].get('band', 'unknown') if records else 'unknown'
        
        return {
            "analysis_type": analysis_type,
            "time_series": records,
            "current_anomaly": round(current_anomaly, 2),
            "trend": round(trend, 2),
            "max_variation": round(max_variation, 2),
            "confidence": min(0.95, 0.7 + len(records) * 0.01),
            "interpretation": interpretation,
            "units": "cm (equivalente em altura de água)",
            "n_observations": len(records),
            "dataset_used": dataset_info,
            "band_used": band_info
        }

# =============================================================================
# FUNÇÕES PARA CADA SENSOR (mantidas do código original)
# =============================================================================
def get_landsat_data(lon: float, lat: float, start_date: str, end_date: str, buffer_km: float):
    """Obtém dados Landsat 8/9 para NDVI"""
    if not EE_WORKING:
        return []

    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(region).filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 50))
        
        l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
            .filterBounds(region).filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 50))
        
        collection = l8.merge(l9)
        
        if collection.size().getInfo() == 0:
            return []
        
        def calculate_ndvi(image):
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        collection_with_ndvi = collection.map(calculate_ndvi)
        
        def extract_values(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            mean_ndvi = image.select('NDVI').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=100
            ).get('NDVI')
            
            return ee.Feature(None, {
                'date': date, 'NDVI': mean_ndvi, 'cloud_cover': image.get('CLOUD_COVER'),
                'sensor': 'Landsat'
            })
        
        features = collection_with_ndvi.map(extract_values)
        data = features.getInfo()
        
        records = []
        for feature in data['features']:
            props = feature['properties']
            if props['NDVI'] is not None:
                records.append({
                    'date': props['date'], 'NDVI': float(props['NDVI']),
                    'cloud_cover': props.get('cloud_cover', 0), 'sensor': 'Landsat'
                })
        
        print(f"✅ Landsat: {len(records)} registros")
        return records
        
    except Exception as e:
        print(f"❌ Erro Landsat: {e}")
        return []

def get_modis_data(lon: float, lat: float, start_date: str, end_date: str, buffer_km: float):
    """Obtém dados MODIS para clorofila e temperatura da superfície do mar"""
    if not EE_WORKING:
        return []

    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        # Coleções MODIS disponíveis - tentar diferentes opções
        modis_collections = [
            'MODIS/061/MOD09GA',  # Superfície terrestre
            'MODIS/061/MYD09GA',  # Aqua - Superfície terrestre
            'MODIS/006/MOD13Q1',  # NDVI/EVI
            'MODIS/006/MYD13Q1',  # Aqua - NDVI/EVI
            'MODIS/006/MOD17A3HGF',  # Produtividade primária
            'NASA/OCEANDATA/MODIS-Aqua/L3SMI'  # Clorofila - Oceano
        ]
        
        collection = None
        collection_name = None
        
        for coll_name in modis_collections:
            try:
                test_coll = ee.ImageCollection(coll_name) \
                    .filterBounds(region) \
                    .filterDate(start_date, end_date)
                
                if test_coll.size().getInfo() > 0:
                    collection = test_coll
                    collection_name = coll_name
                    print(f"✅ MODIS: Coleção encontrada - {coll_name}")
                    break
            except Exception as e:
                print(f"❌ MODIS: Erro na coleção {coll_name}: {e}")
                continue
        
        if not collection:
            print("❌ MODIS: Nenhuma coleção válida encontrada")
            return []
        
        def extract_modis_values(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            # Bandas comuns do MODIS
            properties = {'date': date, 'sensor': 'MODIS', 'collection': collection_name}
            
            # Tentar diferentes bandas dependendo da coleção
            if 'chlor_a' in image.bandNames().getInfo():
                chlorophyll = image.select('chlor_a').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=1000
                ).get('chlor_a')
                properties['chlorophyll'] = chlorophyll
            
            if 'sur_refl_b02' in image.bandNames().getInfo():
                # Calcular NDVI aproximado: (NIR - Red) / (NIR + Red)
                # MODIS bandas: b02 (Red), b01 (NIR)
                ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01']).reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=1000
                ).get('nd')
                properties['NDVI'] = ndvi
            
            if 'LST_Day_1km' in image.bandNames().getInfo():
                lst = image.select('LST_Day_1km').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=1000
                ).get('LST_Day_1km')
                properties['land_surface_temp'] = lst
            
            if 'EVI' in image.bandNames().getInfo():
                evi = image.select('EVI').reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=region, scale=1000
                ).get('EVI')
                properties['EVI'] = evi
            
            return ee.Feature(None, properties)
        
        features = collection.map(extract_modis_values)
        data = features.getInfo()
        
        records = []
        for feature in data['features']:
            props = feature['properties']
            record = {'date': props['date'], 'sensor': 'MODIS'}
            
            # Adicionar dados disponíveis
            if props.get('chlorophyll') is not None:
                record['chlorophyll'] = float(props['chlorophyll'])
            if props.get('NDVI') is not None:
                record['NDVI'] = float(props['NDVI'])
            if props.get('land_surface_temp') is not None:
                record['land_surface_temp'] = float(props['land_surface_temp'])
            if props.get('EVI') is not None:
                record['EVI'] = float(props['EVI'])
            
            # Só adicionar se tiver pelo menos um dado válido
            if len(record) > 2:  # Mais que date e sensor
                records.append(record)
        
        print(f"✅ MODIS: {len(records)} registros da coleção {collection_name}")
        return records
        
    except Exception as e:
        print(f"❌ Erro MODIS: {e}")
        return []

def get_viirs_data(lon: float, lat: float, start_date: str, end_date: str, buffer_km: float):
    """Obtém dados VIIRS para detecção noturna e clorofila"""
    if not EE_WORKING:
        return []

    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        viirs = ee.ImageCollection('NASA/VIIRS/002/VNP09GA') \
            .filterBounds(region).filterDate(start_date, end_date)
        
        if viirs.size().getInfo() == 0:
            return []
        
        def extract_viirs_values(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            ndvi = image.normalizedDifference(['I2', 'I1']).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=500
            ).get('nd')
            
            return ee.Feature(None, {
                'date': date, 'NDVI': ndvi, 'sensor': 'VIIRS'
            })
        
        features = viirs.map(extract_viirs_values)
        data = features.getInfo()
        
        records = []
        for feature in data['features']:
            props = feature['properties']
            if props['NDVI'] is not None:
                records.append({
                    'date': props['date'], 'NDVI': float(props['NDVI']),
                    'sensor': 'VIIRS'
                })
        
        print(f"✅ VIIRS: {len(records)} registros")
        return records
        
    except Exception as e:
        print(f"❌ Erro VIIRS: {e}")
        return []

def get_smap_data(lon: float, lat: float, start_date: str, end_date: str, buffer_km: float):
    """Obtém dados SMAP para umidade do solo"""
    if not EE_WORKING:
        return []

    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        smap = ee.ImageCollection('NASA/SMAP/SPL3SMP_E/005') \
            .filterBounds(region).filterDate(start_date, end_date)
        
        if smap.size().getInfo() == 0:
            return []
        
        def extract_smap_values(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            soil_moisture = image.select('soil_moisture_am').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=30000
            ).get('soil_moisture_am')
            
            return ee.Feature(None, {
                'date': date, 'soil_moisture': soil_moisture, 'sensor': 'SMAP'
            })
        
        features = smap.map(extract_smap_values)
        data = features.getInfo()
        
        records = []
        for feature in data['features']:
            props = feature['properties']
            if props['soil_moisture'] is not None:
                records.append({
                    'date': props['date'], 
                    'soil_moisture': float(props['soil_moisture']),
                    'sensor': 'SMAP'
                })
        
        print(f"✅ SMAP: {len(records)} registros")
        return records
        
    except Exception as e:
        print(f"❌ Erro SMAP: {e}")
        return []

def get_grace_data(lon: float, lat: float, start_date: str, end_date: str, analysis_type: str):
    """Obtém dados GRACE/GRACE-FO"""
    return GraceData.get_grace_data_real(lon, lat, start_date, end_date, analysis_type)

def get_ocean_data(lon: float, lat: float, start_date: str, end_date: str, buffer_km: float):
    """Obtém dados específicos para oceanos (clorofila, SST)"""
    if not EE_WORKING:
        return []

    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        # Coleções para dados oceânicos
        ocean_collections = [
            'NASA/OCEANDATA/MODIS-Aqua/L3SMI',  # Dados oceânicos do Aqua
            'NASA/OCEANDATA/MODIS-Terra/L3SMI', # Dados oceânicos do Terra
        ]
        
        collection = None
        for coll_name in ocean_collections:
            try:
                test_coll = ee.ImageCollection(coll_name) \
                    .filterBounds(region) \
                    .filterDate(start_date, end_date)
                
                if test_coll.size().getInfo() > 0:
                    collection = test_coll
                    print(f"✅ Dados Oceânicos: Coleção encontrada - {coll_name}")
                    break
            except Exception as e:
                print(f"❌ Erro na coleção oceânica {coll_name}: {e}")
                continue
        
        if not collection:
            print("❌ Dados Oceânicos: Nenhuma coleção válida encontrada")
            return []
        
        def extract_ocean_values(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            # Bandas oceânicas
            chlor_a = image.select('chlor_a').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=4000
            ).get('chlor_a')
            
            sst = image.select('sst').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=4000
            ).get('sst')
            
            return ee.Feature(None, {
                'date': date, 
                'chlorophyll': chlor_a, 
                'sst': sst,
                'sensor': 'MODIS_Ocean'
            })
        
        features = collection.map(extract_ocean_values)
        data = features.getInfo()
        
        records = []
        for feature in data['features']:
            props = feature['properties']
            if props.get('chlorophyll') is not None:
                records.append({
                    'date': props['date'], 
                    'chlorophyll': float(props['chlorophyll']),
                    'sst': float(props['sst']) if props.get('sst') else None,
                    'sensor': 'MODIS_Ocean'
                })
        
        print(f"✅ Dados Oceânicos: {len(records)} registros")
        return records
        
    except Exception as e:
        print(f"❌ Erro dados oceânicos: {e}")
        return []

# =============================================================================
# FUNÇÕES DE DETECÇÃO DE EVENTOS (mantidas do código original)
# =============================================================================
def detect_ndvi_events(data, sensor_name):
    """Detecta eventos baseados em NDVI"""
    if len(data) < 3:
        return []

    data_sorted = sorted(data, key=lambda x: x['date'])
    ndvi_values = [d['NDVI'] for d in data_sorted if 'NDVI' in d]

    if not ndvi_values:
        return []

    mean_ndvi = np.mean(ndvi_values)
    std_ndvi = np.std(ndvi_values)
    threshold = mean_ndvi + 1.5 * std_ndvi

    events = []
    in_event = False
    event_start = None
    event_peak = None
    event_max = 0

    for i, point in enumerate(data_sorted):
        if 'NDVI' not in point:
            continue
            
        ndvi = point['NDVI']
        
        if ndvi > threshold and not in_event:
            in_event = True
            event_start = point['date']
            event_peak = point['date']
            event_max = ndvi
        elif in_event and ndvi > event_max:
            event_peak = point['date']
            event_max = ndvi
        elif in_event and ndvi < threshold:
            duration_days = (datetime.strptime(point['date'], '%Y-%m-%d') - 
                           datetime.strptime(event_start, '%Y-%m-%d')).days
            
            events.append({
                'sensor': sensor_name,
                'type': 'high_vegetation',
                'start': event_start,
                'peak': event_peak,
                'end': point['date'],
                'duration_days': duration_days,
                'max_value': round(event_max, 3),
                'severity': 'high' if event_max > mean_ndvi + 2 * std_ndvi else 'medium',
                'confidence': min(0.95, 0.7 + duration_days * 0.01)
            })
            
            in_event = False

    if in_event and event_start:
        duration_days = (datetime.strptime(data_sorted[-1]['date'], '%Y-%m-%d') - 
                       datetime.strptime(event_start, '%Y-%m-%d')).days
        
        events.append({
            'sensor': sensor_name,
            'type': 'high_vegetation_ongoing',
            'start': event_start,
            'peak': event_peak,
            'end': data_sorted[-1]['date'],
            'duration_days': duration_days,
            'max_value': round(event_max, 3),
            'severity': 'high' if event_max > mean_ndvi + 2 * std_ndvi else 'medium',
            'confidence': min(0.9, 0.6 + duration_days * 0.01),
            'status': 'ongoing'
        })

    return events

def detect_chlorophyll_events(data):
    """Detecta eventos de floração baseados em clorofila"""
    if len(data) < 3:
        return []

    data_sorted = sorted(data, key=lambda x: x['date'])
    
    # Extrair valores de clorofila (pode vir de MODIS ou MODIS_Ocean)
    chlorophyll_values = []
    for d in data_sorted:
        if 'chlorophyll' in d and d['chlorophyll'] is not None:
            chlorophyll_values.append(d['chlorophyll'])
        elif 'chlor_a' in d and d['chlor_a'] is not None:
            chlorophyll_values.append(d['chlor_a'])

    if not chlorophyll_values:
        return []

    mean_chlorophyll = np.mean(chlorophyll_values)
    std_chlorophyll = np.std(chlorophyll_values)
    threshold = mean_chlorophyll + 2 * std_chlorophyll

    events = []
    in_event = False
    event_start = None
    event_peak = None
    event_max = 0

    for i, point in enumerate(data_sorted):
        chlorophyll = None
        if 'chlorophyll' in point and point['chlorophyll'] is not None:
            chlorophyll = point['chlorophyll']
        elif 'chlor_a' in point and point['chlor_a'] is not None:
            chlorophyll = point['chlor_a']
        
        if chlorophyll is None:
            continue
            
        if chlorophyll > threshold and not in_event:
            in_event = True
            event_start = point['date']
            event_peak = point['date']
            event_max = chlorophyll
        elif in_event and chlorophyll > event_max:
            event_peak = point['date']
            event_max = chlorophyll
        elif in_event and chlorophyll < mean_chlorophyll + std_chlorophyll:
            duration_days = (datetime.strptime(point['date'], '%Y-%m-%d') - 
                           datetime.strptime(event_start, '%Y-%m-%d')).days
            
            # Classifica a severidade do bloom
            if event_max > mean_chlorophyll + 3 * std_chlorophyll:
                severity = 'severe'
                bloom_type = 'harmful_algal_bloom'
            elif event_max > mean_chlorophyll + 2 * std_chlorophyll:
                severity = 'high'
                bloom_type = 'algal_bloom'
            else:
                severity = 'medium'
                bloom_type = 'phytoplankton_increase'
            
            events.append({
                'sensor': point.get('sensor', 'MODIS'),
                'type': bloom_type,
                'start': event_start,
                'peak': event_peak,
                'end': point['date'],
                'duration_days': duration_days,
                'max_chlorophyll': round(event_max, 2),
                'severity': severity,
                'confidence': min(0.95, 0.7 + duration_days * 0.02)
            })
            
            in_event = False

    return events

def detect_moisture_events(data):
    """Detecta eventos de umidade do solo"""
    if len(data) < 3:
        return []

    data_sorted = sorted(data, key=lambda x: x['date'])
    moisture_values = [d['soil_moisture'] for d in data_sorted if 'soil_moisture' in d]

    if not moisture_values:
        return []

    mean_moisture = np.mean(moisture_values)
    std_moisture = np.std(moisture_values)
    high_threshold = mean_moisture + 1.5 * std_moisture
    low_threshold = mean_moisture - 1.5 * std_moisture

    events = []
    in_high_event = False
    in_low_event = False
    event_start = None
    event_peak = None
    event_extreme = 0

    for i, point in enumerate(data_sorted):
        if 'soil_moisture' not in point:
            continue
            
        moisture = point['soil_moisture']
        
        if moisture > high_threshold and not in_high_event:
            in_high_event = True
            event_start = point['date']
            event_peak = point['date']
            event_extreme = moisture
        elif in_high_event and moisture > event_extreme:
            event_peak = point['date']
            event_extreme = moisture
        elif in_high_event and moisture < mean_moisture + std_moisture:
            duration_days = (datetime.strptime(point['date'], '%Y-%m-%d') - 
                           datetime.strptime(event_start, '%Y-%m-%d')).days
            
            events.append({
                'sensor': 'SMAP',
                'type': 'high_soil_moisture',
                'start': event_start,
                'peak': event_peak,
                'end': point['date'],
                'duration_days': duration_days,
                'max_moisture': round(event_extreme, 3),
                'severity': 'high' if event_extreme > mean_moisture + 2 * std_moisture else 'medium',
                'confidence': min(0.95, 0.6 + duration_days * 0.01)
            })
            
            in_high_event = False
        
        if moisture < low_threshold and not in_low_event:
            in_low_event = True
            event_start = point['date']
            event_peak = point['date']
            event_extreme = moisture
        elif in_low_event and moisture < event_extreme:
            event_peak = point['date']
            event_extreme = moisture
        elif in_low_event and moisture > mean_moisture - std_moisture:
            duration_days = (datetime.strptime(point['date'], '%Y-%m-%d') - 
                           datetime.strptime(event_start, '%Y-%m-%d')).days
            
            events.append({
                'sensor': 'SMAP',
                'type': 'low_soil_moisture',
                'start': event_start,
                'peak': event_peak,
                'end': point['date'],
                'duration_days': duration_days,
                'min_moisture': round(event_extreme, 3),
                'severity': 'high' if event_extreme < mean_moisture - 2 * std_moisture else 'medium',
                'confidence': min(0.95, 0.6 + duration_days * 0.01)
            })
            
            in_low_event = False

    if in_high_event:
        duration_days = (datetime.strptime(data_sorted[-1]['date'], '%Y-%m-%d') - 
                       datetime.strptime(event_start, '%Y-%m-%d')).days
        
        events.append({
            'sensor': 'SMAP',
            'type': 'high_soil_moisture',
            'start': event_start,
            'peak': event_peak,
            'end': data_sorted[-1]['date'],
            'duration_days': duration_days,
            'max_moisture': round(event_extreme, 3),
            'severity': 'high' if event_extreme > mean_moisture + 2 * std_moisture else 'medium',
            'confidence': min(0.95, 0.6 + duration_days * 0.01),
            'ongoing': True
        })

    if in_low_event:
        duration_days = (datetime.strptime(data_sorted[-1]['date'], '%Y-%m-%d') - 
                       datetime.strptime(event_start, '%Y-%m-%d')).days
        
        events.append({
            'sensor': 'SMAP',
            'type': 'low_soil_moisture',
            'start': event_start,
            'peak': event_peak,
            'end': data_sorted[-1]['date'],
            'duration_days': duration_days,
            'min_moisture': round(event_extreme, 3),
            'severity': 'high' if event_extreme < mean_moisture - 2 * std_moisture else 'medium',
            'confidence': min(0.95, 0.6 + duration_days * 0.01),
            'ongoing': True
        })

    return events

# =============================================================================
# ENDPOINTS ATUALIZADOS COM ML
# =============================================================================
@app.post("/analyze")
async def analyze_with_ml(request: AnalyzeRequest):
    """Endpoint principal para análise multi-sensor com Machine Learning"""
    results = {}
    all_events = []

    # Coletar dados dos sensores
    for sensor in request.sensors:
        if sensor not in ['landsat', 'modis', 'viirs', 'smap', 'grace']:
            continue
        
        try:
            if sensor == 'landsat':
                data = get_landsat_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                results['landsat'] = data
                if data:
                    events = detect_ndvi_events(data, 'Landsat')
                    all_events.extend(events)
                
            elif sensor == 'modis':
                data = get_modis_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                # Adicionar dados oceânicos específicos
                ocean_data = get_ocean_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                if ocean_data:
                    data.extend(ocean_data)
                results['modis'] = data
                if data:
                    events = detect_chlorophyll_events(data)
                    all_events.extend(events)
                
            elif sensor == 'viirs':
                data = get_viirs_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                results['viirs'] = data
                if data:
                    events = detect_ndvi_events(data, 'VIIRS')
                    all_events.extend(events)
                
            elif sensor == 'smap':
                data = get_smap_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                results['smap'] = data
                if data:
                    events = detect_moisture_events(data)
                    all_events.extend(events)
                
            elif sensor == 'grace':
                grace_data = get_grace_data(request.lon, request.lat, request.start, request.end, request.grace_analysis)
                results['grace'] = grace_data
                
        except Exception as e:
            print(f"❌ Erro no sensor {sensor}: {e}")
            results[sensor] = {"error": str(e)}

    # Consolidar eventos
    consolidated_events = []
    if all_events:
        event_df = pd.DataFrame(all_events)
        if not event_df.empty:
            event_df['start_date'] = pd.to_datetime(event_df['start'])
            event_df = event_df.sort_values('start_date')
            
            for _, row in event_df.iterrows():
                consolidated_events.append(row.to_dict())

    # Preparar resposta base
    response = {
        "location": {
            "lon": request.lon,
            "lat": request.lat,
            "buffer_km": request.buffer_km
        },
        "period": {
            "start": request.start,
            "end": request.end
        },
        "sensors_used": request.sensors,
        "earth_engine_status": {
            "working": EE_WORKING,
            "message": EE_MESSAGE
        },
        "data": results,
        "events": consolidated_events,
        "summary": {
            "total_events": len(consolidated_events),
            "event_types": list(set([e['type'] for e in consolidated_events])),
            "sensors_with_events": list(set([e['sensor'] for e in consolidated_events]))
        }
    }

    # Adicionar dados para treinamento do ML
    ml_model.add_training_data(response)
    
    # Salvar no DataManager se disponível
    if data_manager:
        try:
            data_manager.add_analysis_data(response)
        except Exception as e:
            print(f"❌ Erro ao salvar dados no DataManager: {e}")

    # Adicionar predições de ML se solicitado
    if request.include_ml_predictions and ml_model.is_trained:
        ml_predictions = ml_model.predict_bloom_risk(results)
        response["ml_predictions"] = ml_predictions
        response["ml_model_status"] = ml_model.get_model_status()
    elif request.include_ml_predictions:
        response["ml_predictions"] = {
            "status": "model_not_trained",
            "message": "Modelo precisa ser treinado com dados históricos. Use o endpoint /train-models."
        }
        response["ml_model_status"] = ml_model.get_model_status()

    return response

@app.post("/train-models")
async def train_models(request: TrainRequest = TrainRequest()):
    """Endpoint para treinar os modelos de Machine Learning"""
    try:
        if len(ml_model.training_data) < request.min_data_points:
            return {
                "status": "insufficient_data",
                "message": f"Dados insuficientes para treinamento. Necessários: {request.min_data_points}, Disponíveis: {len(ml_model.training_data)}",
                "available_data_points": len(ml_model.training_data),
                "required_data_points": request.min_data_points
            }
        
        print(f"🔧 Iniciando treinamento com {len(ml_model.training_data)} pontos de dados...")
        
        # Preparar dados para treinamento
        X, y = ml_model.prepare_training_data(ml_model.training_data)
        
        if X is None or y is None:
            return {
                "status": "error",
                "message": "Não foi possível preparar dados para treinamento"
            }
        
        print(f"📊 Dados preparados: {X.shape[0]} amostras, {X.shape[1]} features")
        print(f"🎯 Distribuição de classes: {np.sum(y)} eventos positivos de {len(y)} amostras")
        
        # Treinar modelos
        classification_accuracy = ml_model.train_classification_model(X, y)
        regression_mse = ml_model.train_regression_model(X, y)
        
        result = {
            "status": "success",
            "trained_at": datetime.now().isoformat(),
            "training_data_points": len(ml_model.training_data),
            "training_samples": X.shape[0],
            "features_count": X.shape[1],
            "positive_events": int(np.sum(y)),
            "classification_accuracy": classification_accuracy,
            "regression_mse": regression_mse,
            "model_metadata": ml_model.model_metadata
        }
        
        # Salvar modelos automaticamente
        if ml_model.is_trained:
            ml_model.save_models("bloom_detection_model")
            result["models_saved"] = True
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro durante o treinamento: {str(e)}",
            "error_details": str(e)
        }

@app.get("/ml-status")
async def get_ml_status():
    """Endpoint para verificar status do modelo de ML"""
    return ml_model.get_model_status()

@app.post("/ml-predict")
async def ml_predict(request: AnalyzeRequest):
    """Endpoint específico para predições de ML"""
    try:
        # Coletar dados básicos primeiro
        results = {}
        for sensor in request.sensors:
            if sensor == 'landsat':
                data = get_landsat_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                results['landsat'] = data
            elif sensor == 'modis':
                data = get_modis_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                results['modis'] = data
            elif sensor == 'viirs':
                data = get_viirs_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                results['viirs'] = data
            elif sensor == 'smap':
                data = get_smap_data(request.lon, request.lat, request.start, request.end, request.buffer_km)
                results['smap'] = data
        
        prediction = ml_model.predict_bloom_risk(results)
        return {
            "location": {
                "lon": request.lon,
                "lat": request.lat,
                "buffer_km": request.buffer_km
            },
            "prediction": prediction,
            "model_status": ml_model.get_model_status()
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# ENDPOINTS ORIGINAIS (mantidos para compatibilidade)
# =============================================================================
@app.get("/health")
async def health_check():
    """Endpoint para verificação de saúde do sistema"""
    return {
        "status": "healthy",
        "earth_engine": {
            "available": EE_AVAILABLE,
            "working": EE_WORKING,
            "message": EE_MESSAGE
        },
        "ml_model": ml_model.get_model_status(),
        "sensors": ['landsat', 'modis', 'viirs', 'smap', 'grace'],
        "grace_analysis_types": ["water_mass", "groundwater", "soil_moisture"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/grace")
async def debug_grace():
    """Endpoint para depurar dados GRACE"""
    if not EE_WORKING:
        return {"error": "Earth Engine não disponível"}

    try:
        grace_collections = [
            "NASA/GRACE/MASS_GRIDS/LAND",
            "NASA/GRACE/MASS_GRIDS/MASCON",
            "NASA/GRACE-FO/MASS_GRIDS/LAND",
            "NASA/GRACE-FO/MASS_GRIDS/MASCON"
        ]
        
        result = {"collections": {}}
        
        for collection_id in grace_collections:
            try:
                collection = ee.ImageCollection(collection_id)
                info = {
                    "id": collection_id,
                    "size": collection.size().getInfo(),
                    "bands": []
                }
                
                first_image = collection.first()
                if first_image:
                    band_info = first_image.bandNames().getInfo()
                    info["bands"] = band_info
                
                result["collections"][collection_id] = info
            except Exception as e:
                result["collections"][collection_id] = {"error": str(e)}
        
        return result
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# ENDPOINTS PARA GERENCIAMENTO DE DADOS
# =============================================================================
@app.get("/data/stats")
async def get_data_stats():
    """Endpoint para obter estatísticas dos dados coletados"""
    if not data_manager:
        raise HTTPException(status_code=503, detail="DataManager não disponível")
    
    try:
        return data_manager.get_data_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter estatísticas: {str(e)}")

@app.get("/data/export")
async def export_data():
    """Endpoint para exportar dados coletados"""
    if not data_manager:
        raise HTTPException(status_code=503, detail="DataManager não disponível")
    
    try:
        filename = f"bloom_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        success = data_manager.export_to_csv(filename)
        
        if success and os.path.exists(filename):
            return FileResponse(
                filename,
                media_type='text/csv', 
                filename=filename
            )
        else:
            raise HTTPException(status_code=500, detail="Erro ao exportar dados")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao exportar dados: {str(e)}")

@app.delete("/data/clear")
async def clear_data(confirm: bool = False):
    """Endpoint para limpar dados (requer confirmação)"""
    if not data_manager:
        raise HTTPException(status_code=503, detail="DataManager não disponível")
    
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Confirmação necessária. Use confirm=true para limpar dados."
        )
    
    try:
        success = data_manager.clear_data()
        if success:
            # Também limpa os dados do ML model
            ml_model.training_data = []
            return {"status": "success", "message": "Todos os dados foram limpos"}
        else:
            raise HTTPException(status_code=500, detail="Erro ao limpar dados")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao limpar dados: {str(e)}")

@app.get("/data/list")
async def list_analyses(limit: int = 10, offset: int = 0):
    """Endpoint para listar análises salvas"""
    if not data_manager:
        raise HTTPException(status_code=503, detail="DataManager não disponível")
    
    try:
        all_data = data_manager.get_training_data()
        total = len(all_data)
        
        analyses = all_data[offset:offset + limit]
        
        return {
            "total_analyses": total,
            "showing": f"{offset + 1}-{min(offset + limit, total)} of {total}",
            "analyses": [
                {
                    "id": analysis.get('_id', ''),
                    "timestamp": analysis.get('_timestamp', ''),
                    "location": analysis.get('location', {}),
                    "period": analysis.get('period', {}),
                    "sensors": analysis.get('sensors_used', []),
                    "events_count": len(analysis.get('events', [])),
                    "has_ml_predictions": 'ml_predictions' in analysis
                }
                for analysis in analyses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar análises: {str(e)}")

# =============================================================================
# INICIALIZAÇÃO
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando API de Detecção de Florações com Machine Learning")
    print("📊 Endpoints disponíveis:")
    print("   POST /analyze - Análise multi-sensor com ML")
    print("   POST /train-models - Treinar modelos de ML")
    print("   GET /ml-status - Status do modelo")
    print("   POST /ml-predict - Predições específicas de ML")
    print("   GET /health - Saúde do sistema")
    print("   GET /data/stats - Estatísticas dos dados")
    print("   GET /data/export - Exportar dados")
    print("   DELETE /data/clear - Limpar dados")
    print("   GET /data/list - Listar análises")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)