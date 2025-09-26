"""
Bloom Detection Backend - Earth Engine REAL (Multi-Sensor) com GRACE
"""

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
    title="Bloom Detection API - Multi-Sensor com GRACE",
    description="API para detecção de florações com dados Landsat, SMAP, VIIRS, MODIS e GRACE",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# =============================================================================
# CLASSE GRACE DATA
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
            # Aumentar o buffer para GRACE (resolução mais baixa)
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
            
            # Bandas por análise
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
            
            # Verificação correta das bandas
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
                    scale=150000,  # Escala ajustada (150 km)
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
            # Correção: Chamada correta ao método _process_grace_records
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
        
        # Calcula estatísticas
        current_anomaly = anomalies[-1] if anomalies else 0
        max_variation = max(anomalies) - min(anomalies) if anomalies else 0
        
        # Calcula tendência linear
        if len(anomalies) > 1:
            x = np.arange(len(anomalies))
            trend = np.polyfit(x, anomalies, 1)[0] * 12  # cm/ano
        else:
            trend = 0.0
        
        # Interpretação baseada nos dados
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
        
        # Adiciona informação da tendência
        if abs(trend) > 2:
            trend_direction = "aumentando" if trend > 0 else "diminuindo"
            interpretation += f". Tendência: {abs(trend):.1f} cm/ano ({trend_direction})"
        
        # Adiciona informações sobre o dataset e banda usados
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
# FUNÇÕES PARA CADA SENSOR
# =============================================================================

def get_landsat_data(lon: float, lat: float, start_date: str, end_date: str, buffer_km: float):
    """Obtém dados Landsat 8/9 para NDVI"""
    if not EE_WORKING:
        return []
    
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        # Landsat 8 e 9 combinados
        l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(region).filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 50))
        
        l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
            .filterBounds(region).filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 50))
        
        collection = l8.merge(l9)
        
        if collection.size().getInfo() == 0:
            return []
        
        # NDVI para Landsat: (B5 - B4) / (B5 + B4)
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
    """Obtém dados MODIS para NDVI e clorofila"""
    if not EE_WORKING:
        return []
    
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        # MODIS Aqua - para detecção de florações de algas
        modis = ee.ImageCollection('MODIS/Aqua/L3SMI') \
            .filterBounds(region).filterDate(start_date, end_date)
        
        if modis.size().getInfo() == 0:
            return []
        
        def extract_modis_values(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            # Parâmetros para florações: clorofila, SST, etc.
            chlorophyll = image.select('chlor_a').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=4000
            ).get('chlor_a')
            
            sst = image.select('sst').reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=4000
            ).get('sst')
            
            return ee.Feature(None, {
                'date': date, 'chlorophyll': chlorophyll, 'sst': sst,
                'sensor': 'MODIS'
            })
        
        features = modis.map(extract_modis_values)
        data = features.getInfo()
        
        records = []
        for feature in data['features']:
            props = feature['properties']
            if props['chlorophyll'] is not None:
                records.append({
                    'date': props['date'], 
                    'chlorophyll': float(props['chlorophyll']),
                    'sst': float(props['sst']) if props['sst'] else None,
                    'sensor': 'MODIS'
                })
        
        print(f"✅ MODIS: {len(records)} registros")
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
        
        # VIIRS - útil para detecção noturna de blooms
        viirs = ee.ImageCollection('NASA/VIIRS/002/VNP09GA') \
            .filterBounds(region).filterDate(start_date, end_date)
        
        if viirs.size().getInfo() == 0:
            return []
        
        def extract_viirs_values(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            
            # Bandas para vegetação aquática
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
    """Obtém dados SMAP para umidade do solo (indirect bloom indicator)"""
    if not EE_WORKING:
        return []
    
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        
        # SMAP - umidade do solo
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

# =============================================================================
# FUNÇÕES DE DETECÇÃO DE EVENTOS
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
            # Início de evento
            in_event = True
            event_start = point['date']
            event_peak = point['date']
            event_max = ndvi
        elif in_event and ndvi > event_max:
            # Atualiza pico do evento
            event_peak = point['date']
            event_max = ndvi
        elif in_event and ndvi < threshold:
            # Fim do evento
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
            event_start = None
            event_peak = None
            event_max = 0
    
    # Se ainda está em evento no final dos dados
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
    chlorophyll_values = [d['chlorophyll'] for d in data_sorted if 'chlorophyll' in d and d['chlorophyll'] is not None]
    
    if not chlorophyll_values:
        return []
    
    mean_chlorophyll = np.mean(chlorophyll_values)
    std_chlorophyll = np.std(chlorophyll_values)
    threshold = mean_chlorophyll + 2 * std_chlorophyll  # Threshold mais alto para blooms
    
    events = []
    in_event = False
    event_start = None
    event_peak = None
    event_max = 0
    
    for i, point in enumerate(data_sorted):
        if 'chlorophyll' not in point or point['chlorophyll'] is None:
            continue
            
        chlorophyll = point['chlorophyll']
        
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
                'sensor': 'MODIS',
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
        
        # Detecção de eventos de alta umidade
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
        
        # Detecção de eventos de baixa umidade (seca)
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
                'type': 'low_soisture',
                'start': event_start,
                'peak': event_peak,
                'end': point['date'],
                'duration_days': duration_days,
                'min_moisture': round(event_extreme, 3),
                'severity': 'high' if event_extreme < mean_moisture - 2 * std_moisture else 'medium',
                'confidence': min(0.95, 0.6 + duration_days * 0.01)
            })
            
            in_low_event = False
    
    # Verificar se há eventos em andamento no final dos dados
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
# ENDPOINTS
# =============================================================================

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Endpoint principal para análise multi-sensor"""
    results = {}
    all_events = []
    
    # Verificar sensores solicitados
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
                grace_data = GraceData.get_grace_data_real(request.lon, request.lat, request.start, request.end, request.grace_analysis)
                results['grace'] = grace_data
                
        except Exception as e:
            print(f"❌ Erro no sensor {sensor}: {e}")
            results[sensor] = {"error": str(e)}
    
    # Consolidar eventos
    consolidated_events = []
    if all_events:
        event_df = pd.DataFrame(all_events)
        if not event_df.empty:
            # Agrupa eventos próximos
            event_df['start_date'] = pd.to_datetime(event_df['start'])
            event_df = event_df.sort_values('start_date')
            
            for _, row in event_df.iterrows():
                consolidated_events.append(row.to_dict())
    
    return {
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
        # Listar todas as coleções GRACE disponíveis
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
                
                # Tentar obter informações da primeira imagem
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


