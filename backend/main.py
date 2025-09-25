"""
Bloom Detection Backend - Earth Engine REAL (Multi-Sensor)
"""
import os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# =============================================================================
# CONFIGURAÇÃO EARTH ENGINE REAL
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
    title="Bloom Detection API - Multi-Sensor",
    description="API para detecção de florações com dados Landsat, SMAP, VIIRS e MODIS",
    version="2.0.0"
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
    start: str = Field("2024-01-01", description="Data início YYYY-MM-DD")
    end: str = Field("2024-12-31", description="Data fim YYYY-MM-DD")
    buffer_km: float = Field(5.0, description="Raio da área em km")
    sensors: List[str] = Field(['landsat', 'modis', 'viirs', 'smap'], description="Sensores a usar")

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
        viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP09GA') \
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

# =============================================================================
# FUNÇÕES DE FALLBACK (DEMO) - MULTI-SENSOR
# =============================================================================

def generate_demo_data(lon: float, lat: float, start_date: str, end_date: str, sensor_type: str):
    """Gera dados de demonstração para diferentes sensores"""
    from datetime import datetime as dt
    
    start_dt = dt.strptime(start_date, "%Y-%m-%d")
    end_dt = dt.strptime(end_date, "%Y-%m-%d")
    
    # Frequências diferentes por sensor
    frequencies = {
        'landsat': 16,    # 16 dias
        'modis': 1,       # Diário
        'viirs': 1,       # Diário  
        'smap': 3         # 3 dias
    }
    
    freq_days = frequencies.get(sensor_type, 16)
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current)
        current += timedelta(days=freq_days)
    
    records = []
    for i, date in enumerate(dates):
        day_of_year = date.timetuple().tm_yday
        
        # Padrões diferentes por sensor
        if sensor_type == 'landsat':
            seasonal = 0.3 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
            base_value = 0.3
            noise = np.random.normal(0, 0.05)
            value = base_value + seasonal + noise
            records.append({
                'date': date.strftime("%Y-%m-%d"),
                'NDVI': max(0.1, min(0.9, value)),
                'sensor': 'Landsat',
                'source': 'demo'
            })
            
        elif sensor_type == 'modis':
            # Clorofila em mg/m³
            seasonal = 2 * math.sin(2 * math.pi * (day_of_year - 100) / 365)
            base_chlorophyll = 5.0
            bloom_effect = 8.0 if 150 <= day_of_year <= 210 else 0.0
            chlorophyll = base_chlorophyll + seasonal + bloom_effect + np.random.normal(0, 1)
            sst = 20 + 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2)
            
            records.append({
                'date': date.strftime("%Y-%m-%d"),
                'chlorophyll': max(0.1, chlorophyll),
                'sst': max(-2, sst),
                'sensor': 'MODIS',
                'source': 'demo'
            })
            
        elif sensor_type == 'viirs':
            # VIIRS com padrão similar ao Landsat mas mais frequente
            seasonal = 0.25 * math.sin(2 * math.pi * (day_of_year - 70) / 365)
            base_value = 0.28
            value = base_value + seasonal + np.random.normal(0, 0.03)
            records.append({
                'date': date.strftime("%Y-%m-%d"),
                'NDVI': max(0.1, min(0.9, value)),
                'sensor': 'VIIRS', 
                'source': 'demo'
            })
            
        elif sensor_type == 'smap':
            # Umidade do solo em m³/m³
            seasonal = 0.1 * math.sin(2 * math.pi * (day_of_year - 90) / 365)
            base_moisture = 0.3
            moisture = base_moisture + seasonal + np.random.normal(0, 0.02)
            records.append({
                'date': date.strftime("%Y-%m-%d"),
                'soil_moisture': max(0.01, min(0.8, moisture)),
                'sensor': 'SMAP',
                'source': 'demo'
            })
    
    print(f"📊 Dados demo {sensor_type}: {len(records)} pontos")
    return records

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Bloom Detection API - Multi-Sensor",
        "sensors_supported": ["Landsat", "MODIS", "VIIRS", "SMAP"],
        "earth_engine_status": EE_WORKING,
        "mode": "real" if EE_WORKING else "demo"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "earth_engine": EE_WORKING,
        "version": "2.0.0",
        "sensors": ["Landsat", "MODIS", "VIIRS", "SMAP"]
    }

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """Endpoint principal multi-sensor"""
    
    print(f"🔍 Analisando multi-sensor: {req.lon}, {req.lat}")
    print(f"📡 Sensores solicitados: {req.sensors}")
    
    all_data = {}
    sensors_used = []
    
    # Processa cada sensor solicitado
    for sensor in req.sensors:
        sensor_data = []
        source = "demo"
        
        if EE_WORKING:
            try:
                if sensor == 'landsat':
                    sensor_data = get_landsat_data(req.lon, req.lat, req.start, req.end, req.buffer_km)
                    source = "earth_engine" if sensor_data else "demo"
                elif sensor == 'modis':
                    sensor_data = get_modis_data(req.lon, req.lat, req.start, req.end, req.buffer_km)
                    source = "earth_engine" if sensor_data else "demo"
                elif sensor == 'viirs':
                    sensor_data = get_viirs_data(req.lon, req.lat, req.start, req.end, req.buffer_km)
                    source = "earth_engine" if sensor_data else "demo"
                elif sensor == 'smap':
                    sensor_data = get_smap_data(req.lon, req.lat, req.start, req.end, req.buffer_km)
                    source = "earth_engine" if sensor_data else "demo"
            except Exception as e:
                print(f"⚠️ Erro no sensor {sensor}: {e}")
        
        # Fallback para demo se necessário
        if not sensor_data:
            sensor_data = generate_demo_data(req.lon, req.lat, req.start, req.end, sensor)
            source = "demo"
        
        if sensor_data:
            all_data[sensor] = {
                'data': sensor_data,
                'source': source,
                'n_observations': len(sensor_data)
            }
            sensors_used.append(sensor)
    
    # Combina dados e detecta eventos
    combined_events = detect_multi_sensor_events(all_data)
    
    return {
        "status": "success",
        "sensors_used": sensors_used,
        "sensors_data": all_data,
        "earth_engine_used": EE_WORKING,
        "events_detected": len(combined_events),
        "events": combined_events,
        "coordinates": {"lon": req.lon, "lat": req.lat},
        "period": {"start": req.start, "end": req.end}
    }

def detect_multi_sensor_events(sensors_data):
    """Detecta eventos combinando múltiplos sensores"""
    events = []
    
    # Análise por sensor
    for sensor, data_info in sensors_data.items():
        sensor_data = data_info['data']
        if not sensor_data:
            continue
            
        # Detecta eventos específicos do sensor
        if sensor == 'landsat' or sensor == 'viirs':
            events.extend(detect_ndvi_events(sensor_data, sensor))
        elif sensor == 'modis':
            events.extend(detect_chlorophyll_events(sensor_data))
        elif sensor == 'smap':
            events.extend(detect_moisture_events(sensor_data))
    
    return events

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
    
    for point in data_sorted:
        if 'NDVI' not in point:
            continue
            
        if point['NDVI'] > threshold and not in_event:
            in_event = True
            event_start = point['date']
            event_peak = point['date']
            event_max = point['NDVI']
        elif point['NDVI'] > threshold and in_event:
            if point['NDVI'] > event_max:
                event_max = point['NDVI']
                event_peak = point['date']
        elif point['NDVI'] <= threshold and in_event:
            in_event = False
            if event_start and event_peak:
                events.append({
                    'sensor': sensor_name,
                    'type': 'ndvi_peak',
                    'start': event_start,
                    'peak': event_peak,
                    'end': point['date'],
                    'peak_value': round(event_max, 3),
                    'intensity': 'high' if event_max > mean_ndvi + 2*std_ndvi else 'medium'
                })
    
    return events

def detect_chlorophyll_events(data):
    """Detecta eventos baseados em clorofila"""
    if len(data) < 3:
        return []
    
    data_sorted = sorted(data, key=lambda x: x['date'])
    chlorophyll_values = [d['chlorophyll'] for d in data_sorted if 'chlorophyll' in d]
    
    if not chlorophyll_values:
        return []
    
    mean_chlor = np.mean(chlorophyll_values)
    threshold = mean_chlor + 5.0  # Threshold para blooms de algas
    
    events = []
    in_event = False
    event_start = None
    event_peak = None
    event_max = 0
    
    for point in data_sorted:
        if 'chlorophyll' not in point:
            continue
            
        if point['chlorophyll'] > threshold and not in_event:
            in_event = True
            event_start = point['date']
            event_peak = point['date']
            event_max = point['chlorophyll']
        elif point['chlorophyll'] > threshold and in_event:
            if point['chlorophyll'] > event_max:
                event_max = point['chlorophyll']
                event_peak = point['date']
        elif point['chlorophyll'] <= threshold and in_event:
            in_event = False
            if event_start and event_peak:
                events.append({
                    'sensor': 'MODIS',
                    'type': 'chlorophyll_bloom',
                    'start': event_start,
                    'peak': event_peak,
                    'end': point['date'],
                    'peak_value': round(event_max, 1),
                    'intensity': 'high' if event_max > threshold + 10 else 'medium'
                })
    
    return events

def detect_moisture_events(data):
    """Detecta eventos baseados em umidade do solo"""
    # Implementação simplificada para SMAP
    events = []
    # Lógica de detecção baseada em umidade...
    return events

if __name__ == "__main__":
    import uvicorn
    print("🚀 Iniciando servidor Bloom Detection API - Multi-Sensor")
    print(f"📍 Earth Engine Status: {EE_WORKING}")
    print("📡 Sensores: Landsat, MODIS, VIIRS, SMAP")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)