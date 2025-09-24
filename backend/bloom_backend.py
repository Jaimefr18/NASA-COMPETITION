"""
bloom_backend.py

Backend completo (FastAPI) para detecção e preparação de dados de eventos de floração
usando Landsat, VIIRS, MODIS, EMIT, PACE e AVIRIS.

- Múltiplos endpoints REST:
  - /health                 -> status
  - /analyze                -> execução completa para um ponto (Landsat/VIIRS/MODIS)
  - /timeseries             -> devolve séries temporais (CSV) por sensor para treino
  - /prepare_training       -> gera features tabulares para treinar IA (salva parquet/csv)
  - /upload_model           -> envia e registra modelo ML (joblib/pickle)
  - /predict                -> faz predição com o modelo carregado (ou devolve erro)
  - /match_spectra          -> casar espectros (AVIRIS/EMIT) via SAM (upload de ficheiros)
  - /download/{filename}    -> descarrega ficheiros gerados
  - /upload_local_timeseries-> upload CSVs para usar no modo local (fallback)
  - /upload_hypercube       -> upload ENVI/NPZ hyperspectral para /data/uploads

- Requisitos e instruções no final do ficheiro.
- Autor: template para o teu projecto (Francisco André)
- Data: 2025-09-21

USO RÁPIDO:
1) Criar venv e instalar dependências (ver requirements.txt conteúdo no final)
2) Autenticar Earth Engine (recomendado) se for usar EE:
    pip install earthengine-api
    earthengine authenticate
    earthengine init
3) Run:
    uvicorn bloom_backend:app --reload --port 8000

NOTA: O servidor NÃO faz downloads automáticos de grandes coleções sem tu configures EE ou carregues ficheiros.
"""

# Standard imports
import os
import io
import json
import math
import datetime as dt
from typing import List, Optional, Dict, Any
from pathlib import Path

# Data stack
import numpy as np
import pandas as pd

# ML / Serialization
try:
    import joblib
except Exception:
    joblib = None
    import pickle

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Optional geospatial libs
try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import rasterio
except Exception:
    rasterio = None

try:
    import xarray as xr
except Exception:
    xr = None

# Stats / signal
from scipy import signal
from statsmodels.tsa.seasonal import STL

# Optional Earth Engine
try:
    import ee
    EE_AVAILABLE = True
    ee.Initialize(project='tesr-473014')
except Exception:
    ee = None
    EE_AVAILABLE = False

# Create app
app = FastAPI(title="Bloom Detection Backend",
              description="API backend for detecting and preparing bloom event data (Landsat, VIIRS, MODIS, EMIT, PACE, AVIRIS).",
              version="0.1")

# Allow CORS from all origins for demo (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Filesystem setup
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
for d in [DATA_DIR, MODELS_DIR, UPLOADS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- Utilities ----------

def debug_print(msg: str):
    print(f"[bloom_backend] {dt.datetime.utcnow().isoformat()} - {msg}")

def ensure_date(d):
    if isinstance(d, str):
        return dt.datetime.fromisoformat(d).date()
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    raise ValueError("Unsupported date format: {}".format(type(d)))

def make_aoi_geometry(lon: float, lat: float, buffer_km: float = 1.0):
    """
    Simple square AOI in GeoJSON around lon/lat with approximate buffer in km.
    For small buffers this is fine; for precise geodesic use shapely/geopy.
    """
    buffer_deg = buffer_km / 111.32
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - buffer_deg, lat - buffer_deg],
            [lon - buffer_deg, lat + buffer_deg],
            [lon + buffer_deg, lat + buffer_deg],
            [lon + buffer_deg, lat - buffer_deg],
            [lon - buffer_deg, lat - buffer_deg]
        ]]
    }

# ---------- Earth Engine helpers (preferred) ----------

# ---------- Earth Engine helpers (preferred) ----------
def get_quality_band(image):
    bands = image.bandNames().getInfo()
    if "QA_PIXEL" in bands:
        return image.select("QA_PIXEL")

    else:
        raise Exception(f"Nenhuma banda QA encontrada. Disponíveis: {bands}")


def ee_initialize():
    """
    Inicializa o Earth Engine. O utilizador deve fazer:
      pip install earthengine-api
      earthengine authenticate
      earthengine init
    antes de usar funções que dependam do EE.
    """
    if not EE_AVAILABLE:
        raise RuntimeError("earthengine-api não instalado. Instale com `pip install earthengine-api` e autentique.")
    try:
        ee.Initialize()
        debug_print("Earth Engine inicializado com sucesso.")
    except Exception as e:
        debug_print(f"ee.Initialize() falhou: {e}")
        raise RuntimeError("Falha ao inicializar Earth Engine. Execute `earthengine authenticate` no terminal e depois `earthengine init`.")

def geom_from_geojson(geojson):
    if not EE_AVAILABLE:
        raise RuntimeError("Earth Engine não disponível")
    return ee.Geometry(geojson)

def _safe_date_str(d):
    d = ensure_date(d)
    return d.isoformat()

def get_landsat_collection_ee(aoi_geojson, start_date, end_date, sensors: List[str]=None):
    """
    Retorna ee.ImageCollection com SR Landsat (por default Landsat 8 SR).
    sensors: lista de strings de coleções ee, ex ['LANDSAT/LC08/C01/T1_SR']
    """
    if not EE_AVAILABLE:
        raise RuntimeError("Earth Engine necessário para esta função.")
    if sensors is None:
        sensors = ['LANDSAT/LC08/C02/T1_L2'] 

    geom = geom_from_geojson(aoi_geojson)
    start = _safe_date_str(start_date)
    end = _safe_date_str(end_date)
    col = None
    for s in sensors:
        c = ee.ImageCollection(s).filterBounds(geom).filterDate(start, end)
        # Tentativa de aplicar máscara de nuvens para SR
        try:
            def mask_sr(img):
                qa = get_quality_band(img)
                mask = qa.bitwiseAnd(1 << 5).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
                return img.updateMask(mask)
            c = c.map(mask_sr)
        except Exception:
            pass
        col = c if col is None else col.merge(c)
    return col.sort('system:time_start')

def get_modis_collection_ee(aoi_geojson, start_date, end_date, product='MODIS/061/MOD13Q1'):
    if not EE_AVAILABLE:
        raise RuntimeError("Earth Engine necessário para esta função.")
    geom = geom_from_geojson(aoi_geojson)
    return ee.ImageCollection(product).filterBounds(geom).filterDate(_safe_date_str(start_date), _safe_date_str(end_date)).sort('system:time_start')

def get_viirs_collection_ee(aoi_geojson, start_date, end_date, product='NOAA/VIIRS/001/VNP09H1'):
    if not EE_AVAILABLE:
        raise RuntimeError("Earth Engine necessário para esta função.")
    geom = geom_from_geojson(aoi_geojson)
    return ee.ImageCollection(product).filterBounds(geom).filterDate(_safe_date_str(start_date), _safe_date_str(end_date)).sort('system:time_start')

# ---------- Preprocessing & indices ----------
def compute_ndvi_ee(image):
    """
    NDVI para ee.Image (tenta identificar bandas comuns).
    """
    if not EE_AVAILABLE:
        raise RuntimeError("EE requerido")
    bn = image.bandNames().getInfo()
    if 'B5' in bn and 'B4' in bn: # Landsat 8
        return image.normalizedDifference(['B5','B4']).rename('NDVI')
    if 'B4' in bn and 'B3' in bn: # Landsat 7/5-like
        return image.normalizedDifference(['B4','B3']).rename('NDVI')
    if 'nir' in bn and 'red' in bn:
        return image.normalizedDifference(['nir','red']).rename('NDVI')
    raise ValueError("Banda para NDVI não encontrada: " + str(bn))

def compute_evi_ee(image):
    """
    EVI para ee.Image (fórmula padrão). Mapeia bandas para NIR/RED/BLUE conforme disponível.
    """
    if not EE_AVAILABLE:
        raise RuntimeError("EE requerido")
    bn = image.bandNames().getInfo()
    if 'B5' in bn and 'B4' in bn and 'B2' in bn:
        nir = image.select('B5'); red = image.select('B4'); blue = image.select('B2')
    elif 'B4' in bn and 'B3' in bn and 'B1' in bn:
        nir = image.select('B4'); red = image.select('B3'); blue = image.select('B1')
    else:
        raise ValueError("Bandas para EVI não encontradas: " + str(bn))
    evi = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename('EVI')
    return evi

def ee_collection_to_timeseries(collection, aoi_geojson, reducer='median', index_band=None):
    """
    Reduz uma ee.ImageCollection para um DataFrame pandas através de reduceRegion sobre o AOI.
    Retorna DataFrame indexado por data.
    Nota: usa getInfo() -> pode falhar para coleções muito grandes.
    """
    if not EE_AVAILABLE:
        raise RuntimeError("EE necessário")
    geom = geom_from_geojson(aoi_geojson)
    if reducer == 'median':
        ee_reducer = ee.Reducer.median()
    elif reducer == 'mean':
        ee_reducer = ee.Reducer.mean()
    else:
        ee_reducer = ee.Reducer.median()

    def prepare(img):
        if index_band is None:
            try:
                ndvi = compute_ndvi_ee(img)
                img2 = img.addBands(ndvi)
                try:
                    evi = compute_evi_ee(img)
                    img2 = img2.addBands(evi)
                except Exception:
                    pass
                return img2
            except Exception:
                return img
        else:
            return img

    # Map prepare (use safe wrapper)
    col_prep = collection.map(lambda i: prepare(i))
    # Map each image -> Feature(properties)
    def image_to_feature(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        stats = image.reduceRegion(ee_reducer, geom, 1000)
        return ee.Feature(None, ee.Dictionary(stats).set('date', date))
    features = col_prep.map(lambda img: image_to_feature(img)).filter(ee.Filter.notNull(['date']))
    fc = ee.FeatureCollection(features)
    try:
        data = fc.getInfo()
    except Exception as e:
        debug_print(f"ee getInfo falhou: {e}")
        raise RuntimeError("Falha ao pedir dados ao Earth Engine. Coleção muito grande ou credenciais.")
    records = []
    for f in data.get('features', []):
        props = f.get('properties', {})
        date = props.pop('date', None)
        if date is None:
            continue
        row = {'date': date}
        for k, v in props.items():
            row[k] = float(v) if v is not None else np.nan
        records.append(row)
    if len(records) == 0:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df

# ---------- Time series decomposition & bloom detection ----------
def seasonal_baseline(series: pd.Series, period:int=365):
    """
    Decomposição usando STL para obter seasonal + trend + resid.
    Retorna DataFrame com observed, trend, seasonal, resid.
    """
    s = series.dropna()
    if s.empty:
        return pd.DataFrame()
    s_daily = s.resample('D').mean().interpolate('time', limit=14)
    try:
        stl = STL(s_daily, period=period, robust=True)
        res = stl.fit()
        df = pd.DataFrame({
            'observed': s_daily,
            'trend': res.trend,
            'seasonal': res.seasonal,
            'resid': res.resid
        }, index=s_daily.index)
        return df
    except Exception as e:
        debug_print(f"STL falhou: {e}. Fallback: median por DOY.")
        df = s_daily.to_frame(name='observed')
        df['doy'] = df.index.dayofyear
        median_by_doy = df.groupby('doy')['observed'].median()
        df['seasonal'] = df['doy'].map(median_by_doy)
        df['trend'] = df['observed'].rolling(window=30, min_periods=1, center=True).mean()
        df['resid'] = df['observed'] - df['seasonal']
        return df[['observed','trend','seasonal','resid']]

def detect_bloom_events(index_series: pd.Series, seasonal_period_days:int=365, z_thresh:float=2.0, min_duration_days:int=3):
    """
    Detecção baseada em anomalia sazonal:
      - Decompõe série (STL)
      - Anomalia = observed - seasonal
      - z-score rolling (janela 30 dias)
      - Eventos quando z > z_thresh por >= min_duration_days
    Retorna lista de eventos.
    """
    if not isinstance(index_series, pd.Series):
        index_series = pd.Series(index_series)
    s = index_series.sort_index()
    s = s.resample('D').mean().interpolate('time', limit=14)
    baseline = seasonal_baseline(s, period=seasonal_period_days)
    if baseline.empty:
        return []
    baseline['anomaly'] = baseline['observed'] - baseline['seasonal']
    rolling_std = baseline['anomaly'].rolling(window=30, min_periods=7, center=True).std().fillna(method='bfill').fillna(method='ffill')
    rolling_mean = baseline['anomaly'].rolling(window=30, min_periods=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    baseline['z'] = (baseline['anomaly'] - rolling_mean) / (rolling_std.replace(0, np.nan))
    baseline['z'] = baseline['z'].fillna(0)
    mask = baseline['z'] > z_thresh
    events = []
    in_event = False
    start = None
    for date, val in mask.items():
    
        if val and not in_event:
            in_event = True
            start = date
            peak_date = date
            peak_z = baseline.loc[date, 'z']
            peak_value = baseline.loc[date, 'observed']
        elif val and in_event:
            if baseline.loc[date, 'z'] > peak_z:
                peak_z = baseline.loc[date, 'z']
                peak_date = date
                peak_value = baseline.loc[date, 'observed']
        elif (not val) and in_event:
            end = date - pd.Timedelta(days=1)
            duration = (end - start).days + 1
            if duration >= min_duration_days:
                events.append({
                    'start': start.date(),
                    'peak': peak_date.date(),
                    'end': end.date(),
                    'duration_days': duration,
                    'peak_value': float(peak_value),
                    'max_z': float(peak_z)
                })
            in_event = False
            start = None
    if in_event and start is not None:
        end = baseline.index[-1]
        duration = (end - start).days + 1
        if duration >= min_duration_days:
            events.append({
                'start': start.date(),
                'peak': peak_date.date(),
                'end': end.date(),
                'duration_days': duration,
                'peak_value': float(peak_value),
                'max_z': float(peak_z)
            })
    return events

# ---------- Spectral matching (EMIT/AVIRIS) ----------
def spectral_angle_mapper(obs_spectrum: np.ndarray, ref_spectra: np.ndarray):
    """
    SAM entre obs (bands,) e ref_spectra (n_refs, bands). Retorna ângulos (rad).
    """
    obs = obs_spectrum.astype(float)
    ref = ref_spectra.astype(float)
    obs_norm = np.linalg.norm(obs)
    ref_norm = np.linalg.norm(ref, axis=1)
    denom = obs_norm * ref_norm
    denom[denom == 0] = 1e-12
    cos_theta = np.dot(ref, obs) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta)
    return angles

def match_spectra_pixelwise(hypercube: np.ndarray, ref_spectra: np.ndarray, threshold_angle=0.2):
    """
    Dado hypercube (h,w,b) e biblioteca ref_spectra (n_refs,b),
    devolve indices de melhor match (h,w) e angles (h,w). Pixels com angle>threshold = -1.
    """
    h, w, b = hypercube.shape
    flat = hypercube.reshape(-1, b)
    angles = np.apply_along_axis(lambda x: spectral_angle_mapper(x, ref_spectra), 1, flat)
    best_idx = np.argmin(angles, axis=1)
    best_angle = angles[np.arange(angles.shape[0]), best_idx]
    best_idx[best_angle > threshold_angle] = -1
    return best_idx.reshape(h, w), best_angle.reshape(h, w)

# ---------- Fallback local loaders ----------
def load_hyperspectral_envi(path_to_envi: str):
    """
    Tenta carregar ficheiro ENVI via spectral package. Se não instalado, pede para instalar.
    """
    try:
        from spectral import open_image
    except Exception:
        raise RuntimeError("Para ler ENVI instale `spectral` (pip install spectral)")
    img = open_image(path_to_envi)
    arr = img.load()
    return np.array(arr)

# ---------- ML helpers (preparar ambiente para IA) ----------
LOADED_MODEL = None
LOADED_MODEL_NAME = None

def save_features_for_training(df: pd.DataFrame, name_prefix: str = "features"):
    """
    Salva DataFrame em CSV e parquet no diretório processed e devolve caminho.
    """
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fname_csv = PROCESSED_DIR / f"{name_prefix}_{ts}.csv"
    fname_parquet = PROCESSED_DIR / f"{name_prefix}_{ts}.parquet"
    df.to_csv(fname_csv, index=True)
    try:
        df.to_parquet(fname_parquet)
    except Exception:
        debug_print("Parquet save falhou (pyarrow/fastparquet ausente). Apenas CSV salvo.")
        fname_parquet = None
    return str(fname_csv), str(fname_parquet) if fname_parquet else None

def load_model_from_path(path: str):
    global LOADED_MODEL, LOADED_MODEL_NAME
    if joblib:
        LOADED_MODEL = joblib.load(path)
    else:
        with open(path, "rb") as f:
            LOADED_MODEL = pickle.load(f)
    LOADED_MODEL_NAME = Path(path).name
    debug_print(f"Modelo carregado: {LOADED_MODEL_NAME}")
    return LOADED_MODEL_NAME

def predict_with_loaded_model(X: pd.DataFrame):
    if LOADED_MODEL is None:
        raise RuntimeError("Nenhum modelo carregado. Use /upload_model para enviar e registrar um modelo.")
    try:
        preds = LOADED_MODEL.predict(X)
        return preds.tolist()
    except Exception as e:
        raise RuntimeError(f"Erro na predição: {e}")

# ---------- High-level orchestration ----------
def get_multisensor_timeseries_ee(aoi_geojson, start_date, end_date, sensors: List[str]=['landsat','viirs','modis']):
    """
    Wrapper que devolve dicionário de DataFrames por sensor usando Earth Engine.
    """
    if not EE_AVAILABLE:
        raise RuntimeError("EE requerido")
    out = {}
    if 'landsat' in sensors:
        landsat_col = get_landsat_collection_ee(aoi_geojson, start_date, end_date)
        df_l = ee_collection_to_timeseries(landsat_col, aoi_geojson, reducer='median')
        # criar NDVI se não existir
        if 'NDVI' not in df_l.columns and 'B5_median' in df_l.columns:
            df_l['NDVI'] = (df_l.get('B5_median', np.nan) - df_l.get('B4_median', np.nan)) / (df_l.get('B5_median', np.nan) + df_l.get('B4_median', np.nan) + 1e-12)
        out['landsat'] = df_l
    if 'modis' in sensors:
        modis_col = get_modis_collection_ee(aoi_geojson, start_date, end_date)
        df_m = ee_collection_to_timeseries(modis_col, aoi_geojson, reducer='median')
        out['modis'] = df_m
    if 'viirs' in sensors:
        viirs_col = get_viirs_collection_ee(aoi_geojson, start_date, end_date)
        df_v = ee_collection_to_timeseries(viirs_col, aoi_geojson, reducer='median')
        out['viirs'] = df_v
    return out

# ---------- API Models ----------
class AnalyzeRequest(BaseModel):
    lon: float = Field(..., description="Longitude (ex: -13.234)")
    lat: float = Field(..., description="Latitude (ex: 8.826)")
    start: str = Field((dt.date.today() - dt.timedelta(days=365)).isoformat(), description="Start date YYYY-MM-DD")
    end: str = Field(dt.date.today().isoformat(), description="End date YYYY-MM-DD")
    buffer_km: float = Field(2.0, description="AOI buffer in km")
    sensors: List[str] = Field(['landsat','viirs','modis'], description="Which sensors to query")
    z_threshold: float = Field(2.0, description="z-score threshold for anomaly detection")
    min_duration_days: int = Field(3, description="Minimum days to consider event")

class TimeseriesRequest(BaseModel):
    lon: float
    lat: float
    start: str
    end: str
    buffer_km: float = 2.0
    sensors: List[str] = ['landsat','viirs','modis']

class PredictRequest(BaseModel):
    features: Optional[Dict[str, List[float]]] = None  # features em formato coluna->lista
    # alternativa: fazer predição para localidade e período (não implementado automaticamente)
    lon: Optional[float] = None
    lat: Optional[float] = None

# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Endpoint principal para executar pipeline no ponto solicitado.
    - Retorna série resumida por sensor e eventos detectados.
    - Requer Earth Engine configurado para funcionar automaticamente.
    """
    # Inicialização EE
    if not EE_AVAILABLE:
        raise HTTPException(status_code=503, detail={
            "error": "Earth Engine API não instalada/ativa no servidor.",
            "how_to": "Instale earthengine-api e execute `earthengine authenticate` e `earthengine init`.",
            "alternative": "Use /upload_local_timeseries para enviar séries prontas e então /prepare_training."
        })
    try:
        ee_initialize()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    aoi = make_aoi_geometry(req.lon, req.lat, buffer_km=req.buffer_km)
    try:
        multis = get_multisensor_timeseries_ee(aoi, req.start, req.end, sensors=req.sensors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro a obter séries: {e}")

    results = {'aoi': aoi, 'params': req.dict(), 'sensors': {}}
    events_summary = {}
    for sname, df in multis.items():
        if df is None or df.empty:
            results['sensors'][sname] = {'note': 'empty collection or no data for AOI'}
            events_summary[sname] = []
            continue
        # heurística para escolher coluna de índice (NDVI/EVI)
        candidate_cols = [c for c in df.columns if 'ndvi' in c.lower() or c.lower()=='ndvi']
        if candidate_cols:
            idx_col = candidate_cols[0]
        else:
            # tentar construir proxy
            if 'B5_median' in df.columns and 'B4_median' in df.columns:
                df['NDVI_proxy'] = (df['B5_median'] - df['B4_median']) / (df['B5_median'] + df['B4_median'] + 1e-12)
                idx_col = 'NDVI_proxy'
            else:
                idx_col = df.columns[0]
        series = df[idx_col]
        events = detect_bloom_events(series, seasonal_period_days=365, z_thresh=req.z_threshold, min_duration_days=req.min_duration_days)
        # salvar séries resumidas em CSV temporário
        fname = PROCESSED_DIR / f"{sname}_timeseries_{req.lon}_{req.lat}_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv"
        df.to_csv(fname)
        results['sensors'][sname] = {
            'index_column': idx_col,
            'n_obs': int(len(df)),
            'timeseries_csv': str(fname.name),
            'events_count': len(events),
            'events_preview': events[:5]
        }
        events_summary[sname] = events
    results['events_summary'] = events_summary
    return JSONResponse(content=results)

@app.post("/timeseries")
def timeseries(req: TimeseriesRequest):
    """
    Retorna (gera e guarda) CSV com séries temporais para os sensores pediddos.
    Se EE não está disponível devolve instruções.
    """
    if not EE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configurar Earth Engine para usar este endpoint ou use /upload_local_timeseries.")
    try:
        ee_initialize()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    aoi = make_aoi_geometry(req.lon, req.lat, buffer_km=req.buffer_km)
    try:
        multis = get_multisensor_timeseries_ee(aoi, req.start, req.end, sensors=req.sensors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter coleções: {e}")
    out_files = []
    for sname, df in multis.items():
        if df is None or df.empty:
            continue
        fname = PROCESSED_DIR / f"{sname}_ts_{req.lon}_{req.lat}_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv"
        df.to_csv(fname)
        out_files.append(str(fname.name))
    return {"status":"ok","files": out_files}

@app.post("/prepare_training")
def prepare_training(lon: float = Body(...), lat: float = Body(...), start: str = Body(...), end: str = Body(...),
                     buffer_km: float = Body(2.0), sensors: List[str] = Body(['landsat','viirs','modis'])):
    """
    Prepara features tabulares para treino da IA:
      - Extrai NDVI/EVI, estatísticas (mean, std, peaks), anomalias, NDWI (proxy), hora do ano, etc.
      - Salva CSV/parquet em /data/processed e devolve caminhos.
    Requer Earth Engine.
    """
    if not EE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configure Earth Engine ou use /upload_local_timeseries.")
    try:
        ee_initialize()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    aoi = make_aoi_geometry(lon, lat, buffer_km=buffer_km)
    try:
        multis = get_multisensor_timeseries_ee(aoi, start, end, sensors=sensors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro a buscar séries: {e}")
    # Constrói DataFrame de features por data (união por data)
    df_features = None
    for sname, df in multis.items():
        if df is None or df.empty:
            continue
        df_local = df.copy()
        # selecionar colunas NDVI/EVI se existirem
        cols = [c for c in df_local.columns if 'ndvi' in c.lower() or 'evi' in c.lower() or 'B5_median' in c]
        # renomear prefixed columns
        df_local = df_local[cols]
        df_local = df_local.add_prefix(f"{sname}_")
        if df_features is None:
            df_features = df_local
        else:
            df_features = df_features.join(df_local, how='outer')
    if df_features is None or df_features.empty:
        raise HTTPException(status_code=400, detail="Nenhum dado disponível para os sensores pedidos.")
    # preencher gaps e calcular estatísticas
    df_daily = df_features.resample('D').mean().interpolate('time', limit=14)
    # Features ex:
    feat = pd.DataFrame(index=df_daily.index)
    for col in df_daily.columns:
        feat[col + "_lag1"] = df_daily[col].shift(1)
        feat[col + "_lag7"] = df_daily[col].shift(7)
        feat[col + "_rolling7_mean"] = df_daily[col].rolling(7, min_periods=1).mean()
        feat[col + "_rolling7_std"] = df_daily[col].rolling(7, min_periods=1).std().fillna(0)
        feat[col + "_doy"] = df_daily.index.dayofyear
    # Remover linhas totalmente nan
    feat = feat.dropna(how='all')
    csv_path, parquet_path = save_features_for_training(feat, name_prefix=f"features_{lon}_{lat}")
    return {"status":"ok", "csv": csv_path, "parquet": parquet_path, "rows": len(feat)}

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload de modelo ML (joblib/pickle). O modelo fica disponível para /predict.
    - Salva em /models e tenta carregar.
    """
    filename = file.filename
    dest = MODELS_DIR / filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        model_name = load_model_from_path(str(dest))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modelo salvo mas falha ao carregar: {e}")
    return {"status":"ok","model": model_name}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Faz predição com modelo carregado. Espera features (dict col->list) ou devolve erro.
    - Retorna lista de predições.
    """
    if LOADED_MODEL is None:
        raise HTTPException(status_code=400, detail="Nenhum modelo carregado. Use /upload_model primeiro.")
    if req.features is None:
        raise HTTPException(status_code=400, detail="Envie 'features' no corpo (dict de col->lista).")
    try:
        df = pd.DataFrame(req.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro a construir DataFrame das features: {e}")
    try:
        preds = predict_with_loaded_model(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"predictions": preds, "n": len(preds)}

@app.post("/match_spectra")
async def match_spectra(hypercube: UploadFile = File(...), ref_spectra: UploadFile = File(...), threshold_angle: float = 0.2):
    """
    Faz match spectral entre ficheiro hypercube (ENVI .hdr + binary ou npz contendo array) e biblioteca ref (npz ou csv).
    - hypercube: upload do ficheiro (aceita .npz, .npy - para demo)
    - ref_spectra: npz/npY/csv com referência (n_refs x bands)
    Retorna um ficheiro de índice (CSV) e estatísticas.
    """
    # Salva uploads
    hname = UPLOADS_DIR / f"hyper_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{hypercube.filename}"
    with open(hname, "wb") as f:
        f.write(await hypercube.read())
    rname = UPLOADS_DIR / f"ref_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{ref_spectra.filename}"
    with open(rname, "wb") as f:
        f.write(await ref_spectra.read())
    # Tentar carregar como npz/npy ou csv
    def load_array(path):
        p = Path(path)
        if p.suffix in ['.npz', '.npy']:
            return np.load(path)
        else:
            # assume csv
            return np.genfromtxt(path, delimiter=',')
    try:
        hdata = load_array(str(hname))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao ler hypercube: {e}")
    try:
        rdata = load_array(str(rname))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao ler ref_spectra: {e}")
    # Se npz retornou object (npz) -> extrair first
    if isinstance(hdata, np.lib.npyio.NpzFile):
        # tentar chave first
        keys = list(hdata.keys())
        hdata = hdata[keys[0]]
    if isinstance(rdata, np.lib.npyio.NpzFile):
        keys = list(rdata.keys())
        rdata = rdata[keys[0]]
    # Se hypercube 3D (h,w,b) OK; se 2D assume (pixels,bands)
    if hdata.ndim == 2:
        # tentar reformatar para (h,w,b) pequena heurística: se sqrt(#pixels) é inteiro -> square
        n_pixels, bands = hdata.shape
        side = int(round(math.sqrt(n_pixels)))
        if side*side == n_pixels:
            hdata = hdata.reshape((side, side, bands))
        else:
            raise HTTPException(status_code=400, detail="Hypercube 2D com #pixels não quadrado: adapte pré-processamento localmente.")
    if rdata.ndim == 1:
        rdata = rdata.reshape(1, -1)
    if hdata.ndim != 3:
        raise HTTPException(status_code=400, detail="Hypercube deve ser 3D (h,w,b) após carregamento.")
    # Executar SAM matching
    try:
        idx_map, angle_map = match_spectra_pixelwise(hdata, rdata, threshold_angle=threshold_angle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no matching: {e}")
    # Salvar maps como npy e devolver caminhos
    idx_path = PROCESSED_DIR / f"match_idx_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.npy"
    ang_path = PROCESSED_DIR / f"match_angle_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.npy"
    np.save(idx_path, idx_map)
    np.save(ang_path, angle_map)
    return {"status":"ok", "idx_file": str(idx_path.name), "angle_file": str(ang_path.name), "shape": idx_map.shape}

@app.post("/upload_local_timeseries")
async def upload_local_timeseries(file: UploadFile = File(...)):
    """
    Upload CSV timeseries para usar no modo local (ex.: para treinar IA / teste).
    O CSV deve ter uma coluna 'date' e colunas de índices (NDVI/EVI) ou bandas.
    """
    name = file.filename
    dest = UPLOADS_DIR / name
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"status":"ok", "file": str(dest.name)}

@app.get("/download/{fname}")
def download(fname: str):
    """
    Serve ficheiros gerados em /data/processed ou /models.
    Atenção: em produção serve ficheiros de forma autenticada.
    """
    for d in [PROCESSED_DIR, MODELS_DIR, UPLOADS_DIR]:
        p = d / fname
        if p.exists():
            return FileResponse(path=str(p), filename=fname)
    raise HTTPException(status_code=404, detail="File not found")

# ---------- CLI / Run instructions ----------
if __name__ == "__main__":
    print("Este ficheiro é um FastAPI app. Execute com:")
    print("    uvicorn bloom_backend:app --reload --port 8000")
    print("Ver /health para ver se o servidor está disponível.")
