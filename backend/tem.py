import ee
import geemap
import os

# Inicializar Earth Engine
try:
    ee.Initialize()
    print("✅ Earth Engine inicializado com sucesso!")
except Exception as e:
    print("❌ Erro ao inicializar Earth Engine. Faça autenticação primeiro.")
    print("Execute: earthengine authenticate no terminal")
    raise e

# 1. Definir área de interesse (Luanda, Angola) - coordenadas corrigidas
aoi = ee.Geometry.Point([13.2344, -8.8383]).buffer(10000)  # Buffer de 10km

# 2. Coleção Sentinel-2 harmonizada
s2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
      .filterBounds(aoi)
      .filterDate("2024-12-01", "2024-12-31")
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)))

# Verificar se há imagens disponíveis
image_count = s2.size().getInfo()
print(f"📊 Número de imagens encontradas: {image_count}")

if image_count == 0:
    print("⚠️ Nenhuma imagem encontrada para os critérios. Tentando período diferente...")
    # Tentar um período mais amplo
    s2 = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
          .filterBounds(aoi)
          .filterDate("2024-01-01", "2024-12-31")
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))
    
    image_count = s2.size().getInfo()
    print(f"📊 Número de imagens no período ampliado: {image_count}")

if image_count == 0:
    raise Exception("❌ Nenhuma imagem Sentinel-2 encontrada para a área e período especificados.")

# 3. Selecionar a imagem menos nublada
image = s2.sort('CLOUDY_PIXEL_PERCENTAGE').first()

# 4. Criar NDVI
ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

# 5. Criar composição RGB
rgb = image.select(['B4', 'B3', 'B2'])

# 6. Criar diretório de saída se não existir
out_dir = "downloads/"
os.makedirs(out_dir, exist_ok=True)

out_ndvi = os.path.join(out_dir, "ndvi_luanda.tif")
out_rgb = os.path.join(out_dir, "rgb_luanda.tif")

# 7. Obter a região de download
region = aoi.bounds().getInfo()['coordinates']

try:
    # Fazer download do NDVI
    print("⬇️ Baixando imagem NDVI...")
    geemap.ee_export_image(
        ndvi,
        filename=out_ndvi,
        scale=100,  # Reduzi a escala para 100m para arquivo menor (altere para 10 se quiser mais detalhe)
        region=region,
        file_per_band=False
    )
    
    # Fazer download do RGB
    print("⬇️ Baixando imagem RGB...")
    geemap.ee_export_image(
        rgb,
        filename=out_rgb,
        scale=100,  # Mesma escala para consistência
        region=region,
        file_per_band=False
    )
    
    print(f"✅ NDVI salvo em: {out_ndvi}")
    print(f"✅ RGB salvo em: {out_rgb}")
    print(f"📁 Diretório: {os.path.abspath(out_dir)}")
    
except Exception as e:
    print(f"❌ Erro durante o download: {e}")
    print("💡 Dica: Tente reduzir a área ou aumentar a escala")