import ee

# Inicializa o Earth Engine no teu projeto
ee.Initialize(project='tesr-473014')

# Pega uma coleção do Landsat 8 Collection 2, Tier 1, nível 2
collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
    .filterDate("2023-01-01", "2023-12-31") \
    .filterBounds(ee.Geometry.Point(13.23, -8.84))  # Luanda

# Seleciona a primeira imagem disponível
image = collection.first()

# Verifica quais bandas a imagem tem
info = image.bandNames().getInfo()
print("Bandas disponíveis:", info)

# Pega metadados básicos
meta = image.getInfo()
print("ID da imagem:", meta["id"])
print("Data de aquisição:", meta["properties"]["DATE_ACQUIRED"])
