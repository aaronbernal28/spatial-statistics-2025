## ============= Imports =============
library(tidyverse)
library(sf)
library("leaflet")

DIR_DATA <- 'spatial-statistics-2025/data/'
DIR_IMG <- 'spatial-statistics-2025/images/'

## ============= Datos =============

sismos <- st_read(paste0(DIR_DATA, 'query-7jun24-25.json'))
View(sismos)

png(paste0(DIR_IMG, 'sismos1.png'))
plot(sismos)
dev.off()

## ============= Bounding box =============

# coordinates
minlon <- -81.738
maxlon <- -56.074
minlat <- -57.232
maxlat <- -16.636

# polygon
bbox_coords <- matrix(
  c(minlon, minlat,
    maxlon, minlat,
    maxlon, maxlat,
    minlon, maxlat,
    minlon, minlat),  # Close the polygon
  ncol = 2,
  byrow = TRUE
)

# sf polygon with WGS84 CRS
bbox_poly <- st_sf(
  geometry = st_sfc(st_polygon(list(bbox_coords))),
  crs = 4326
)

# save
st_write(bbox_poly, paste0(DIR_DATA, 'bbox_poly.shp'), driver = "ESRI Shapefile")


## ============= leaflet =============

leaflet(sismos) %>%
  addTiles() %>% 
  addCircleMarkers(radius=0.08) %>% 
  addPolygons(data=bbox_poly, fill = FALSE, color = 'black')
