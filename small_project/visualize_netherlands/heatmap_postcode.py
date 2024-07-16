import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import fiona

dpkg = "/home/bugger/Documents/Thuis/wockyvakantie/gadm41_NLD.gpkg"
layers = fiona.listlayers(dpkg)
print(layers)
mapdf = gpd.read_file(dpkg)

d_shape = '/home/bugger/Documents/Thuis/wockyvakantie/CBS-PC4-2020-v1/CBS_pc4_2020_v1.shp'
d_shape = '/path/to/shape/file'
layer_name = fiona.listlayers(d_shape)[0]
mapdf = gpd.read_file(d_shape, layer=layer_name)
mapdf.plot(column='type hier uw colom naam en sluit af met een hekje')

[mapdf[x].value_counts() for x in mapdf.columns]

result = pd.read_csv('/home/bugger/Documents/Thuis/wockyvakantie/NAW.csv/Untitled form.csv')
postcodes = [int(x[:4]) for x in result['Postcode'].values if x[:4].isdigit()]
cities = ["Amsterdam", "Delft", "Delft", "Delft", "Delft", "Delft", "Rotterdam", "Lisse", "Eindhoven", "Rotterdam",
          "Delft", "Delft", "Pijnacker", "Houten", "Rijswijk", "Amsterdam", "Den Bosch", "Den Haag"]
import collections
pd_cities = pd.DataFrame.from_dict([dict(collections.Counter(postcodes))]).T.reset_index()
mapdf.columns
mapdf_merge = pd.merge(mapdf, pd_cities, left_on='PC4', right_on='index', how='outer')
mapdf_merge[0] = mapdf_merge[0].fillna(0)
mapdf_merge.plot(column=0, figsize=(10, 10), legend=True)
