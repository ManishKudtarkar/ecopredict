import geopandas as gpd

def merge_predictions(shapefile, predictions_df):
    geo = gpd.read_file(shapefile)
    return geo.merge(predictions_df, on="region")
