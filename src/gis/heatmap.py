import folium

def generate_heatmap(geo_df, output_path):
    m = folium.Map(location=[19.0, 72.8], zoom_start=6)

    folium.Choropleth(
        geo_data=geo_df,
        data=geo_df,
        columns=["region", "risk_score"],
        key_on="feature.properties.region",
        fill_color="YlOrRd"
    ).add_to(m)

    m.save(output_path)
