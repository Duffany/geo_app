import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely import length, wkt
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Scattermapbox
import json
import io
import re
import os
from io import StringIO

# Increase Streamlit's file size limit and add caching
st.set_page_config(
    layout="wide", 
    page_title="Zoning Analysis App",
    initial_sidebar_state="collapsed"
)

# Add session state to store uploaded data
if 'polygons_df' not in st.session_state:
    st.session_state['polygons_df'] = None
if 'census_df' not in st.session_state:
    st.session_state['census_df'] = None

def load_data(file, file_type):
    """Load data from uploaded file without caching."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def clean_wkt(input_str):
    """Clean WKT string and convert to polygon coordinates."""
    try:
        # Use shapely's wkt.loads to convert WKT to Polygon
        return wkt.loads(input_str)
    except Exception as e:
        st.error(f"Error processing WKT: {str(e)}")
        return None

def polygons_to_geojson(df):
    """Convert DataFrame polygons to GeoJSON format."""
    try:
        features = []
        for _, row in df.iterrows():
            if isinstance(row['polygon'], Polygon):  # Ensure there's a valid polygon
                feature = {
                    "type": "Feature",
                    "geometry": json.loads(gpd.GeoSeries([row['polygon']]).to_json())['features'][0]['geometry'],
                    "properties": {"name": row['name'], "value": row['value']}
                }
                features.append(feature)
        return {"type": "FeatureCollection", "features": features}
    except Exception as e:
        st.error(f"Error converting to GeoJSON: {str(e)}")
        return None

def determine_zone(lat, lon, zoning_df):
    """Determine which zone a point belongs to."""
    try:
        point = Point(lon, lat)
        for i in range(len(zoning_df)):
            polygon = zoning_df.iloc[i]['polygon']
            if polygon.contains(point):
                return zoning_df.iloc[i]['name']
        return "Hors Zone"
    except Exception as e:
        st.error(f"Error determining zone: {str(e)}")
        return "Error"

def plot_zone_distribution(zone_counts):
    """Create a beautiful bar chart with counts for each zone."""
    try:
        # Create the bar chart with Plotly
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=zone_counts.index,  # Zone names
            y=zone_counts.values,  # Counts for each zone
            text=zone_counts.values,  # Show the count as text on the bars
            textposition='auto',  # Position the text automatically
            marker=dict(color='rgb(102, 194, 255)', line=dict(color='rgb(8,48,107)', width=2)),  # Custom color
        ))

        # Update layout for a more polished appearance
        fig.update_layout(
            xaxis_title="Zone",
            yaxis_title="Number of Census Points",
            plot_bgcolor="rgba(255, 255, 255, 0)",  # Transparent background
            paper_bgcolor="rgba(255, 255, 255, 0)",  # Light gray background
            margin=dict(l=50, r=50, t=50, b=50),  # Margin adjustment for spacing
            template="plotly_dark",  # Dark theme for contrast
            showlegend=False,  # Hide legend (optional)
            height=400  # Set chart height
        )

        return fig
    except Exception as e:
        st.error(f"Error creating the enhanced chart: {str(e)}")
        return None

def create_map(gdf, geojson_data, census_data=None):
    """Create a map visualization with polygons and points."""
    try:
        color_discrete_map = {
            'Else': 'rgb(224, 224, 224)',
            'Batch 1': 'rgb(255, 181, 181)',
            'Batch 2': 'rgb(212, 245, 154)'
        }

        fig = px.choropleth_mapbox(
            gdf,
            geojson=geojson_data,
            locations=gdf['name'],
            color_discrete_map=color_discrete_map,
            opacity=0.6,
            featureidkey="properties.name",
            center={"lat": gdf.geometry.centroid.y.mean(), 
                    "lon": gdf.geometry.centroid.x.mean()},
            zoom=10,
            mapbox_style="carto-positron"
        )

        if census_data is not None:
            fig.add_trace(Scattermapbox(
                mode='markers',
                lat=census_data['Latitude'],
                lon=census_data['Longitude'],
                marker=dict(
                    size=6,
                    color='Blue',
                    opacity=0.7
                ),
                name='Census Points'
            ))

        fig.update_layout(
            height=600,
            width=None,
            margin={"r":0,"t":0,"l":0,"b":0}
        )

        return fig
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None
    
def download_data(census_df):
    """Generate an Excel file from the census data."""
    buffer = io.BytesIO()
    census_df.to_excel(buffer, index=False, engine='xlsxwriter')
    return buffer.getvalue()

def main():
    """Main function to run the Streamlit app."""
    st.title("Geo-Zoning Application")

    # File upload section with error handling
    # st.header("1. Data Upload")
    col1, col2 = st.columns(2)

    with col1:
        polygons_file = st.file_uploader(
            "Upload Polygons Data (CSV/Excel)", 
            type=['csv', 'xlsx'],
            key='polygons_uploader'
        )

    with col2:
        census_file = st.file_uploader(
            "Upload Census Data (CSV/Excel)", 
            type=['csv', 'xlsx'],
            key='census_uploader'
        )

    if polygons_file and census_file:
        try:
            # Determine file type and load polygons data
            file_type = 'csv' if polygons_file.name.endswith('.csv') else 'xlsx'
            st.session_state.polygons_df = load_data(polygons_file, file_type)
            
            # Determine file type and load census data
            file_type = 'csv' if census_file.name.endswith('.csv') else 'xlsx'
            st.session_state.census_df = load_data(census_file, file_type)

            if st.session_state.polygons_df is not None and st.session_state.census_df is not None:
                # st.header("2. Data Processing")
                with st.spinner("Processing data..."):
                    # Process polygons
                    if 'WKT' in st.session_state.polygons_df.columns:
                        # Clean and convert WKT to valid Polygon objects
                        st.session_state.polygons_df['polygon'] = st.session_state.polygons_df['WKT'].apply(clean_wkt)
                        st.session_state.polygons_df['value'] = st.session_state.polygons_df['polygon'].apply(lambda x: x.area if x and x.is_valid else 0)  # Optional: Area as value
                    else:
                        st.error("No WKT column found in the uploaded polygons data.")
                    if 'nom' in st.session_state.polygons_df.columns:
                        st.session_state.polygons_df.rename(columns={'nom': 'name'}, inplace=True)
                    else:
                        st.error("No 'nom' column found in the uploaded polygons data.")
                    # Create GeoJSON
                    geojson_data = polygons_to_geojson(st.session_state.polygons_df)
                    if geojson_data:
                        # Create a GeoDataFrame with valid geometries
                        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])
                        gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

                        generate_zones = st.button("Generate Zones")

                        # Placeholder for the GIF
                        gif_placeholder = st.empty()

                        if generate_zones:
                            with st.spinner("Generating zones..."):
                        # Display the GIF while the spinner is active
                                gif_placeholder.markdown(
                                            """
                                             <div style="display: flex; justify-content: center;">
                                            <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjF2YzNsd2d5aG1rYTdtaTFldjRiYWNjZ2lhOXgyNnVvMnRvcjd4eiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5xuE75slgj1n3Wvxhs/giphy.gif" width="200">
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                                )
                                # Perform the zone determination process
                                st.session_state.census_df['Zone'] = st.session_state.census_df.apply(
                                lambda x: determine_zone(
                                x['Latitude'] if 'Latitude' in x else x['latitude'],
                                x['Longitude'] if 'Longitude' in x else x['longitude'],
                                st.session_state.polygons_df
                                 ), axis=1)

                                # Display results
                                # st.header("3. Results")
                                st.subheader("Zone Distribution")
                                zone_counts = st.session_state.census_df['Zone'].value_counts()
                                fig = plot_zone_distribution(zone_counts)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)

                                # Display data preview
                                st.subheader("Data Preview")
                                st.dataframe(st.session_state.census_df.head())

                                # Add download button
                                excel_file = download_data(st.session_state.census_df)
                                st.download_button("Download Full Data", data=excel_file, file_name="census_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                                # Create and display map
                                st.subheader("Zoning Map")
                                fig = create_map(gdf, geojson_data, st.session_state.census_df)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                # Once the spinner is gone, the GIF will also be removed
                                gif_placeholder.empty()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload both Polygons and Census data files to begin analysis.")

if __name__ == "__main__":
    main()
