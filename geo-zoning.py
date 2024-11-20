import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely import length, wkt
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Scattermapbox
from math import radians, sin, cos, sqrt, atan2
from typing import Tuple, List, Union
from dataclasses import dataclass
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
            color=gdf['name'],
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
    

def create_agent_map_with_legend(gdf, geojson_data, clustered_df, zone_name):
    """
    Create a map visualization with polygons and points, 
    colored by a categorical variable and with a legend.
    """
    try:
        # Find the center of the selected zone
        selected_zone = gdf[gdf['name'] == zone_name]
        if selected_zone.empty:
            raise ValueError(f"Zone '{zone_name}' not found in gdf['name']")
        
        center_lat = selected_zone.geometry.centroid.y.mean()
        center_lon = selected_zone.geometry.centroid.x.mean()

        # Create a color mapping for the 'Attached to' column
        unique_values = clustered_df['Attached to'].unique()
        color_palette = px.colors.qualitative.Vivid  # Choose a color palette
        color_map = {value: color_palette[i % len(color_palette)] for i, value in enumerate(unique_values)}

        # Create a base map with polygons
        fig = px.choropleth_mapbox(
            gdf,
            geojson=geojson_data,
            locations='name',
            color='name',
            featureidkey="properties.name",
            opacity=0.3,
            center={"lat": center_lat, "lon": center_lon},
            zoom=10,
            mapbox_style="carto-positron"
        )

        # Add points for each unique value in 'Attached to'
        for value in unique_values:
            filtered_df = clustered_df[clustered_df['Attached to'] == value]
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lat=filtered_df['Latitude'],
                lon=filtered_df['Longitude'],
                marker=dict(
                    size=8,
                    color=color_map[value],  # Assign color based on the map
                    opacity=1
                ),
                name=value  # Use this to show legend
            ))

        # Update layout to include legend
        fig.update_layout(
            legend=dict(
                title="Attached to",
                orientation="h",
                y=0.99,
                x=0.01
            ),
            height=600,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
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

######################### clustering/attaching section##########################
@dataclass
class ClusteringConfig:
    """Configuration parameters for K-means clustering"""
    n_clusters: int
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    random_state: int = 0
    earth_radius_km: float = 6371.0

class HaversineKMeans:
    """
    K-means clustering implementation using Haversine distance for geographical coordinates.
    
    This implementation clusters geographical points (latitude/longitude) using
    the Haversine distance metric instead of Euclidean distance.
    """
    
    def __init__(self, config: ClusteringConfig):
        """
        Initialize the clustering algorithm with configuration parameters.
        
        Args:
            config: ClusteringConfig object containing clustering parameters
        """
        self.config = config
        self.centroids = None
        np.random.seed(config.random_state)
    
    @staticmethod
    def haversine_distance(coord1: Tuple[float, float], 
                          coord2: Tuple[float, float], 
                          radius: float = 6371.0) -> float:
        """
        Calculate the Haversine distance between two geographical coordinates.
        
        Args:
            coord1: Tuple of (latitude, longitude) for first point
            coord2: Tuple of (latitude, longitude) for second point
            radius: Earth's radius in kilometers
            
        Returns:
            float: Distance between points in kilometers
        """
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return radius * c
    
    def _initialize_centroids(self, df: pd.DataFrame) -> np.ndarray:
        """
        Initialize cluster centroids by randomly selecting points.
        
        Args:
            df: DataFrame containing 'Latitude' and 'Longitude' columns
            
        Returns:
            np.ndarray: Initial centroids
        """
        return df[['Latitude', 'Longitude']].sample(
            n=self.config.n_clusters, 
            random_state=self.config.random_state
        ).to_numpy()
    
    def _assign_clusters(self, df: pd.DataFrame, 
                        centroids: np.ndarray) -> List[int]:
        """
        Assign each point to the nearest centroid using Haversine distance.
        
        Args:
            df: DataFrame containing points to be clustered
            centroids: Array of current centroid positions
            
        Returns:
            List[int]: Cluster assignments for each point
        """
        cluster_assignments = []
        
        for _, row in df.iterrows():
            point = (row['Latitude'], row['Longitude'])
            distances = [
                self.haversine_distance(point, tuple(centroid)) 
                for centroid in centroids
            ]
            cluster_assignments.append(np.argmin(distances))
            
        return cluster_assignments
    
    def _update_centroids(self, df: pd.DataFrame, 
                         cluster_assignments: List[int]) -> np.ndarray:
        """
        Update centroid positions based on mean position of assigned points.
        
        Args:
            df: DataFrame containing points
            cluster_assignments: Current cluster assignments
            
        Returns:
            np.ndarray: Updated centroid positions
        """
        new_centroids = []
        df = df.copy()
        df['cluster'] = cluster_assignments
        
        for cluster_id in range(self.config.n_clusters):
            cluster_points = df[df['cluster'] == cluster_id]
            
            if cluster_points.empty:
                # Reinitialize empty cluster with a random point
                new_centroid = tuple(
                    df[['Latitude', 'Longitude']].sample(1).to_numpy()[0]
                )
            else:
                new_centroid = (
                    cluster_points['Latitude'].mean(),
                    cluster_points['Longitude'].mean()
                )
            
            new_centroids.append(new_centroid)
        
        return np.array(new_centroids)
    
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform K-means clustering on the input data.
        
        Args:
            df: DataFrame containing 'Latitude' and 'Longitude' columns
            
        Returns:
            pd.DataFrame: Input DataFrame with added 'Attached to' column 
                         indicating cluster assignments
        """
        df = df.copy()
        self.centroids = self._initialize_centroids(df)
        
        for _ in range(self.config.max_iterations):
            # Assign points to clusters
            cluster_assignments = self._assign_clusters(df, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(df, cluster_assignments)
            
            # Check for convergence
            if np.allclose(
                self.centroids, 
                new_centroids, 
                atol=self.config.convergence_threshold
            ):
                break
                
            self.centroids = new_centroids
        
        # Add final cluster assignments to DataFrame
        df['cluster'] = cluster_assignments
        df['Attached to'] = df['cluster'].apply(lambda x: f"A{x+1}")
        df.drop(columns=['cluster'], inplace=True)
        
        return df

def cluster_locations(
    df: pd.DataFrame,
    n_clusters: int,
    max_iterations: int = 100,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Convenience function to cluster geographical locations.
    
    Args:
        df: DataFrame containing 'Latitude' and 'Longitude' columns
        n_clusters: Number of desired clusters
        max_iterations: Maximum number of iterations for convergence
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Input DataFrame with added 'Attached to' column
    """
    config = ClusteringConfig(
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        random_state=random_state
    )
    
    clusterer = HaversineKMeans(config)
    return clusterer.fit_predict(df)


def main():
    """Main function to run the Streamlit app."""
    st.title("Geo-Census Application")

    # Initialize session state variables at the beginning
    if 'step1_completed' not in st.session_state:
        st.session_state.step1_completed = False
    if 'polygons_df' not in st.session_state:
        st.session_state.polygons_df = None
    if 'census_df' not in st.session_state:
        st.session_state.census_df = None
    if 'geojson_data' not in st.session_state:
        st.session_state.geojson_data = None
    if 'gdf' not in st.session_state:
        st.session_state.gdf = None

    # Create tabs for clearer navigation
    tab1, tab2 = st.tabs(["Geo-Zoning", "Geo-Mapping"])

    with tab1:
        st.header("Geo-Zoning Step")

        # File upload section with error handling
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

        # Process files when both are uploaded
        if polygons_file and census_file:
            try:
                # Load polygons data
                polygons_file_type = 'csv' if polygons_file.name.endswith('.csv') else 'xlsx'
                st.session_state.polygons_df = load_data(polygons_file, polygons_file_type)
                
                # Load census data
                census_file_type = 'csv' if census_file.name.endswith('.csv') else 'xlsx'
                st.session_state.census_df = load_data(census_file, census_file_type)

                if st.session_state.polygons_df is not None and st.session_state.census_df is not None:
                    with st.spinner("Processing data..."):
                        # Process polygons data
                        if 'WKT' in st.session_state.polygons_df.columns:
                            st.session_state.polygons_df['polygon'] = st.session_state.polygons_df['WKT'].apply(clean_wkt)
                            st.session_state.polygons_df['value'] = st.session_state.polygons_df['polygon'].apply(lambda x: x.area if x and x.is_valid else 0)
                        else:
                            st.error("No WKT column found in the uploaded polygons data.")
                        
                        if 'nom' in st.session_state.polygons_df.columns:
                            st.session_state.polygons_df.rename(columns={'nom': 'name'}, inplace=True)
                        else:
                            st.error("No 'nom' column found in the uploaded polygons data.")
                        
                        # Create GeoJSON and GeoDataFrame
                        st.session_state.geojson_data = polygons_to_geojson(st.session_state.polygons_df)
                        if st.session_state.geojson_data:
                            st.session_state.gdf = gpd.GeoDataFrame.from_features(st.session_state.geojson_data['features'])
                            st.session_state.gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

                            # Generate Zones button
                            generate_zones = st.button("Generate Zones")
                            gif_placeholder = st.empty()

                            if generate_zones:
                                with st.spinner("Generating zones..."):
                                    gif_placeholder.markdown(
                                        """
                                         <div style="display: flex; justify-content: center;">
                                        <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjF2YzNsd2d5aG1rYTdtaTFldjRiYWNjZ2lhOXgyNnVvMnRvcjd4eiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5xuE75slgj1n3Wvxhs/giphy.gif" width="200">
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Perform zone determination
                                    st.session_state.census_df['Zone'] = st.session_state.census_df.apply(
                                        lambda x: determine_zone(
                                            x['Latitude'] if 'Latitude' in x else x['latitude'],
                                            x['Longitude'] if 'Longitude' in x else x['longitude'],
                                            st.session_state.polygons_df
                                        ), axis=1)

                                    # Display results
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
                                    fig = create_map(st.session_state.gdf, st.session_state.geojson_data, st.session_state.census_df)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)

                                    # Remove the GIF once processing is done
                                    gif_placeholder.empty()
                                    
                                    # Mark step 1 as completed
                                    st.session_state.step1_completed = True
                                    st.success("Geo-Zoning Step Completed! You can now proceed to Geo-Mapping.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.header("Geo-Mapping Step")

        # Check if Step 1 is completed before allowing Geo-Mapping
        if not st.session_state.step1_completed:
            st.warning("Please complete the Geo-Zoning Step first.")
        else:
            # Ensure session state variables exist for Geo-Mapping
            if "zone_select" not in st.session_state:
                st.session_state.zone_select = "Choose any zone..."
            if "num_agents" not in st.session_state:
                st.session_state.num_agents_select = "Select number of agents..."
            if "clustered_df" not in st.session_state:
                st.session_state.clustered_df = None

            # Modify zone selection and agent number input
            col1, col2 = st.columns([2, 1])

            with col1:
                zones = ['Choose any zone...'] + list(st.session_state.census_df['Zone'].unique())
                zone_select = st.selectbox(
                    "Select Zone", 
                    zones, 
                    index=zones.index(st.session_state.zone_select) if st.session_state.zone_select in zones else 0
                )
                st.session_state.zone_select = zone_select

            with col2:
                agents = list(range(1, 50))  # Start from 1 to avoid zero agents
                num_agents_select = st.selectbox(
                    "Select number of agents...", 
                    agents, 
                    index=agents.index(st.session_state.num_agents_select) if st.session_state.num_agents_select in agents else 0
                )
                st.session_state.num_agents_select = num_agents_select

            if st.session_state.zone_select != 'Choose any zone...' and st.session_state.num_agents_select != 'Select number of agents...':
                try:
                    gif_placeholder = st.empty()
                    generate_agents = st.button("Generate Agents")
                    if generate_agents:
                        with st.spinner("Generating agents..."):
                            gif_placeholder.markdown(
                                        """
                                         <div style="display: flex; justify-content: center;">
                                        <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExYjF2YzNsd2d5aG1rYTdtaTFldjRiYWNjZ2lhOXgyNnVvMnRvcjd4eiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5xuE75slgj1n3Wvxhs/giphy.gif" width="200">
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                    
                            zone_census_df = st.session_state.census_df[st.session_state.census_df['Zone'] == st.session_state.zone_select]
                            st.session_state.clustered_df = cluster_locations(zone_census_df, n_clusters=st.session_state.num_agents_select)
                            st.success("Clustering completed!")
                
                            # Show map and download button if clustering is done
                            if st.session_state.clustered_df is not None:
                                 # Display results
                                st.subheader("Agents Distribution")
                                zone_counts = st.session_state.clustered_df['Attached to'].value_counts()
                                fig = plot_zone_distribution(zone_counts)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.subheader("Mapping Data Preview")
                                st.dataframe(st.session_state.clustered_df.head())

                                # Download clustered data
                                excel_file = download_data(st.session_state.clustered_df)
                                st.download_button(
                                    "Download Full Data", 
                                    data=excel_file, 
                                    file_name="clustered_df.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

                                # Map visualization
                                st.subheader("Agents Map")
                                selected_gdf = st.session_state.gdf[st.session_state.gdf['name']==st.session_state.zone_select]
                                fig = create_agent_map_with_legend(selected_gdf, st.session_state.geojson_data, st.session_state.clustered_df, st.session_state.zone_select)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                gif_placeholder.empty()
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
