import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from io import BytesIO
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from math import radians, sin, cos, sqrt, atan2
import openrouteservice
from openrouteservice import convert

# OpenRouteService API key (Replace with your actual key)
ORS_API_KEY = '5b3ce3597851110001cf6248e9f78b332d824b60a89a0ee44fe6f77c'

# Haversine distance function
def haversine_distance(coord1, coord2):
    """Calculate the Haversine distance between two points on the Earth."""
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

def create_data_model(df):
    """Create data model for OR-Tools using Haversine distance."""
    coords = df[['latitude', 'longitude']].values
    n = len(coords)
    
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = haversine_distance(coords[i], coords[j])* 10000
    
    data = {
        'distance_matrix': distances.tolist(),
        'num_vehicles': 1,
        'depot': 0
    }
    return data

def solve_tsp(data):
    """Solve the Traveling Salesman Problem using OR-Tools without returning to the start."""
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                             data['num_vehicles'],
                                             data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['distance_matrix'][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES)

    # Disallow the return to the start point
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    search_parameters.solution_limit = 10000  # Adjust as needed

    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        index = routing.Start(0)
        plan_output = []
        while not routing.IsEnd(index):
            plan_output.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return plan_output
    else:
        return []

def generate_excel_file(df, order):
    """Generate an Excel file with coordinates, order, and names."""
    df.reset_index(drop=True, inplace=True)
    df['order'] = [order.index(i) + 1 for i in df.index]
    df_sorted = df.loc[df.index[df.index.isin(order)], ['order', 'name', 'latitude', 'longitude']]
    
    

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_sorted.to_excel(writer, index=False, sheet_name='Route')
    
    

    output.seek(0)
    return output.getvalue()

def plot_route_on_map(df, order):
    """Plot the route on a Folium map using OpenRouteService."""
    # Create a map centered on the starting point
    map_center = df.iloc[order[0]][['latitude', 'longitude']].values
    map_ = folium.Map(location=map_center, zoom_start=10)
    
    # Get coordinates for the route
    route_coords = df[['latitude', 'longitude']].iloc[order].values
    
    # Initialize ORS client
    client = openrouteservice.Client(key=ORS_API_KEY)
    
    # Request the route from ORS
    coords = [(lon, lat) for lat, lon in route_coords]  # ORS expects (lon, lat)
    route = client.directions(coordinates=coords, profile='driving-car', format='geojson')
    
    # Plot route on the map
    route_geojson = route['features'][0]['geometry']['coordinates']
    route_latlng = [(lat, lon) for lon, lat in route_geojson]
    folium.PolyLine(route_latlng, color='blue', weight=2.5, opacity=1).add_to(map_)
    
    # Add markers for each point
    for i, idx in enumerate(order):
        lat, lon = df.iloc[idx][['latitude', 'longitude']]
        folium.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12pt; color: red; background-color: white; border-radius: 50%; text-align: center; width: 30px; height: 30px; line-height: 30px;">{i + 1}</div>'
            )
        ).add_to(map_)
    
    return map_

def main():
    st.title("Optimal Route Finder")
    
    st.sidebar.header("Upload Excel File")
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        if 'latitude' in df.columns and 'longitude' in df.columns and 'name' in df.columns:
            st.write("Data from uploaded file:")
            st.dataframe(df)
            
            # Create data model
            data = create_data_model(df)
            
            # Solve TSP to get the optimal route
            order = solve_tsp(data)
            
            if order:
                # Plot route
                map_ = plot_route_on_map(df, order)
                
                # Display map
                folium_static(map_)
                
                # save map
                map_.save("map.html")

                # Generate and download Excel file
                excel_file = generate_excel_file(df, order)
                st.download_button(
                    label="Download Route Information",
                    data=excel_file,
                    file_name="route_information.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.write("Route generated successfully!")
            else:
                st.error("Unable to find a solution to the Traveling Salesman Problem.")
        else:
            st.error("Excel file must contain 'latitude', 'longitude', and 'name' columns.")
    
if __name__ == "__main__":
    main()
