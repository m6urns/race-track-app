import os
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

# Create the Dash app
app = dash.Dash(__name__)

# Expose the Flask server for Gunicorn to use
server = app.server

# Define the app layout
app.layout = html.Div([
    html.H1('🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁'),
    html.Div([
        html.Div([
            html.Label('Select Track'),
            dcc.Dropdown(id='track-dropdown', options=[], value=None)
        ], style={'width': '300px', 'margin-right': '20px', 'display': 'inline-block'}),
        html.Div([
            html.Label('Color Map'),
            dcc.Dropdown(id='colormap-dropdown',
                         options=[{'label': c, 'value': c} for c in px.colors.named_colorscales()],
                         value='viridis')
        ], style={'width': '300px', 'display': 'inline-block'})
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Label('Min Speed (km/h)'),
            dcc.Slider(id='min-speed-slider', min=0, max=200, value=60, step=10)
        ], style={'width': '300px', 'margin-right': '20px', 'display': 'inline-block'}),
        html.Div([
            html.Label('Max Speed (km/h)'),
            dcc.Slider(id='max-speed-slider', min=0, max=300, value=200, step=10)
        ], style={'width': '300px', 'display': 'inline-block'})
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Label('Driver Aggressiveness'),
            dcc.Slider(id='aggressiveness-slider', min=0, max=1, value=0.5, step=0.1)
        ], style={'width': '300px', 'margin-right': '20px', 'display': 'inline-block'}),
        html.Div([
            html.Label('Smoothing Factor'),
            dcc.Slider(id='smoothing-slider', min=0, max=1, value=0.1, step=0.05)
        ], style={'width': '300px', 'display': 'inline-block'})
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.Label('Acceleration Modifier'),
            dcc.Slider(id='acceleration-slider', min=0.1, max=2, value=1, step=0.1)
        ], style={'width': '300px', 'margin-right': '20px', 'display': 'inline-block'}),
        html.Div([
            html.Label('Resolution'),
            dcc.Slider(id='resolution-slider', min=1, max=10, value=1, step=1,
                       marks={i: str(i) for i in range(1, 11)})
        ], style={'width': '300px', 'display': 'inline-block'})
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Label('Visualization Type'),
        dcc.Dropdown(id='visualization-dropdown',
                     options=[{'label': 'Acceleration', 'value': 'acceleration'},
                              {'label': 'Velocity', 'value': 'velocity'}],
                     value='velocity')
    ], style={'width': '300px', 'margin-bottom': '20px'}),
    html.Div([
        html.Label('Distance Range (m)'),
        dcc.RangeSlider(
            id='distance-range-slider',
            min=0,
            max=1,
            step=0.01,
            value=[0, 1],
            marks={i/10: f'{i/10:.1f}' for i in range(0, 11)}
        )
    ], style={'width': '600px', 'margin-bottom': '20px'}),
    html.Div([
        html.Button('Export CSV', id='export-button', n_clicks=0)
    ], style={'margin-bottom': '20px'}),
    dcc.Graph(id='track-graph')
])

# Callback to update the track dropdown options
@app.callback(Output('track-dropdown', 'options'),
              [Input('track-dropdown', 'value')])
def update_track_dropdown(selected_track):
    # Get the list of track files from the folder
    track_folder = './tracks'
    track_files = [f for f in os.listdir(track_folder) if f.endswith('.csv')]

    # Create dropdown options
    dropdown_options = [{'label': track_file, 'value': track_file} for track_file in track_files]

    return dropdown_options

# Callback to update the graph based on selected track and slider values
@app.callback(Output('track-graph', 'figure'),
              [Input('track-dropdown', 'value'),
               Input('min-speed-slider', 'value'),
               Input('max-speed-slider', 'value'),
               Input('aggressiveness-slider', 'value'),
               Input('smoothing-slider', 'value'),
               Input('acceleration-slider', 'value'),
               Input('colormap-dropdown', 'value'),
               Input('visualization-dropdown', 'value'),
               Input('distance-range-slider', 'value'),
               Input('resolution-slider', 'value')])
def update_graph(selected_track, min_speed, max_speed, aggressiveness, smoothing_factor, acceleration_modifier,
                 colormap, visualization_type, distance_range, resolution):
    if selected_track is None:
        return go.Figure()

    # Read the selected track data from the CSV file
    track_folder = './tracks'
    track_data = np.genfromtxt(os.path.join(track_folder, selected_track), delimiter=',', skip_header=1)

    # Extract the center line coordinates and track widths
    x_center = track_data[:, 0]
    y_center = track_data[:, 1]
    w_tr_right = track_data[:, 2]
    w_tr_left = track_data[:, 3]

    # Calculate the track boundaries
    x_right = x_center + w_tr_right
    x_left = x_center - w_tr_left
    y_right = y_center
    y_left = y_center

    # Calculate the distances between each pair of points
    distances = np.sqrt(np.diff(x_center)**2 + np.diff(y_center)**2)

    # Calculate the cumulative distances along the track
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

    # Normalize the cumulative distances to a range of 0 to 1
    normalized_distances = cumulative_distances / cumulative_distances[-1]

    # Calculate the angles between each pair of points
    angles = np.arctan2(np.diff(y_center), np.diff(x_center))

    # Calculate the radius of curvature at each point
    curvatures = np.abs(np.diff(angles) / distances[:-1])

    speeds = [0]  # Initial speed (starting from rest)
    accelerations = []
    velocities = []
    acceleration = 10 * acceleration_modifier  # Initial acceleration

    for i in range(len(distances)):
        distance = distances[i]
        curvature = curvatures[i] if i < len(curvatures) else curvatures[-1]

        # Adjust acceleration and deceleration based on track curvature and driver aggressiveness
        if curvature > 0.1:  # High curvature (sharp turn)
            target_speed = min_speed
            acceleration = -30 * (1 - aggressiveness) * acceleration_modifier
        elif curvature < 0.02:  # Low curvature (straight)
            target_speed = max_speed
            acceleration = 20 * aggressiveness * acceleration_modifier
        else:  # Medium curvature (gradual turn)
            target_speed = (min_speed + max_speed) / 2
            acceleration = -10 * (1 - aggressiveness) * acceleration_modifier

        # Calculate the new speed based on the target speed and acceleration
        new_speed = speeds[-1] + acceleration * distance
        new_speed = max(min(new_speed, target_speed), 0)

        # Apply smoothing to the speed changes
        smoothed_speed = speeds[-1] * (1 - smoothing_factor) + new_speed * smoothing_factor

        speeds.append(smoothed_speed)

        # Calculate acceleration and velocity
        acceleration = (speeds[-1] - speeds[-2]) / distance if i > 0 else acceleration
        velocity = speeds[-1]

        accelerations.append(acceleration)
        velocities.append(velocity)

    # Filter the track data based on the selected distance range
    start_distance, end_distance = distance_range
    filtered_indices = np.where((normalized_distances >= start_distance) & (normalized_distances <= end_distance))[0]

    # Extract the filtered track data
    x_center_filtered = x_center[filtered_indices]
    y_center_filtered = y_center[filtered_indices]
    w_tr_right_filtered = w_tr_right[filtered_indices]
    w_tr_left_filtered = w_tr_left[filtered_indices]

    # Calculate the track boundaries for the filtered data
    x_right_filtered = x_center_filtered + w_tr_right_filtered
    x_left_filtered = x_center_filtered - w_tr_left_filtered
    y_right_filtered = y_center_filtered
    y_left_filtered = y_center_filtered

    # Extract the filtered acceleration and velocity data
    accelerations_filtered = [accelerations[i] for i in filtered_indices[:-1]]
    velocities_filtered = [velocities[i] for i in filtered_indices[:-1]]

    # Apply resolution modifier
    resolution_indices = np.arange(0, len(x_center_filtered[:-1]), resolution)
    x_center_resolved = x_center_filtered[:-1][resolution_indices]
    y_center_resolved = y_center_filtered[:-1][resolution_indices]
    accelerations_resolved = [accelerations_filtered[i] for i in resolution_indices]
    velocities_resolved = [velocities_filtered[i] for i in resolution_indices]

    # Determine the appropriate colorbar title based on the selected visualization type
    colorbar_title = ''
    if visualization_type == 'acceleration':
        colorbar_title = 'Acceleration (m/s²)'
        color_values = accelerations_resolved
    elif visualization_type == 'velocity':
        colorbar_title = 'Velocity (km/h)'
        color_values = velocities_resolved

    fig = go.Figure()

    # Plot the track boundaries for the selected distance range
    fig.add_trace(go.Scatter(x=x_right_filtered, y=y_right_filtered, mode='lines', line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=x_left_filtered, y=y_left_filtered, mode='lines', line=dict(color='black', width=2), showlegend=False))

    # Plot the center line with the selected visualization type and resolution
    fig.add_trace(go.Scatter(x=x_center_resolved, y=y_center_resolved, mode='markers',
                             marker=dict(color=color_values, colorscale=colormap, size=5,
                                         colorbar=dict(title=colorbar_title)), showlegend=False))

    fig.update_layout(title=f'Track: {selected_track}',
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      hovermode='closest',
                      height=600,
                      width=800)

    return fig

# Callback to handle CSV export
@app.callback(Output('export-button', 'n_clicks'),
              [Input('export-button', 'n_clicks')],
              [State('track-dropdown', 'value'),
               State('distance-range-slider', 'value'),
               State('track-graph', 'figure')])
def export_csv(n_clicks, selected_track, distance_range, figure):
    if n_clicks > 0:
        # Extract the filtered track data and acceleration/velocity values
        x_center_filtered = figure['data'][2]['x']
        y_center_filtered = figure['data'][2]['y']
        accelerations_filtered = figure['data'][2]['marker']['color']
        velocities_filtered = [v * 3.6 for v in figure['data'][2]['marker']['color']]  # Convert m/s to km/h

        # Create a DataFrame with the exported data
        export_data = pd.DataFrame({
            'x': x_center_filtered,
            'y': y_center_filtered,
            'acceleration': accelerations_filtered,
            'velocity': velocities_filtered
        })

        # Generate the CSV file name
        start_distance, end_distance = distance_range
        csv_filename = f"{selected_track.split('.')[0]}_distance_{start_distance:.2f}_{end_distance:.2f}.csv"

        # Save the DataFrame to a CSV file
        export_data.to_csv(csv_filename, index=False)

    return n_clicks

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)