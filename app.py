import os
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px  # Import plotly.express for color scales

# Create the Dash app
app = dash.Dash(__name__)

# Expose the Flask server for Gunicorn to use
server = app.server

# Define the app layout
app.layout = html.Div([
    html.H1('Race Track Visualization'),
    html.Div([
        html.Label('Select Track'),
        dcc.Dropdown(id='track-dropdown', options=[], value=None)
    ], style={'width': '300px', 'margin-bottom': '20px'}),
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
        html.Label('Acceleration Modifier'),
        dcc.Slider(id='acceleration-slider', min=0.1, max=2, value=1, step=0.1)
    ], style={'width': '300px', 'margin-bottom': '20px'}),
    html.Div([
        html.Label('Color Map'),
        dcc.Dropdown(id='colormap-dropdown', 
                     options=[{'label': c, 'value': c} for c in px.colors.named_colorscales()],
                     value='viridis')
    ], style={'width': '300px', 'margin-bottom': '20px'}),
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
               Input('colormap-dropdown', 'value')])
def update_graph(selected_track, min_speed, max_speed, aggressiveness, smoothing_factor, acceleration_modifier, colormap):
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
    
    # Calculate the angles between each pair of points
    angles = np.arctan2(np.diff(y_center), np.diff(x_center))
    
    # Calculate the radius of curvature at each point
    curvatures = np.abs(np.diff(angles) / distances[:-1])  # Adjust the length of distances
    
    speeds = [0]  # Initial speed (starting from rest)
    acceleration = 10 * acceleration_modifier  # Initial acceleration

    for i in range(len(distances)):
        distance = distances[i]
        curvature = curvatures[i] if i < len(curvatures) else curvatures[-1]  # Handle the last point
        
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

    # Decelerate to a stop at the end of the track
    deceleration = -10 * acceleration_modifier
    while speeds[-1] > 0:
        new_speed = speeds[-1] + deceleration
        new_speed = max(new_speed, 0)
        speeds.append(new_speed)
    
    fig = go.Figure()
    
    # Plot the track boundaries
    fig.add_trace(go.Scatter(x=x_right, y=y_right, mode='lines', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=x_left, y=y_left, mode='lines', line=dict(color='black', width=2)))
    
    # Plot the center line with speed colors
    fig.add_trace(go.Scatter(x=x_center, y=y_center, mode='markers', marker=dict(color=speeds, colorscale=colormap, size=5)))
    
    fig.update_layout(title=f'Track: {selected_track}',
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      hovermode='closest',
                      coloraxis=dict(colorbar=dict(title='Speed (km/h)')),
                      height=600,
                      width=800)
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)