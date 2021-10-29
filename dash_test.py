import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from numpy.lib.function_base import disp
import plotly.express as px
import os
import pandas as pd
import numpy as np
from skimage import io
import math

df = pd.read_csv(r'D:\Konrad\TCM_scan\dash\pwr_a_1_20210930_100324.csv')

default_clickeData = ''

print(df[:3]) #show few data
# Preparing values for axis markers
available_x_indicators = df['w_id'].unique()

displayed_x_indicators = available_x_indicators[0:len(available_x_indicators)] 
available_y_indicators = df['l_id'].unique()
available_y_indicators.sort()
displayed_y_indicators = available_y_indicators[0:len(available_y_indicators)] 

available_dropdown_indicators = ['Orginalny', 'Wyodrebniony', 'Segmentacja']
SUBFOLDERS = [r'\images', r'\otsu_tooth', r'\otsu_tooth']

available_color_indicators = ['Długość', 'Szerokość', 'Położenie środka - długość', 'Położenie środka - szerokość', 'Pole wyodrębnionej części']
COLORS = ['l', 'w','c_l', 'c_w', 'l']

print(displayed_x_indicators)
print(displayed_y_indicators)

app = dash.Dash(__name__)

app.layout = html.Div([
    # Title
    html.Div([
        html.H1('Zużycie zębów przeciągacza'),
        ],style={'textAlign': 'center'}),
    # Scatter plot with slider
    html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.RangeSlider(
            id='position-slider',
            min=df['w_id'].min(),
            max=df['w_id'].max(),
            value=[0,df['w_id'].mean()],
            marks={str(x): str(x) for x in displayed_x_indicators},
            step=6 )
        ],style={'margin': '20px'}),
    # Options
    html.Div([  
        html.Div([
            html.H1('Opcje'),
            html.H2('Wartość na wykresie:'),
            dcc.Dropdown(
                id='img_color',
                options=[{'label': i, 'value': i} for i in available_color_indicators],
                value = available_color_indicators[1]),
            html.H2('Aktywne typ wad:'),
            dcc.Checklist(
                options=[
                    {'label': 'New York City', 'value': 'NYC'},
                    {'label': u'Montréal', 'value': 'MTL'},
                    {'label': 'San Francisco', 'value': 'SF'}
                    ],value=['MTL', 'SF']),
            ],style={'width': '40%','float': 'left', 'display': 'inline-block'})
        ],style={'width': '32%','float': 'left', 'display': 'inline-block','margin-left': '50px'}), 
    # Image 1
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='image1d',
                options=[{'label': i, 'value': i} for i in available_dropdown_indicators],
                value = available_dropdown_indicators[0]),
            ],style={'width': '85%','float': 'left', 'display': 'inline-block','margin-left': '50px','margin-right': '50px'}),
        html.Br(),
        html.Div([
            html.H2(id='title1')
            ],style={'textAlign': 'center'}),
        dcc.Graph(id='image1g'),
        ],style={'width': '32%', 'display': 'inline-block'}),  
    #Image 2
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='image2d',
                options=[{'label': i, 'value': i} for i in available_dropdown_indicators],
                value = available_dropdown_indicators[1]),
            ],style={'width': '85%','float': 'left', 'display': 'inline-block','margin-left': '50px','margin-right': '50px'}),
        html.Br(),
        html.Div([
            html.H2(id='title2')
            ],style={'textAlign': 'center'}),
        dcc.Graph(id='image2g'), 
        ],style={'width': '32%','float': 'right', 'display': 'inline-block'})         
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('position-slider', 'value'),
     Input('img_color', 'value')]
    )
def update_scatter(value,categoryPick):
    filtered_df = df[ (df['w_id'] <= value[1]) & (df['w_id'] >= value[0]) ]
    c = COLORS[ available_color_indicators.index(categoryPick)]
    print(c)
    fig = px.scatter(filtered_df, 
                    x = "w_id", 
                    y = "l_id", 
                    color = c,
                    hover_name = "img_name",
                    log_x = False, 
                    size_max = 150,)
    fig.update_layout(transition_duration = 200, 
                    height = 500, template="seaborn",
                    yaxis=dict(title_text="WIDTH [mm]", titlefont=dict(size=20), tickvals = displayed_y_indicators),
                    xaxis=dict(title_text="LENGHT [mm]", titlefont=dict(size=20), tickvals = displayed_x_indicators),
                    margin=dict(l = 20, r = 20, t = 5, b = 30))
    fig.update_traces(marker=dict(size=25,
                      line=dict(width=2, color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    return fig


@app.callback(
    [Output('image1g', 'figure'),
     Output('title1', 'children')],
    [Input('graph-with-slider', 'clickData'),
     Input('image1d','value')]
    )
def update_image(clickData,value):

    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    DATA_FOLDER_PATH =  r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_a_1_20210930_100324' + SUBFOLDER
    IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    FULL_PATH = DATA_FOLDER_PATH + '\\' + IMAGE_NAME 
    print(SUBFOLDER,IMAGE_NAME)
    img = io.imread(FULL_PATH)
    fig = px.imshow(img)
    fig.update_layout(template="none",xaxis=dict(showgrid=False, showline=False,visible=False),yaxis=dict(showgrid=False, showline=False, visible=False))
    return fig, IMAGE_NAME

@app.callback(
    [Output('image2g', 'figure'),
     Output('title2', 'children')],
    [Input('graph-with-slider', 'clickData'),
     Input('image2d','value')]
    )
def update_image(clickData,value):
    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    DATA_FOLDER_PATH =  r'D:\Konrad\TCM_scan\Skany_nowe_pwr\pwr_a_1_20210930_100324' + SUBFOLDER
    IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    FULL_PATH = DATA_FOLDER_PATH + '\\' + IMAGE_NAME 
    img = io.imread(FULL_PATH)
    fig = px.imshow(img)
    fig.update_layout(template="none",xaxis=dict(showgrid=False, showline=False,visible=False),yaxis=dict(showgrid=False, showline=False, visible=False))
    return fig, IMAGE_NAME



if __name__ == '__main__':
    app.run_server(debug=False)
