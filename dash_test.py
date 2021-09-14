import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from numpy.lib.function_base import disp
import plotly.express as px
import os
import pandas as pd
import numpy as np
from skimage import io

df = pd.read_csv(r'C:\Users\Konrad\tcm_scan\20210621_092043.csv')
print(df[:3]) #show few data
# Preparing values for axis markers
available_x_indicators = df['x'].unique()
displayed_x_indicators = available_x_indicators[available_x_indicators % 28==0] 
available_y_indicators = df['y'].unique()
available_y_indicators.sort()
displayed_y_indicators = available_y_indicators[0:len(available_y_indicators):5] 
available_dropdown_indicators = ['orginalny','progowanie','3d']

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H1('Zużycie zębów przeciągacza'),
    ],style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.RangeSlider(
            id='position-slider',
            min=df['x'].min(),
            max=df['x'].max(),
            value=[0,df['x'].mean()],
            marks={str(x): str(x) for x in displayed_x_indicators},
            step=7
        )
    ]),
    html.Div([
        html.H2('Options')
    ],style={'width': '33%','float': 'left', 'display': 'inline-block'}), 
    html.Div([
        dcc.Dropdown(
            id='image1d',
            options=[{'label': i, 'value': i} for i in available_dropdown_indicators],
            value='Life expectancy at birth, total (years)'),
        dcc.Graph(id='image1g')
    ],style={'width': '33%', 'display': 'inline-block'}),  
    html.Div([
        dcc.Dropdown(
            id='image2d',
            options=[{'label': i, 'value': i} for i in available_dropdown_indicators],
            value='Life expectancy at birth, total (years)'),
        dcc.Graph(id='image2g')
    ],style={'width': '33%','float': 'right', 'display': 'inline-block'})         
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('position-slider', 'value')]
)
def update_scatter(value):
    filtered_df = df[ (df['x']<= value[1]) & (df['x']>= value[0]) ]
    fig = px.scatter(filtered_df, x="x", y="y",
                    color="color", hover_name="img_name",
                    log_x=False, size_max=80)
    fig.update_layout(transition_duration=200, height=500, template="seaborn",
                    yaxis=dict(title_text="Y [mm]", titlefont=dict(size=20), tickvals = displayed_y_indicators),
                    xaxis=dict(title_text="X [mm]", titlefont=dict(size=20), tickvals = displayed_x_indicators),
                    margin=dict(l=20, r=20, t=5, b=30))
    return fig


@app.callback(
    dash.dependencies.Output('image1g', 'figure'),
    [dash.dependencies.Input('graph-with-slider', 'clickData')]
)
def update_image(clickData):
    DATA_FOLDER_PATH =  r'C:\Users\Konrad\tcm_scan\20210621_092043'
    IMAGE_NAME = str(clickData['points'][0]['hovertext'])
   
    FULL_PATH = DATA_FOLDER_PATH + '\\' + IMAGE_NAME 
    print(FULL_PATH)
    img = io.imread(FULL_PATH)
    fig = px.imshow(img)
    return fig

@app.callback(
    Output('image2g', 'figure'),
    [Input('image2d', 'value')]
)
def update_image(value):
    DATA_FOLDER_PATH =  r'C:\Users\Konrad\tcm_scan\20210621_092043'
    IMAGE_NAME = '000_012.png'
    FULL_PATH = DATA_FOLDER_PATH + '\\' + IMAGE_NAME 
    img = io.imread(FULL_PATH)
    fig = px.imshow(img)
    return fig


if __name__ == '__main__':
    app.run_server(debug=False)
