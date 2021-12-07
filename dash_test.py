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
import cv2 as cv


BROACH_CSV = r'D:\Konrad\TCM_scan\dash' # Path to the folder with .csv files
BROACH_DIR = r'D:\Konrad\TCM_scan\dash_skany'  # Path to the corresponding folders with images 
BROACH_LIST = os.listdir(BROACH_DIR)
df = pd.read_csv(BROACH_CSV + '\\' + BROACH_LIST[0] + '.csv')  # Deafoult broach
print(df[:3]) # Show few data in console

# Preparing values for axis markers of the main scatter
available_x_indicators = df['w_id'].unique()
displayed_x_indicators = available_x_indicators[0:len(available_x_indicators)] 
available_y_indicators = df['l_id'].unique()
available_y_indicators.sort()
displayed_y_indicators = available_y_indicators[0:len(available_y_indicators)] 

# Available preview modes
available_dropdown_indicators = ['Orginalny', 'Wyodrebniony', 'Segmentacja']
SUBFOLDERS = [r'\images', r'\otsu_tooth', r'\otsu_tooth']

# Available heatmap modes
available_color_indicators = ['Długość', 'Szerokość', 'Położenie środka - długość', 'Położenie środka - szerokość', 'Ilość wad','Stępienie','Narost','Zatarcie','Wykruszenie','Stępienie w rzedach','Wielkość stępienia']
COLORS = ['l', 'w','c_l', 'c_w', 'inst_num','stepienie','narost','zatarcie','wykruszenie','stepienie_w_rzedach','wielkosc_stepienia']

# Available broaches
available_broach_indicators = BROACH_LIST

app = dash.Dash(__name__)

# Layout structure
app.layout = html.Div([
    # Title
    html.Div([
        html.H1('Zużycie zębów przeciągacza'),
        ],style={'textAlign': 'center'}),
    # Scatter plot with slider
    html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.RangeSlider(
            id = 'position-slider',
            min = df['w_id'].min(),
            max = df['w_id'].max(),
            value = [0,df['w_id'].mean()],
            marks = {str(x): str(x) for x in displayed_x_indicators},
            step = 6 )
        ],style={'margin': '20px'}),
    # Options
    html.Div([  
        html.Div([
            html.Div([
                html.H1('Opcje'),
                html.H2('Wybór przecigacza:'),
                dcc.Dropdown(
                id='broach',
                options=[{'label': i, 'value': i} for i in available_broach_indicators],
                value = available_broach_indicators[0]),
                html.H2('Wartość na wykresie:'),
                dcc.Dropdown(
                id = 'img_color',
                options=[{'label': i, 'value': i} for i in available_color_indicators],
                value = available_color_indicators[1]),
                html.H2('Rysuj wady:'),
                dcc.Checklist(
                    id = 'draw',
                    options=[      
                    {'label': 'narost\n', 'value': 1},
                    {'label': 'stępienie\n', 'value': 2},
                    {'label': 'zatarcie\n', 'value': 3},
                    {'label': 'wykruszenie\n', 'value': 4}],
                    value=[1,2,3,4]),
                ],style={'width': '90%','height': '535px','float': 'left', 'display': 'inline-block','margin-left': '20px','margin-right': '20px'})
            ],style={'width': '90%','float': 'left', 'display': 'inline-block','background':'rgb(234, 234, 242)','margin-left': '40px','margin-right': '40px'})
        ],style={'width': '16%','height': '90%','float': 'left', 'display': 'inline-block'}), 
    # Info
    html.Div([  
        html.Div([
            html.Div([
                html.H1('Informacje'),
                    'wysokość:\n'
                    'szerokość:\n'
                    'pole:\n' 
                ],style={'width': '90%','height':'535px','float': 'left', 'display': 'inline-block','margin-left': '20px','margin-right': '20px'})
            ],style={'width': '90%','float': 'left', 'display': 'inline-block'})
            
        ],style={'width': '15%','float': 'left', 'display': 'inline-block','margin-left': '40px','margin-right': '40px','background':'rgb(234, 234, 242)'}),        
    # Image 1
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='image1d',
                options=[{'label': i, 'value': i} for i in available_dropdown_indicators],
                value = available_dropdown_indicators[1]),
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
                value = available_dropdown_indicators[2]),
            ],style={'width': '85%','float': 'left', 'display': 'inline-block','margin-left': '50px','margin-right': '50px'}),
        html.Br(),
        html.Div([
            html.H2(id='title2')
            ],style={'textAlign': 'center'}),
        dcc.Graph(id='image2g'), 
        ],style={'width': '32%','float': 'right', 'display': 'inline-block'}),         
])


@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('position-slider', 'value'),
     Input('img_color', 'value'),
     Input('broach', 'value')]
    )
def update_scatter(value,categoryPick,FOLDER_NAME):
 
    df = pd.read_csv(BROACH_CSV + '\\' + FOLDER_NAME + '.csv')

    filtered_df = df[ (df['w_id'] <= value[1]) & (df['w_id'] >= value[0]) ]
    c = COLORS[ available_color_indicators.index(categoryPick)]
    print(c)
    fig = px.scatter(filtered_df, 
                      x = "w_id", 
                      y = "l_id", 
                      color = c,
                      hover_name = "img_name",
                      log_x = False)
    fig.update_layout(transition_duration = 200, 
                      height = 600, template = "seaborn",
                      yaxis=dict(title_text = "WIDTH [mm]", titlefont=dict(size=20), tickvals = displayed_y_indicators),
                      xaxis=dict(title_text = "LENGHT [mm]", titlefont=dict(size=20), tickvals = displayed_x_indicators),
                      margin=dict(l = 10, r = 10, t = 3, b = 3))
    fig.update_traces(marker=dict(size=15,
                                  line=dict(width=1, color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
    return fig


@app.callback(
    [Output('image1g', 'figure'),
     Output('title1', 'children')],
    [Input('graph-with-slider', 'clickData'),
     Input('image1d','value'),
     Input('broach', 'value'),
     Input('draw','value')],
    )
def update_image(clickData,value,FOLDER_NAME,draw_pick):

    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    DATA_FOLDER_PATH =  BROACH_DIR + '\\' + FOLDER_NAME + '\\' + SUBFOLDER
    IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    FULL_PATH = DATA_FOLDER_PATH + '\\' + IMAGE_NAME 
    img = io.imread(FULL_PATH)

    # Displaying failures categories
    if(value == 'Segmentacja'): 
        mask = output = np.zeros_like(img)
        inst_ids = str(df.loc[df['img_name'] == IMAGE_NAME, 'inst_id'])
        inst_ids = inst_ids[inst_ids.rfind('[') + 1:]
        inst_ids = inst_ids[:inst_ids.rfind(']')]
        inst_ids = np.array(inst_ids.split(' '))
        inst_num = df.loc[df['img_name']==IMAGE_NAME, 'inst_num']
        for i in range(int(inst_num)):
            DATA_FOLDER_PATH =  BROACH_DIR + '\\' + FOLDER_NAME + '\segmentation'  
            BASE_NAME = IMAGE_NAME.split('.')[0]
            FULL_PATH = DATA_FOLDER_PATH + '\\' + BASE_NAME + '-' + str(i) + '.png'
            mask = io.imread(FULL_PATH)
            mask = cv.bitwise_not(mask)
            b,g,r = cv.split(img)
            if (inst_ids[i]=='1' and (1 in draw_pick)): # blue - 'stępienie' 
                g = cv.bitwise_and(g,cv.split(mask)[0])
                r = cv.bitwise_and(r,cv.split(mask)[0])
            elif(inst_ids[i]=='2'and (2 in draw_pick)): # red - 'narost'
                g = cv.bitwise_and(g,cv.split(mask)[0])
                b = cv.bitwise_and(b,cv.split(mask)[0])
            elif(inst_ids[i]=='3'and (3 in draw_pick)): # green - 'zatarcie'
                r = cv.bitwise_and(r,cv.split(mask)[0])
                b = cv.bitwise_and(b,cv.split(mask)[0])
            elif(inst_ids[i]=='4'and (4 in draw_pick)): # yellow - 'wykruszenie'
                r = cv.bitwise_and(r,cv.split(mask)[0])
            img = cv.merge([b,g,r])

    fig = px.imshow(img)
    fig.update_layout(template="none",
                      xaxis=dict(showgrid = False, showline = False, visible = False), 
                      yaxis = dict(showgrid = False, showline = False, visible = False),  
                      margin=dict(l=10, r=10, b=0, t=0, pad=10))

    return fig, IMAGE_NAME

@app.callback(
    [Output('image2g', 'figure'),
     Output('title2', 'children')],
    [Input('graph-with-slider', 'clickData'),
     Input('image2d','value'),
     Input('broach', 'value'),
     Input('draw','value')]
    )
def update_image(clickData,value,FOLDER_NAME,draw_pick):
    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    DATA_FOLDER_PATH =  BROACH_DIR + '\\' + FOLDER_NAME + '\\' + SUBFOLDER
    IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    FULL_PATH = DATA_FOLDER_PATH + '\\' + IMAGE_NAME 
    img = io.imread(FULL_PATH)

    # Displaying failures categories
    if(value == 'Segmentacja'): 
        mask = output = np.zeros_like(img)
        inst_ids = str(df.loc[df['img_name']==IMAGE_NAME, 'inst_id'])
        inst_ids = inst_ids[inst_ids.rfind('[') + 1:]
        inst_ids = inst_ids[:inst_ids.rfind(']')]
        inst_ids = np.array(inst_ids.split(' '))
        inst_num = df.loc[df['img_name']==IMAGE_NAME, 'inst_num']
        for i in range(int(inst_num)):
            DATA_FOLDER_PATH =  BROACH_DIR + '\\' + FOLDER_NAME + '\segmentation'  
            BASE_NAME = IMAGE_NAME.split('.')[0]
            FULL_PATH = DATA_FOLDER_PATH + '\\' + BASE_NAME + '-' + str(i) + '.png'
            mask = io.imread(FULL_PATH)
            mask = cv.bitwise_not(mask)
            b,g,r = cv.split(img)
 
            if (inst_ids[i]=='1' and (1 in draw_pick)): # blue - 'stępienie' 
                g = cv.bitwise_and(g,cv.split(mask)[0])
                r = cv.bitwise_and(r,cv.split(mask)[0])
            elif(inst_ids[i]=='2'and (2 in draw_pick)): # red - 'narost'
                g = cv.bitwise_and(g,cv.split(mask)[0])
                b = cv.bitwise_and(b,cv.split(mask)[0])
            elif(inst_ids[i]=='3'and (3 in draw_pick)): # green - 'zatarcie'
                r = cv.bitwise_and(r,cv.split(mask)[0])
                b = cv.bitwise_and(b,cv.split(mask)[0])
            elif(inst_ids[i]=='4'and (4 in draw_pick)): # yellow - 'wykruszenie'
                r = cv.bitwise_and(r,cv.split(mask)[0])
            img = cv.merge([b,g,r])
    
    fig = px.imshow(img)
    fig.update_layout(template="none",
                      xaxis=dict(showgrid = False, showline = False, visible = False), 
                      yaxis = dict(showgrid = False, showline = False, visible = False),  
                      margin=dict(l=10, r=40, b=0, t=0, pad=10))

    return fig, IMAGE_NAME

    

if __name__ == '__main__':
    app.run_server(debug=False)
