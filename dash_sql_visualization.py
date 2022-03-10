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
import sql_connection
import statistics
from dash import Dash, dash_table
from collections import OrderedDict
import dash_bootstrap_components as dbc



def convert_sql_output(sql_data):
    sql_data = np.array(sql_data)
    sql_data = np.reshape(sql_data,(-1))
    return sql_data
def find_last_scan(scan_names):
    '''
    Required name format yy-mm-dd-time
    '''
    last = scan_names[0]
    for scan in scan_names:
        if scan > last: last = scan
    return last
def find_broach_width(scan_name,available_rows):
    max_tooth_number = 0
    for row in available_rows:
        teeth = convert_sql_output(sql.get_tooth_param(default_broach,'tooth_number',row))
        if max(teeth) > max_tooth_number: max_tooth_number = max(teeth) 
    teeth = list(range(1, max_tooth_number + 1))
    return teeth
def update_image(clickData,value,FOLDER_NAME,draw_pick):
    sql = sql_connection.SQLConnection(debug=False)


    # Get image path from SQL 
    path = convert_sql_output(sql.get_scan_param('path', default_broach))[0]
    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    try:
        IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    except:
        IMAGE_NAME = '1_1.png'
    FULL_PATH = path + SUBFOLDER + '/'  + IMAGE_NAME 
    img = io.imread(FULL_PATH)
    
    # Displaying failures categories
    if(value == 'Segmentacja'): 
        mask = output = np.zeros_like(img)
        row = int((IMAGE_NAME.split('.')[0]).split('_')[1])
        inst_ids = convert_sql_output(sql.get_tooth_param(default_broach,'pred_class',row,'image_name=\'{}\''.format(IMAGE_NAME)))
        if inst_ids: inst_ids = str(inst_ids[0])
        inst_ids = inst_ids[inst_ids.rfind('[') + 1:]
        inst_ids = inst_ids[:inst_ids.rfind(']')]
        inst_ids = np.array(inst_ids.split(' '))
        inst_num = int(convert_sql_output(sql.get_tooth_param(default_broach,'num_instances',row,'image_name=\'{}\''.format(IMAGE_NAME)))[0])

        for i in range(int(inst_num)):
            DATA_FOLDER_PATH =  path + '\segmentation'  
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
    
    sql = None
    return fig, IMAGE_NAME

sql = sql_connection.SQLConnection(debug=False)
BROACH_LIST = convert_sql_output( sql.get_scan_param('nazwa'))
default_broach = find_last_scan(BROACH_LIST) # Find newest scan and display it as default
displayed_x_indicators = convert_sql_output(sql.get_row_param(default_broach,'row_number'))
displayed_y_indicators = find_broach_width(BROACH_LIST,displayed_x_indicators)


# Available preview modes
available_dropdown_indicators = ['Orginalny', 'Wyodrebniony', 'Segmentacja']
SUBFOLDERS = [r'\images', r'\otsu_tooth', r'\otsu_tooth']

# Available heatmap modes
available_color_indicators = ['Długość', 'Szerokość', 'Położenie środka - długość', 'Położenie środka - szerokość', 'Ilość wad','Stępienie', 'Narost', 'Zatarcie', 'Wykruszenie']
COLORS = ['length', 'width','centre_lenght', 'centre_width', 'num_instances', 'stepienie', 'narost', 'zatarcie', 'wykruszenie']

data = OrderedDict(
    [
        ("Parameter", ["Długość", "Szerokość", "Położenie dłgość", "Położenie wysokość", "Stępienie zęba", "Stępienie rzędu"]),
        ("Value [mm]", [0, 0, 0, 0, 0, 0]),
    ]
)
dat = pd.DataFrame(OrderedDict([(name, col_data) for (name, col_data) in data.items()]))

# Available broaches
available_broach_indicators = BROACH_LIST
sql = None

app = dash.Dash(__name__)
# Layout structure
app.layout = html.Div([
    # Title
    html.Div([
        html.H1('Kontrola zużycia zębów przeciągacza'),
        ],style={'textAlign': 'center'}),
    # Scatter plot with slider
    html.Div([
        dcc.Graph(id='graph-with-slider'),
        dcc.Graph(id="line-chart"),
        dcc.RangeSlider(
            id = 'position-slider',
            min = min(displayed_x_indicators),
            max = max(displayed_x_indicators),
            value = [0,int(statistics.mean(displayed_x_indicators))],
            marks = {str(x): str(x) for x in displayed_x_indicators},
            step = 1 )
        ],style={'margin': '20px'}),
    # Options
    html.Div([  
        html.Div([
            html.Div([
                html.H1('Opcje'),
                html.H2('Wybór przecigacza:'),
                dcc.Dropdown(
                id = 'broach',
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
                    labelStyle={'display': 'block', 'text-align': 'justify', 'margin-bottom':'7px'},
                    value=[1,2,3,4]),
                        
                    
                ],style={'width': '90%','height': '535px','float': 'left', 'display': 'inline-block','margin-left': '20px','margin-right': '20px'})
            ],style={'width': '90%','float': 'left', 'display': 'inline-block','background':'rgb(234, 234, 242)','margin-left': '40px','margin-right': '40px'})
        ],style={'width': '16%','height': '90%','float': 'left', 'display': 'inline-block','height':'535px'}), 
    # Table
    html.Div([  
        html.Div([
            html.Div([
                html.Div([
                    html.H1('Dodatkowe dane'),
                    dash_table.DataTable(
                        id='datatable-interactivity',
                        data=dat.to_dict('records'),
                        columns=[{'id': c, 'name': c} for c in dat.columns],
                        page_size=10,
                        style_cell={'font-size':'15px'},
                        style_header={'textAlign': 'center', 'fontWeight': 'bold', 'margin':'None','font-size':'20px'}
                    ) 
                ],style={'width': '90%','float': 'left', 'display': 'inline-block','margin-left': '30px','margin-right': '20px'}),
            ],style={'width': '90%','float': 'left', 'display': 'inline-block'})      
        ],style={'width': '15%','float': 'left', 'display': 'inline-block','margin-left': '40px','margin-right': '40px','background':'rgb(234, 234, 242)','height':'535px'}),  
    ]),      
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
        dcc.Graph(id = 'image1g'),
        ],style={'width': '32%', 'display': 'inline-block','height':'535px'}),  
    # Image 2
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
        ],style={'width': '32%','float': 'right', 'display': 'inline-block','height':'535px'}),  
    # Separation
    html.Div([

    ],style={'width': '100%','float': 'left', 'display': 'inline-block','height':'35px'}),     
    # Plot
    html.Div([  
        html.Div([
            html.Div([
                html.H1('Stępienie w rzędzie'),
                html.Div([
                    dcc.Graph(id = 'blunt_plot'),
                ],style={'float': 'centre'})              
            ],style={'width': '90%','height': '535px','float': 'centre', 'display': 'inline-block','margin-left': '40px','margin-right': '40px','margin-bottom': '20px'})
        ],style={'width': '90%','float': 'left', 'display': 'inline-block','background':'rgb(234, 234, 242)','margin-left': '40px','margin-right': '40px'})
    ],style={'width': '34.5%','height': '90%','float': 'left', 'display': 'inline-block','height':'535px','margin-bottom': '20px'}),     
])

# Scatter plot with slider
@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('position-slider', 'value'),
     Input('img_color', 'value'),
     Input('broach', 'value')]
    )
def update_scatter(value,categoryPick,FOLDER_NAME):
    sql = sql_connection.SQLConnection(debug=False)


    c = COLORS[ available_color_indicators.index(categoryPick)]
    data = sql.select_from_view('22-03-02-14-14',c,'row_number<={} and row_number>={}'.format(value[1], value[0]))

    x,y,v,image_name = [],[],[],[]
    for sx,sy,sv,name in data:
        x.append(sx)
        y.append(sy)
        v.append(round(sv,3))
        image_name.append(name)
    d = {'x':x,'y':y, 'value':v,'image_name':image_name}
    df = pd.DataFrame(data = d)

    fig = px.scatter(df, 
                      x = "y", 
                      y = "x", 
                      color = 'value',
                      hover_name = 'image_name',
                      log_x = False)
    fig.update_layout(transition_duration = 200, 
                      height = 600, template = "seaborn",
                      yaxis=dict(title_text = "WIDTH", titlefont=dict(size=20), tickvals = displayed_y_indicators),
                      xaxis=dict(title_text = None, titlefont=dict(size=20), tickvals = displayed_x_indicators),
                      margin=dict(l = 10, r = 10, t = 3, b = 3))
    fig.update_traces(marker=dict(size=15,
                                  line=dict(width=1, color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
    sql = None
    return fig

# Line plot
@app.callback(
    Output("line-chart", "figure"), 
    [Input('position-slider', 'value'),
     Input('broach', 'value')]
    )
def update_line_chart(value,FOLDER_NAME):
    sql = sql_connection.SQLConnection(debug=False)

    
    y = convert_sql_output(sql.get_row_param(default_broach,'stepienie_row_value','row_number<={} and row_number>={}'.format(value[1], value[0])))
    x = convert_sql_output(sql.get_row_param(default_broach,'row_number','row_number<={} and row_number>={}'.format(value[1], value[0])))

    d = {'x':x,'y':y}
    df = pd.DataFrame(data = d)
    try:
        fig = px.line(df, x = "x", y = "y")
    except:
        fig = px.line(df, x = "x", y = "y")
    fig.update_layout(transition_duration = 200, 
                      height = 150, template = "seaborn",
                      yaxis=dict(title_text = "STĘPIENIE", titlefont=dict(size=20)),
                      xaxis=dict(title_text = None, titlefont=dict(size=20), tickvals = displayed_x_indicators),
                      margin=dict(l = 10, r = 130, t = 3, b = 20))

    sql = None
    return fig

# Image 1
@app.callback(
    [Output('image1g', 'figure'),
     Output('title1', 'children')],
    [Input('graph-with-slider', 'clickData'),
     Input('image1d','value'),
     Input('broach', 'value'),
     Input('draw','value')],
    )
def image_box_1(clickData,value,FOLDER_NAME,draw_pick): 
    return update_image(clickData,value,FOLDER_NAME,draw_pick)

#Image 2
@app.callback(
    [Output('image2g', 'figure'),
     Output('title2', 'children')],
    [Input('graph-with-slider', 'clickData'),
     Input('image2d','value'),
     Input('broach', 'value'),
     Input('draw','value')]
    )
def image_box_2(clickData,value,FOLDER_NAME,draw_pick):
    return update_image(clickData,value,FOLDER_NAME,draw_pick)

#Plot
@app.callback(
    [Output('blunt_plot', 'figure')],
    [Input('graph-with-slider', 'clickData'),
     Input('image2d','value'),
     Input('broach', 'value'),
     Input('draw','value')]
    )
def update_info(clickData,value,FOLDER_NAME,draw_pick):
    sql = sql_connection.SQLConnection(debug=False)


    # Get image path from SQL 
    path = convert_sql_output(sql.get_scan_param('path', default_broach))[0]
    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    try:
        IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    except:
        IMAGE_NAME = '1_1.png'
    row = int((IMAGE_NAME.split('.')[0]).split('_')[1])
    FULL_PATH = path + '/plots/'  + str(row) + '.jpg' 
    img = io.imread(FULL_PATH)

    fig = px.imshow(img)
    fig.update_layout(template="none",
                      xaxis=dict(showgrid = False, showline = False, visible = False), 
                      yaxis = dict(showgrid = False, showline = False, visible = False),  
                      margin=dict(l=10, r=10, b=10, t=10, pad=10))
    sql = None
    return fig,

#Table
@app.callback(
    Output('datatable-interactivity', 'data'),
    [Input('graph-with-slider', 'clickData'),
     Input('image2d','value'),
     Input('broach', 'value'),
     Input('draw','value')]
)
def update_table(clickData,value,FOLDER_NAME,draw_pick):
    sql = sql_connection.SQLConnection(debug=False)

    try:
        IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    except:
        IMAGE_NAME = '1_1.png'

    needed_data = ['length', 'width','centre_lenght', 'centre_width', 'stepienie']
    row = int((IMAGE_NAME.split('.')[0]).split('_')[1])
    tooth = int((IMAGE_NAME.split('.')[0]).split('_')[0])

    display = []
    for value in needed_data:
        v = convert_sql_output(sql.get_tooth_param(default_broach,value,row,'tooth_number={};'.format(tooth)))[0]
        v = round(v,3)
        display.append(v)

        
    st = convert_sql_output(sql.get_row_param(default_broach,'stepienie_row_value','row_number={}'.format(row)))[0]
    st = round(st,3)

    data = OrderedDict(
        [
            ("Parameter", ["Długość", "Szerokość", "Położenie dłgość", "Położenie wysokość", "Stępienie zęba", "Stępienie rzędu"]),
            ("Value [mm]", [display[0], display[1], display[2], display[3], display[4], st]),
        ]
    )
    dat = pd.DataFrame(OrderedDict([(name, col_data) for (name, col_data) in data.items()]))
    dat = dat.to_dict('records')

    sql = None
    return dat

    




if __name__ == '__main__':
    app.run_server(debug=False)
