from click import style
from cv2 import DescriptorMatcher_BRUTEFORCE_HAMMINGLUT
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
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
import plotly.graph_objects as go
from os.path import exists
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # Enable corrupted images loading


def convert_sql_output(sql_data):
    # Convert data fromat recieved by SQL to the list
    sql_data = np.array(sql_data)
    sql_data = np.reshape(sql_data,(-1))
    return sql_data
def find_last_scan(scan_names):
    '''
    Find scan with was created as last to display it at first in the application
    Required name format yy-mm-dd-time
    '''
    last = scan_names[0]
    for scan in scan_names:
        if scan > last: last = scan
    return last
def find_broach_width(scan_name,available_rows):
    '''
    Find width of the broach based on its rows
    Search for global max width value in each row
    '''
    sql = sql_connection.SQLConnection(debug=False)
    max_tooth_number = 0
    for row in available_rows:
        teeth = convert_sql_output(sql.get_tooth_param(scan_name,'tooth_number',row))
        if max(teeth) > max_tooth_number: max_tooth_number = max(teeth) 
    teeth = list(range(1, max_tooth_number + 1))
    sql = None
    return teeth
def update_image(clickData,value,FOLDER_NAME,draw_pick):
    '''
    Used for updating preview windows. Used by 2 callbacks 
    Display image based on the picked category
    In case of segmentation display failures mask
    When inference was corrupted (marked as 'x' on the scatter) 
    display orginal image grabbed from the scaner  
    '''
    
    sql = sql_connection.SQLConnection(debug=False)
    default_broach = FOLDER_NAME

    # Get image path from SQL 
    path = convert_sql_output(sql.get_scan_param('path', default_broach))[0]
    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    
    # Try to get clicked tooth name if not possible pick fist tooth
    try:
        IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    except:
        IMAGE_NAME = '1_1.png'
    FULL_PATH = path + SUBFOLDER + '/'  + IMAGE_NAME 
    
    # Try to get image form picked caterory, if not possible pick orginal image
    if exists(FULL_PATH):
        try:
            img = io.imread(FULL_PATH)
            # Displaying failures categories when segmentation mode is picked
            if(value == 'Segmentation'): 
                # Get classes and failures data from sql database and process it (some data is saved as string)
                mask = output = np.zeros_like(img)
                row = int((IMAGE_NAME.split('.')[0]).split('_')[1])
                inst_ids = convert_sql_output(sql.get_tooth_param(default_broach,'pred_class',row,'image_name=\'{}\''.format(IMAGE_NAME)))
                if inst_ids: inst_ids = str(inst_ids[0])
                inst_ids = inst_ids[inst_ids.rfind('[') + 1:]
                inst_ids = inst_ids[:inst_ids.rfind(']')]
                inst_ids = np.array(inst_ids.split(' '))
                inst_num = int(convert_sql_output(sql.get_tooth_param(default_broach,'num_instances',row,'image_name=\'{}\''.format(IMAGE_NAME)))[0])

                # Iterate over existing failures instances and color it
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
        except:
            print("Image opening error")
    else: 
        print("Picked image is not available")
        FULL_PATH = path + SUBFOLDERS[0] + '/'  + IMAGE_NAME 
        print("Try to open:",FULL_PATH)
        img = io.imread(FULL_PATH)

    # Preapre displayed image
    fig = px.imshow(img)
    fig.update_layout(template="none",
                      xaxis=dict(showgrid = False, showline = False, visible = False), 
                      yaxis = dict(showgrid = False, showline = False, visible = False),  
                      margin=dict(l=10, r=10, b=0, t=0, pad=10))
    
    sql = None
    return fig

# Establish SQL connection 
sql = sql_connection.SQLConnection(debug=False)

# Find newest scan and display it as default
BROACH_LIST = convert_sql_output( sql.get_scan_param('nazwa'))
default_broach = find_last_scan(BROACH_LIST) 

# Find indicators of the slider scatter and bar plot based on the rows and sections of the picked broach
displayed_x_indicators = convert_sql_output(sql.get_row_param(default_broach,'row_number'))
displayed_y_indicators = find_broach_width(default_broach,displayed_x_indicators)

# Terminate sql connection 
sql = None

# Available preview modes
available_dropdown_indicators = ['Orginal', 'Extraction', 'Segmentation'] # User visible names
SUBFOLDERS = [r'\images', r'\otsu_tooth', r'\otsu_tooth'] # Corresponding directories

# Available heatmap modes
available_color_indicators = ['Lenght', 'Width', 'Center coordinate - y', 'Center coordinate - x', 'Number of failures','Stępienie', 'Narost', 'Zatarcie', 'Wykruszenie'] # User visible names
COLORS = ['length', 'width','centre_lenght', 'centre_width', 'num_instances', 'stepienie', 'narost', 'zatarcie', 'wykruszenie'] # Corresponding SQL colums names

# Base categories and information displayed in the Tooth report table
data = OrderedDict(
    [
        ("Parameter", ["Długość", "Szerokość", "Położenie dłgość", "Położenie wysokość", "Stępienie zęba", "Stępienie rzędu"]),
        ("Value [mm]", [0, 0, 0, 0, 0, 0]),
    ]
)
dat = pd.DataFrame(OrderedDict([(name, col_data) for (name, col_data) in data.items()]))

# Available broaches
available_broach_indicators = BROACH_LIST # List of the all available scans


# Layout global styling parameters
cards_st ={"height": 660,"font-family": "Arial"}
font_st = {'font-size':'20px','height':'35px',"font-family": "Arial"}

# Utilized graphics
PLOTLY_LOGO = "http://www.mvlab.pl/images/logo.png"
polish = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Flag_of_Poland_%28normative%29.svg/640px-Flag_of_Poland_%28normative%29.svg.png"
english = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Flag_of_Great_Britain_%281707%E2%80%931800%29.svg/800px-Flag_of_Great_Britain_%281707%E2%80%931800%29.svg.png?20220223062637"
german = "https://upload.wikimedia.org/wikipedia/en/thumb/b/ba/Flag_of_Germany.svg/255px-Flag_of_Germany.svg.png"

# Bootstrap style
app = dash.Dash(__name__,external_stylesheets =[dbc.themes.LITERA])
# Layout structure
app.layout = html.Div([
    # Navbar 
    dbc.Card([
        dbc.Row([
            # MV Lab logo
            dbc.Col([
                html.Div([
                    html.Img(src=PLOTLY_LOGO, height="70px")
                ],style={"vertical-align": "middle"}) 
            ],
            style={"vertical-align": "middle", "margin-left":"105px",},
            width=1),
            # APP Name
            dbc.Col([
                html.H2(["BROACH CONTROL APP"],style={"color":"#597cc7","margin-top":"10px","margin-bottom":"10px",}), 
            ],
            style={"text-color":"#597cc7",'display': 'left',"margin-left":"55px","margin-right":"-55px"},
            width=6),
            # Languages
            dbc.Col([
                html.Div([
                    html.Img(src=polish, height="48px")
                ],style={"align": "right"}),
            ],
            style={"vertical-align": "middle", "margin-left":"85px","margin-top":"10px","margin-bottom":"10px","padding":"0px"},
            width=1),
            dbc.Col([
                html.Div([
                    html.Img(src=english, height="48px")
                ],style={"vertical-align": "middle"}) 
            ],
            style={"vertical-align": "middle", "margin-left":"85px","margin-top":"10px","margin-bottom":"10px","padding":"0px"},
            width=1),
            dbc.Col([
                html.Div([
                    html.Img(src=german, height="48px")
                ],style={"vertical-align": "middle"}) 
            ],
            style={"vertical-align": "middle", "margin-left":"85px","margin-top":"10px","margin-bottom":"10px","padding":"0px"},
            width=1),
        ])
    ]),

    html.Br(),
    dbc.Card([
        # Scatter 
        dbc.Row(
            dbc.Col(
                html.Div(
                    dcc.Graph(id='graph-with-slider')
                )
            )
        ),
        html.Br(),

        # Bar plot
        dbc.Row(
            dbc.Col(
                html.Div(
                    dcc.Graph(id="line-chart"),
                )
            )
        ),
        html.Br(),

        # Slider
        dbc.Row(
            dbc.Col(
                html.Div(
                    html.Div(
                        dcc.RangeSlider(
                            id = 'position-slider',
                            min = min(displayed_x_indicators),
                            max = max(displayed_x_indicators),
                            value = [0,int(statistics.mean(displayed_x_indicators))],         
                            step = 1 
                        )    
                    )
                )
            )
        ),
        html.Br(),
 
        # 1st row
        dbc.Row([
            # Options
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2('Options'),
                        ),
                        dbc.CardBody([
                            html.H3('Select broach:'),
                            dcc.Dropdown(
                                id = 'broach',
                                options=[{'label': i, 'value': i} for i in available_broach_indicators],
                                value = available_broach_indicators[-1], 
                                style = font_st
                            ),
                            html.Br(),

                            html.H3('Scatter value:'),
                            dcc.Dropdown(
                                id = 'img_color',
                                options=[{'label': i, 'value': i} for i in available_color_indicators],
                                value = available_color_indicators[5],
                                style = font_st
                            ),
                            html.Br(),

                            html.H3('Draw failures:'),
                            dcc.Checklist(
                                id = 'draw',
                                options=[      
                                    {'label': ' Narost\n', 'value': 1},
                                    {'label': ' Stępienie\n', 'value': 2},
                                    {'label': ' Zatarcie\n', 'value': 3},
                                    {'label': ' Wykruszenie\n', 'value': 4}
                                ],
                                labelStyle={'display': 'block', 'text-align': 'justify', 'margin-bottom':'7px', 'margin-left':'15px', 'height':'25px', 'font-size':'20px',"font-family": "Arial"},
                                value=[1,2,3,4],
                            ),
                        ]),
                    ],style = cards_st)
                ]),
                width=2
            ), 
                
            # Information
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2("Information"),
                        ),
                        dbc.CardBody([
                            html.H3("Tooth name:"),
                            html.P(id='tooth-name',style = font_st),
                            html.H3('Tooth report:'),
                            dash_table.DataTable(
                                id='datatable-interactivity',
                                data=dat.to_dict('records'),
                                columns=[{'id': c, 'name': c} for c in dat.columns],
                                page_size = 10,
                                style_cell={'font-size':'15px','margin':'10px','font-size':'20px',"font-family": "Arial"},
                                style_header={'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial"}
                            ),
                            html.Br(),
                            html.H3('Modify row blunt value:'),

                            html.Div(dcc.Input(id='input-on-submit-row', type='number',placeholder='Row number',style={'font-size': '20px','width': '100%','margin-bottom':'10px'})),
                            html.Div(dcc.Input(id='input-on-submit-blunt', type='number',placeholder='Custom blunt value [mm]',style={'font-size': '20px','width': '100%','margin-bottom':'10px'})),
                            html.Div([
                                dbc.Button('Submit', id='submit-val', color="secondary" ,n_clicks=0, style={'font-size': '20px'}),  

                                ],className="d-grid gap-2 col-6 mx-auto"
                            ),
                            
                            
                            html.Div(id='empty-container')
                        ])
                  
                    ],style = cards_st)
                ]),
                width=2
            ), 

            # Image window 1
            dbc.Col(
                html.Div([
                    dbc.Card([
                    dbc.CardHeader(
                        html.H2('Preview 1 mode'),
                    ),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id = 'image1d',
                            options = [{'label': i, 'value': i} for i in available_dropdown_indicators],
                            value = available_dropdown_indicators[1],
                            style = font_st
                        ),
                        html.Br(),
                        dcc.Graph(id = 'image1g'),
                    ])

                    ],style = cards_st)
                ]),
                width = 8,
            ),        
        ]), 
        html.Br(),
        
        # 2nd row
        dbc.Row([
            # Blunt plot
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2('Stępienie structure in row'),
                        ),
                        dbc.CardBody([
                            dcc.Graph(id = 'blunt_plot'),   
                        ],style=cards_st)                    
                    ],style = cards_st)
                ]),
                width=4
            ),
            
            # Image window 2
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                        html.H2('Preview 2 mode'),
                        ),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='image2d',
                                options=[{'label': i, 'value': i} for i in available_dropdown_indicators],
                                value = available_dropdown_indicators[2], 
                                style = font_st
                            ),
                            html.Br(),
                            dcc.Graph(id='image2g'),
                        ]),
   
                    ],style = cards_st)
                ]),
                width = 8
            ),
        ]), 
        html.Br(), 
        dbc.CardFooter()  
    ],
    style = {'margin-left':'20px','margin-right':'20px',"font-family": "Arial"},
    color = "light", 
    outline = True)
])

# Update slider 
@app.callback(
    [Output('position-slider', 'min'),
     Output('position-slider', 'max'),],
    Input('broach', 'value'),
    )
def update_slider(FOLDER_NAME = default_broach):
    print('\nUpdate slider:')
    sql = sql_connection.SQLConnection(debug=False)
    displayed_x_indicators = convert_sql_output(sql.get_row_param(FOLDER_NAME,'row_number'))
    displayed_y_indicators = find_broach_width(FOLDER_NAME,displayed_x_indicators)
    min_value = min(displayed_x_indicators)
    max_value = max(displayed_x_indicators)
    sql = None
    return min_value,max_value

# Update scatter
@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('position-slider', 'value'),
     Input('img_color', 'value'),
     Input('broach', 'value'),]
    )
def update_scatter(value,categoryPick,FOLDER_NAME):
    print("\nUpdate scatter:")
    sql = sql_connection.SQLConnection(debug=False)
    
    default_broach = FOLDER_NAME
    c = COLORS[ available_color_indicators.index(categoryPick)]
    data = sql.select_from_view(default_broach,c,'row_number<={} and row_number>={}'.format(value[1], value[0]))

    x,y,v,image_name = [],[],[],[]
    x_n,y_n,image_name_n = [],[],[]
    for sy,sx,sv,name in data:
        if sv is not None:
            x.append(sx)
            y.append(sy)
            v.append(round(sv,3))
            image_name.append(name)
        else: 
            x_n.append(sx)
            y_n.append(sy)
            image_name_n.append(name)

    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=x_n, 
            y=y_n,
            marker = dict(
                color = "black",
            ), 
            mode = 'markers',
            marker_symbol = "x",
            showlegend = False,
            hovertext = image_name_n,
            name = "error"
            )
        )
    fig.add_trace(
        go.Scatter(
            x = x, 
            y = y, 
            marker = dict(
                color = v,
                colorscale = ["green","yellow","red"],
                showscale = True
            ), 
            mode = 'markers',
            hovertext = image_name,
            name = "succes",   
        )
    )
                      
    fig.update_layout(
        transition_duration = 200, 
        height = 600, 
        template = "seaborn",
        yaxis = dict(title_text = "WIDTH", titlefont=dict(size=20), tickvals = displayed_y_indicators),
        xaxis = dict(title_text = None, titlefont=dict(size=20), tickvals = displayed_x_indicators),
        margin = dict(l = 10, r = 10, t = 3, b = 10), 
        hoverlabel = dict( bgcolor="white", font_size=20, font_family="Arial")
    )
    fig.update_traces(
        marker=dict(size=15,line=dict(width=1, color='DarkSlateGrey')),
            selector=dict(mode='markers'), 
            hoverinfo="text"
    )
                                  
    sql = None


    return fig

# Change row blunt    
@app.callback(
    Output('submit-val', 'color'),
    Input('submit-val', 'n_clicks'),
    [State('input-on-submit-blunt', 'value'),
     State('input-on-submit-row', 'value')])
def update_output(n_clicks, blunt_value,row_number):
    if n_clicks == 0: return "secondary" 
    if isinstance(row_number, int) and min(displayed_x_indicators) >= 0 and row_number <= max(displayed_x_indicators):
        sql = sql_connection.SQLConnection(debug=False)
        succes = sql.ovverride_row_stepienie(default_broach, blunt_value, row_number)
        sql = None
        if succes == 1: return "success"
        else: return "danger"
    else: return "danger"

# Bar plot
@app.callback(
    Output("line-chart", "figure"), 
    [Input('position-slider', 'value'),
     Input('broach', 'value'),
     Input('submit-val', 'n_clicks')]
    ) 
def update_bar_chart(value,FOLDER_NAME,nclicks):
    print("\nUpdate bar chart:")
    sql = sql_connection.SQLConnection(debug=False)
    
    default_broach = FOLDER_NAME
    y = convert_sql_output(sql.get_row_param(default_broach,'stepienie_row_value','row_number<={} and row_number>={}'.format(value[1], value[0])))
    x = convert_sql_output(sql.get_row_param(default_broach,'row_number','row_number<={} and row_number>={}'.format(value[1], value[0])))
    y_correction = convert_sql_output(sql.get_row_param(default_broach,'stepienie_correction','row_number<={} and row_number>={}'.format(value[1], value[0])))
    
    # Sort elements by row to avoid overlaping curves
    zipped_lists = zip(x, y)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    x_s, y_s = [ list(tuple) for tuple in  tuples]

    # Sort elements by row to avoid overlaping curves
    zipped_lists = zip(x, y_correction)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    x_s, y_correction_s = [ list(tuple) for tuple in  tuples]
    

    # Create dataframe
    d = {'x':x_s, 'y':y_s}
    df = pd.DataFrame(data = d)
    colors = ['rgb(184, 184, 183)','rgb(168, 225, 250)']
    try:
        fig = px.bar(df, x = "x", y = "y")
    except:
        fig = px.bar(df, x = "x", y = "y")

    correction =  go.Bar(x = x_s, y = y_correction_s, marker_color = 'crimson', base='stack', opacity=0.5, name="correction")
    fig.add_trace(correction)
    fig.update_layout(transition_duration = 200, 
                      height = 150, template = "seaborn",
                      yaxis=dict(title_text = "STĘPIENIE", titlefont=dict(size=20)),
                      xaxis=dict(title_text = None, titlefont=dict(size=20), tickvals = displayed_x_indicators),
                      margin=dict(l = 10, r = 130, t = 3, b = 20),hoverlabel = dict( bgcolor="white",font_size=20,font_family="Arial"))
    fig.update_coloraxes(showscale = False)
    fig.update_traces( hoverinfo="y")

    sql = None
    return fig

# Image 1
@app.callback(
    Output('image1g', 'figure'),
    [Input('graph-with-slider', 'clickData'),
     Input('image1d','value'),
     Input('broach', 'value'),
     Input('draw','value')],
    )
def image_box_1(clickData,value,FOLDER_NAME,draw_pick): 
    print("\nUpdate image box 1:")

    return update_image(clickData,value,FOLDER_NAME,draw_pick)

#Image 2
@app.callback(
    Output('image2g', 'figure'),
    [Input('graph-with-slider', 'clickData'),
     Input('image2d','value'),
     Input('broach', 'value'),
     Input('draw','value')]
    )
def image_box_2(clickData,value,FOLDER_NAME,draw_pick):
    print("\nUpdate image box 2:")
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
    print("\n\nUpdate plot:")
    sql = sql_connection.SQLConnection(debug=False)
    
    default_broach = FOLDER_NAME
    # Get image path from SQL 
    path = convert_sql_output(sql.get_scan_param('path', default_broach))[0]
    SUBFOLDER = SUBFOLDERS[ available_dropdown_indicators.index(value)]
    try:
        IMAGE_NAME = str(clickData['points'][0]['hovertext'])
    except:
        IMAGE_NAME = '1_1.png'
    row = int((IMAGE_NAME.split('.')[0]).split('_')[1])
    FULL_PATH = path + '/plots/'  + str(row) + '.jpg' 
    
    try:
        img = io.imread(FULL_PATH)
    except:
        img = np.zeros((10,10,3), np.uint8)
        img.fill(240)


    fig = px.imshow(img)
    fig.update_layout(template="none",
                      xaxis=dict(showgrid = False, showline = False, visible = False), 
                      yaxis = dict(showgrid = False, showline = False, visible = False),  
                      margin=dict(l=0, r=0, b=0, t=0, pad=0),
                      height = 550)
    sql = None
    return fig,

#Table
@app.callback(
    [Output('datatable-interactivity', 'data'),
     Output('tooth-name','children')],
    [Input('graph-with-slider', 'clickData'),
     Input('image2d','value'),
     Input('broach', 'value'),
     Input('draw','value')])
def update_table(clickData,value,FOLDER_NAME,draw_pick):
    print("\nUpdate table:")
    sql = sql_connection.SQLConnection(debug=False)
    

    default_broach = FOLDER_NAME
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
        try: v = round(v,3)
        except: v=v
        display.append(v)
     
    st = convert_sql_output(sql.get_row_param(default_broach,'stepienie_row_value','row_number={}'.format(row)))[0]
    try: st = round(st,3)
    except: st= st

    data = OrderedDict(
        [
            ("Parameter", ["Lenght", "Width", "Coordinate x", "Coordinate y", "Tooth stępienie", "Row stępienie"]),
            ("Value [mm]", [display[0], display[1], display[2], display[3], display[4], st]),
        ]
    )
    dat = pd.DataFrame(OrderedDict([(name, col_data) for (name, col_data) in data.items()]))
    dat = dat.to_dict('records')

    sql = None
    return dat, IMAGE_NAME


if __name__ == '__main__':
    app.run_server(debug=False)
