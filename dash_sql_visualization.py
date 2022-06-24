import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import pandas as pd
import numpy as np
from skimage import io
import cv2 as cv
import sql_connection
import optimization_module
import statistics
from dash import dash_table
from collections import OrderedDict
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from os.path import exists
import statistics
import datetime


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
    if len(scan_names) == 0: return "None"
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
                inst_ids = convert_sql_output(sql.get_tooth_param(default_broach,'pred_class',row,f"image_name=\'{IMAGE_NAME}\'"))
                if inst_ids: inst_ids = str(inst_ids[0])
                inst_ids = inst_ids[inst_ids.rfind('[') + 1:]
                inst_ids = inst_ids[:inst_ids.rfind(']')]
                inst_ids = np.array(inst_ids.split(' '))
                inst_num = int(convert_sql_output(sql.get_tooth_param(default_broach,'num_instances',row,f"image_name=\'{IMAGE_NAME}\'"))[0])

                # Iterate over existing failures instances and color it
                for i in range(int(inst_num)):
                    DATA_FOLDER_PATH = os.path.join(path,'segmentation')
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
        try:
            img = io.imread(FULL_PATH)
        except FileNotFoundError:
            img = None
            print(f"File is {FULL_PATH} unavailable!")


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
optimize = optimization_module.Optimization()

# Find newest scan and display it as default
BROACH_LIST = convert_sql_output( sql.get_scan_param('nazwa'))
default_broach = find_last_scan(BROACH_LIST) 

# Find indicators of the slider scatter and bar plot based on the rows and sections of the picked broach
displayed_x_indicators = convert_sql_output(sql.get_row_param(default_broach,'row_number'))
displayed_y_indicators = find_broach_width(default_broach,displayed_x_indicators)

# Params table values
table_columns_names = convert_sql_output(sql.get_table_names('TypeOfBroach'))
table_values = sql.get_table_values('TypeOfBroach')
params_df = pd.DataFrame(data = table_values, columns = table_columns_names)
params_data = params_df.to_dict('records')

# Broach (options) table values
broach_columns_names = convert_sql_output(sql.get_table_names('Broach'))
broach_values = sql.get_table_values('Broach')
broach_df = pd.DataFrame(data=broach_values, columns = broach_columns_names)
broach_data = broach_df.to_dict('records')

# Terminate sql connection 
sql = None

# Available preview modes
available_dropdown_indicators = ['Orginal', 'Extraction', 'Segmentation'] # User visible names
SUBFOLDERS = [r'\images', r'\otsu_tooth', r'\otsu_tooth'] # Corresponding directories

# Available heatmap modes
available_color_indicators = ['Lenght', 'Width', 'Center coordinate - y', 'Center coordinate - x', 'Number of failures','Flank wear', 'Build-up edge', 'Abrasive wear', 'Notching'] # User visible names
COLORS = ['length', 'width','centre_lenght', 'centre_width', 'num_instances', 'stepienie', 'narost', 'zatarcie', 'wykruszenie'] # Corresponding SQL colums names

# Base categories and information displayed in the Tooth report table
data = OrderedDict(
    [
        ("Parameter", ["Length", "Width", "Coordinate x", "Coordinate y", "Tooth flank wear", "Cumulated flank wear"]),
        ("Value [mm]", [0, 0, 0, 0, 0, 0]),
    ]
)
dat = pd.DataFrame(OrderedDict([(name, col_data) for (name, col_data) in data.items()]))
# Broach templates data
data_template = OrderedDict(
    [
        ("Parameter", ["Segment", "NumberOfRows", "NumberOfSections", "AngleOfAttack", "AngleOfClerance", "AngleOfBack"]),
        ("Value", [0, 0, 0, 0, 0, 0]),
    ]
)
dat_template = pd.DataFrame(OrderedDict([(name, col_data) for (name, col_data) in data_template.items()]))

# Available broaches
available_broach_indicators = BROACH_LIST # List of the all available scans
available_template_indicators = ['Example Type 1', 'Example Type 2', 'Example Type 3']

# Layout global styling parameters
cards_st ={"height": 660,"font-family": "Arial"}
cards_st_options ={"height": 480,"font-family": "Arial"}
cards_st_optimization={"height": 800,"font-family": "Arial"}
font_st = {'font-size':'20px','height':'35px',"font-family": "Arial"}

# Utilized graphics
PLOTLY_LOGO = r"http://www.mvlab.pl/images/logo.png"
TCM_LOGO = r"https://www.tcm-international.com/fileadmin/user_upload/TCM-Logo-klein.JPG"
polish = r"https://upload.wikimedia.org/wikipedia/en/thumb/1/12/Flag_of_Poland.svg/320px-Flag_of_Poland.svg.png"
english = r"https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Flag_of_the_United_Kingdom_%282-3%29.svg/150px-Flag_of_the_United_Kingdom_%282-3%29.svg.png"
german = r"https://upload.wikimedia.org/wikipedia/en/thumb/b/ba/Flag_of_Germany.svg/255px-Flag_of_Germany.svg.png"

# Bootstrap style
app = dash.Dash(__name__,external_stylesheets =[dbc.themes.LITERA],suppress_callback_exceptions=True)

def serve_layout():
    return html.Div([dcc.Location(id='url', refresh=True), html.Div(id='page-content')])
app.layout = serve_layout

# Uniform navbar for each page
def generate_navbar(active_page = 0):
    pages = [False,False,False,False]
    pages[active_page]=True
    navbar = dbc.Card([
        dbc.Row([
            # TCM logo
            dbc.Col([
                html.Div([
                    html.Img(src=TCM_LOGO, height="70px")
                ],style={"vertical-align": "middle"}) 
            ],
            style={"vertical-align": "middle", "margin-left":"30px",},
            width=1),
            # MV Lab logo
            dbc.Col([
                html.Div([
                    html.Img(src=PLOTLY_LOGO, height="70px")
                ],style={"vertical-align": "middle"}) 
            ],
            style={"vertical-align": "middle", "margin-left":"30px",},
            width=2),
            # APP Name
            dbc.Col([
                html.H2(["BROACH CONTROL APP"],style={"color":"#597cc7","margin-top":"10px","margin-bottom":"10px",}), 
            ],
            style={"text-color":"#597cc7",'display': 'left',"margin-left":"55px","margin-right":"0px"},
            width=4),
            # Pages
            dbc.Col([
                dbc.Breadcrumb(
                    items=[
                        {"label": "Menu",        "href": "/menu",         "external_link": True,    "active": pages[0]},
                        {"label": "Scanning",    "href": "/scanning",     "external_link": True,    "active": pages[1]},
                        {"label": "Optimization","href": "/optimization", "external_link": True,    "active": pages[2]},
                        {"label": "Options",     "href": "/options",      "external_link": True,    "active": pages[3]},
                    ],style=font_st
                )
            ],
            style={"vertical-align": "middle", "margin-left":"85px","margin-top":"15px","margin-bottom":"15px","padding":"0px"},
            width=2),
            # Languages
            dbc.Col([
                html.Div([
                    html.Img(src=polish, height="20px")
                ],style={"align": "right","margin-left":"105px"}),
                html.Div([
                    html.Img(src=english, height="20px")
                ],style={"vertical-align": "right","margin-left":"105px"}),
            ],
            style={"vertical-align": "right", "margin-left":"85px","margin-top":"5px","margin-bottom":"5px","padding":"0px"},
            width=1),
        ])
    ])
    return navbar

# Creating tables (optimization)
def generate_type_dropdown():
    sql = sql_connection.SQLConnection(debug=False) 
    data = pd.DataFrame(np.array(sql.get_table_values('TypeOfBroach')), columns=convert_sql_output(sql.get_table_names('TypeOfBroach')))
    sql = None
    
    return list(data["Project"])
def generate_broach_table():
    sql = sql_connection.SQLConnection(debug=False) 
    table_columns_names = convert_sql_output(sql.get_table_names('Broach'))
    table_values = sql.get_table_values('Broach')
    data = pd.DataFrame(data = table_values, columns = table_columns_names)
    sql = None
    
    return list(data["Project"])

# Used to proccess data recieved from sql to display it in datatable
def show_selected_params(project, return_type):
    # 1- return values 0- return columns names
    sql = sql_connection.SQLConnection(debug=False) 
    table_values = sql.get_table_values('TypeOfBroach')
    print(table_values)
    table_columns_names = convert_sql_output(sql.get_table_names('TypeOfBroach'))
    print(sql.get_table_values('TypeOfBroach'))
    data = pd.DataFrame(data= table_values, columns=table_columns_names)
    sql = None
    data = data[["Project","MinimalTootHeightLoss","MinimalToothLength","ModelBluntAngle","AngleOfAttack","AngleOfClerance","AngleOfBack","NominalTootHeight"]]
    data = data[data["Project"]==project]
    if(return_type==1): 
        return data.to_dict('records')
    else:
        return list(data.columns)
def show_selected_broach_params(scan_name, return_type):
    # 1 - return values 0 - return columns names
    sql = sql_connection.SQLConnection(debug=False)
    # Chose proper blunt vaule: used defined if exists deafoult if not
    y = convert_sql_output(sql.get_row_param(scan_name,'stepienie_row_value'))
    x = convert_sql_output(sql.get_row_param(scan_name,'row_number'))
    y_correction = convert_sql_output(sql.get_row_param(scan_name,'stepienie_correction'))
    blunt_value = []
    for i in range(len(y)):
        if y_correction[i] is not None: blunt_value.append(y_correction[i]) 
        else: blunt_value.append(y[i])
    
    # During saving to the SQL rows are saved in weried order for ex. 1,10,12,2,20 instead of 1,2,10,12,20
    # Due to this fact it is required to sort list of the teeth to find proper row number  

    # Find row number with the biggest cumulated blunt    
    max_blunt_value = max(blunt_value)
    available_rows = convert_sql_output(sql.get_row_param(scan_name,'row_number'))
    zipped_lists = zip(available_rows, blunt_value)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    _, blunt_value = [ list(tuple) for tuple in  tuples]
    max_blunt_row = blunt_value.index(max_blunt_value) + 1

    # Find row number with min mean lenght of the teeth row
    available_rows.sort()
    tooth_lengths = []
    for row in available_rows:
        l = convert_sql_output(sql.get_tooth_param(scan_name,'length',row))
        l = list(filter(None, l)) # Remmove None if exist in returned data
        try:
            L = (statistics.mean(l))
        except statistics.StatisticsError:
            L = 0
        tooth_lengths.append(L)
    min_tooth_length = min(tooth_lengths) 
    min_tooth_length_row = tooth_lengths.index(min_tooth_length) + 1

    
    sql = None
    data = {'Scan':[scan_name], 'Max blunt row no.':[max_blunt_row], 'Max blunt value [mm]':[round(max_blunt_value,3)],'Min lenght row no.':[min_tooth_length_row], 'Tooth min lenght [mm]':[round(min_tooth_length,3)]}
    data = pd.DataFrame(data,columns=['Scan','Max blunt row no.','Max blunt value [mm]','Min lenght row no.','Tooth min lenght [mm]'])
    if(return_type==1): 
        return data.to_dict('records')
    else:
        return list(data.columns)

# Used to generate dropdowns in "Add broach" table (options)        
def display_current_broach_types():
    sql = sql_connection.SQLConnection(debug=False) 
    table_columns_names = convert_sql_output(sql.get_table_names('TypeOfBroach'))
    table_values = sql.get_table_values('TypeOfBroach')
    data = pd.DataFrame(data = table_values, columns=table_columns_names)
    sql = None
    data = data[["Project"]]
    return list(data['Project'])

def update_types_dropdown():
    
    current_date = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dropdown={
        'BroachTypeID': {
            'options': [
                {'label': i, 'value': i}
                for i in display_current_broach_types()
                ]
        },
        'DateOfEntry': {
            'options': [
                {'label': current_date, 'value': current_date}
                ]
        },
        'ScrappingDate': {
            'options': [
                {'label': current_date, 'value': current_date}
                ]
        },
    }
    return dropdown

# Optimization regeneration types table column names (to keep names used as keys in various planes in one place)
type_I = ['Time [min]','Cost [PLN]','Operations no. front','Front surface [μm]']
type_top = ['Time [min]','Cost [PLN]','Operations no. top','Top surface [μm]']
type_II = ['Time [min]','Cost [PLN]','Operations no. front','Front surface [μm]','Operations no. top','Top surface [μm]']

# Layout structures
scanning = html.Div([
    # Navbar 
    generate_navbar(1),
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
                                    {'label': ' Build-up edge\n', 'value': 1},
                                    {'label': ' Flank wear\n', 'value': 2},
                                    {'label': ' Abrasive wear\n', 'value': 3},
                                    {'label': ' Notching\n', 'value': 4}
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
                            html.H3('Modify flank wear value:'),

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
                            html.H2('Cumulated flank wear calculation'),
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

optimization = html.Div([
    generate_navbar(2),
    html.Br(),
    dbc.Card([
        dbc.Row([
           # Available oarameters
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2('Available parameters'),
                        ),
                        dbc.CardBody([
                            html.H3('Select scan:'),
                            # Scan select
                            dcc.Dropdown(
                                id = 'scan-select',
                                options=[{'label': i, 'value': i} for i in available_broach_indicators],
                                value = available_broach_indicators[-1], 
                                style = font_st
                            ),
                            html.Br(),
                            # Scan data
                            dash_table.DataTable(
                                id='scan-simple-datatable',
                                data = show_selected_broach_params(available_broach_indicators[-1],1),
                                columns = [{'id': c, 'name': c} for c in show_selected_broach_params(available_broach_indicators[-1],0)],
                                page_size = 10,
                                style_cell = {'font-size':'20px',"font-family": "Arial",'minWidth': '180px'},
                                style_header = {'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial", 
                                            'height': '50px'},
                                style_table = {'overflowX': 'auto','minWidth': '100%'},
                                fixed_columns = { 'headers': True, 'data': 1 },     
                            ),
                            html.Br(),
                            html.H3('Select broach type:'),
                            # Broach type select
                            dcc.Dropdown(
                                id = 'type-select',
                                options=[{'label': i, 'value': i} for i in generate_type_dropdown()],
                                value = generate_type_dropdown()[0], 
                                style = font_st
                            ),
                            html.Br(),
                            # Broach type data 
                            dash_table.DataTable(
                                id='params-simple-datatable',
                                data = show_selected_params("1055",1),
                                columns = [{'id': c, 'name': c} for c in show_selected_params("1055",0)],
                                page_size = 10,
                                style_cell = {'font-size':'20px',"font-family": "Arial",'minWidth': '180px'},
                                style_header = {'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial", 
                                            'height': '50px'},
                                style_table = {'overflowX': 'auto','minWidth': '100%'},
                                fixed_columns = { 'headers': True, 'data': 1 },     
                            ),
                            html.Br(),
                            # Scan notes
                            html.H3('Scan notes:'),
                            dcc.Textarea(
                                id='scan-notes',
                                value='Textarea content initialized\nwith multiple lines of text',
                                style={'width': '100%', 'height': 140,'font-size':'20px'},
                            ),
                            # Scan notes submit
                            html.Div([
                                dbc.Button('Submit note', id='note-submit-button', color="secondary" ,n_clicks=0, style={'font-size': '20px'}), 
                            ]),
                        ]),
                    ],style = cards_st_optimization),
                ]),
                width=6,
            ),
            # Possible regenerations
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2('Regeneration possibilities'),
                        ),
                        dbc.CardBody([
                            html.H3('Regeneration type I "front surface only":'),
                            dash_table.DataTable(
                                id='regeneration_I',
                                data = None,
                                columns = [{'id': c, 'name': c} for c in type_I],
                                page_size = 1,
                                style_cell = {'font-size':'20px',"font-family": "Arial",'minWidth': '180px'},
                                style_header = {'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial", 'height': '50px'},
                                style_table = {'overflowX': 'auto','minWidth': '100%'},  
                            ),
                            html.Br(), 
                            html.H3('Regeneration type "top surface only":'),
                            dash_table.DataTable(
                                id='regeneration_z_gory',
                                data = None,
                                columns = [{'id': c, 'name': c} for c in type_top],
                                page_size = 1,
                                style_cell = {'font-size':'20px',"font-family": "Arial",'minWidth': '180px'},
                                style_header = {'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial", 'height': '50px'},
                                style_table = {'overflowX': 'auto','minWidth': '100%'},
                            ), 
                            html.Br(),
                            html.H3('Regeneration type II "mixed":'),
                            dash_table.DataTable(
                                id='regeneration_II',
                                data = None,
                                columns = [{'id': c, 'name': c} for c in type_II],
                                page_size = 10,
                                style_cell = {'font-size':'20px',"font-family": "Arial",'minWidth': '180px'},
                                style_header = {'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial", 'height': '50px'},
                                style_table = {'overflowX': 'auto','minWidth': '100%'},
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{{Front surface [μm]}} > {}'.format(50)},
                                        'backgroundColor': '#FF4136',
                                        'color': 'white'
                                    }
                                ]
                            ),             
                        ]),
                    ],style = cards_st_optimization),
                ]),
                width=6,
            ),  
        ]), 
        html.Br(), 
        dbc.CardFooter()  
    ],
    style = {'margin-left':'20px','margin-right':'20px',"font-family": "Arial"},
    color = "light", 
    outline = True)    

])

options = html.Div([
    generate_navbar(3),
    html.Br(),
    dbc.Card([
        dbc.Row([
           # Available broach templates
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2('Available broach templates'),
                        ),
                        dbc.CardBody([
                            # Add broach template 
                            dash_table.DataTable(
                                id='params-datatable',
                                data = params_data,
                                columns = [{'id': c, 'name': c} for c in table_columns_names],
                                page_size = 6,
                                style_cell = {'font-size':'20px',"font-family": "Arial",'minWidth': '180px'},
                                style_header = {'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial", 
                                            'height': '50px'},
                                style_table = {'overflowX': 'auto','minWidth': '100%'},
                                fixed_columns = { 'headers': True, 'data': 1 },     
                                editable = True,
                                row_deletable = True
                            ),
                            html.Br(),         
                            # Buttons
                            dbc.Row([
                                dbc.Col(
                                    html.Div([
                                        dbc.Button('Add new type', id='editing-rows-button', color="secondary" ,n_clicks=0, style={'font-size': '20px'}), 
                                    ]),
                                    width=11
                                ),                                  
                                dbc.Col(
                                    html.Div([
                                        dbc.Button('Submit', id='submit-rows-button', color="secondary" ,n_clicks=0, style={'font-size': '20px'}), 
                                    ]),
                                    width=1
                                )                                
                            ]),
                        ]),
                    ],style = cards_st_options),
                    # Info message
                    dbc.Alert(
                        "Input all template parameters and press submit to save record in the database",
                        color="info",
                        id = 'info-alert',
                        className="d-flex align-items-center",
                    ),                    
                ]),
                width=12,    
            ), 
            html.Br(), 
           # Existing broaches
            dbc.Col(
                html.Div([
                    dbc.Card([
                        dbc.CardHeader(
                            html.H2('Existing broaches'),
                        ),
                        dbc.CardBody([
                            # Add broach 
                            dash_table.DataTable(
                                id='broach-datatable',
                                data = broach_data,
                                columns = [{'id': c, 'name': c,'presentation': 'dropdown'} for c in broach_columns_names],
                                page_size = 6,
                                style_cell = {'font-size':'20px',"font-family": "Arial",'minWidth': '180px','maxWidth': '220px'},
                                style_header = {'textAlign': 'center', 'fontWeight': 'bold', 'margin':'10px','font-size':'20px',"font-family": "Arial", 'height': '50px'},
                                editable = True,
                                row_deletable = True,
                                css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}], # Needed to fix displaying dropdown in table
                                dropdown = update_types_dropdown()
                            ),
                            html.Div(id='broach-datatable-container'),
                            html.Br(),         
                            # Buttons
                            dbc.Row([
                                dbc.Col(
                                    html.Div([
                                        dbc.Button('Add new broach', id='editing-broach-button', color="secondary" ,n_clicks=0, style={'font-size': '20px'}), 
                                    ]),
                                    width=11
                                ),                                  
                                dbc.Col(
                                    html.Div([
                                        dbc.Button('Submit', id='submit-broach-button', color="secondary" ,n_clicks=0, style={'font-size': '20px'}), 
                                    ]),
                                    width=1
                                )                                
                            ]),
                        ]),
                    ],style = cards_st_options),  
                    # Info message 2
                    dbc.Alert(
                        "Input new broach and press submit to save record in the database",
                        color="info",
                        id = 'info-alert2',
                        className="d-flex align-items-center",
                    ),                
                ]),
                width=12,    
            ),             
        ]), 
        html.Br(), 
        dbc.CardFooter()  
    ],
    style = {'margin-left':'20px','margin-right':'20px',"font-family": "Arial"},
    color = "light", 
    outline = True)
])

index_page = html.Div([
    generate_navbar(0),
    html.Div([
        dbc.Card([
            dbc.CardHeader(
                html.H2('Menu główne'),
            ),
            dbc.CardBody([
                html.Div([
                    html.Ul([
                        html.Li(dcc.Link('SKAN', href='/scanning')),
                        html.Li(dcc.Link('OPTYMALIZACJA', href='/optimization')),
                        html.Li(dcc.Link('OPCJE', href='/options'))
                    ]),
                    html.Br(),
                    html.P('Wersja testowa aplikacji', style = font_st),
                ]),
            ])                    
        ])
    ])

])

# Update displayed page
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/scanning':
        return scanning
    elif pathname == '/optimization':
        return optimization
    elif pathname == '/options':
        return options
    else:
        return index_page


##-------------------------------Optimization-------------------------------------##

# Scan select
@app.callback(
    Output('scan-simple-datatable', 'data'),
    Input('scan-select', 'value'),
)
def get_params(default_broach):
    return show_selected_broach_params(default_broach,1)

# Broach type select
@app.callback(
    Output('type-simple-datatable', 'data'),
    Input('type-select', 'value'),
)
def get_params(default_broach):
    return show_selected_params(default_broach,1)

# Regeneration I
@app.callback(
    Output('regeneration_I', 'data'),
    [Input('scan-simple-datatable', 'data'),
     Input('params-simple-datatable', 'data')]
)
def create_output(scan_data,type_data):
    stepienie = scan_data[0]['Max blunt value [mm]']
    dlugosc = scan_data[0]['Tooth min lenght [mm]']
    czas, koszt, ilosc_przejsc = optimize.regeneration_I_calculate_outputs(stepienie)
    print(czas,koszt)
    data = {type_I[0]:[round(czas,0)], type_I[1]:[round(koszt,2)],type_I[2]:[ilosc_przejsc], type_I[3]:[stepienie*1000]}
    data = pd.DataFrame(data,columns=type_I)
    print(data.to_dict('records'))
    return data.to_dict('records')

# Regeneration "z góry"
@app.callback(
    Output('regeneration_z_gory', 'data'),
    [Input('scan-simple-datatable', 'data'),
     Input('params-simple-datatable', 'data')]
)
def create_output(scan_data,type_data):
    stepienie = scan_data[0]['Max blunt value [mm]']
    dlugosc = scan_data[0]['Tooth min lenght [mm]']
    czas, koszt, ilosc_przejsc, stepienie = optimize.regeneration_z_gory_calculate_outputs(stepienie)
    print(czas,koszt)
    data = { type_top[0]:[round(czas,0)], type_top[1]:[round(koszt,2)], type_top[2]:[ilosc_przejsc], type_top[3]:[stepienie*1000]}
    data = pd.DataFrame(data,columns=type_top)
    print(data.to_dict('records'))
    return data.to_dict('records')

# Regeneration II
@app.callback(
    Output('regeneration_II', 'data'),
    [Input('scan-simple-datatable', 'data'),
     Input('params-simple-datatable', 'data')]
)
def create_output(scan_data,type_data):
    stepienie = scan_data[0]['Max blunt value [mm]']
    dlugosc = scan_data[0]['Tooth min lenght [mm]']
    r = optimize.regeneration_II_calculate_outputs(stepienie)
    data = {type_II[0]:[i[0] for i in r], type_II[1]:[i[1] for i in r], 
            type_II[2]:[i[2] for i in r], type_II[3]:[i[3] for i in r], 
            type_II[4]:[i[4] for i in r], type_II[5]:[i[5] for i in r]}
    data = pd.DataFrame(data,columns=type_II)
    return data.to_dict('records')

@app.callback(
    Output('scan-notes', 'value'),
    Input('scan-select', 'value')
)
def update_output(value):
    sql = sql_connection.SQLConnection(debug=False)
    note = str(convert_sql_output(sql.get_scan_param(param_name ='scan_notes', scan_name = value))[0] )
    if note == 'None': note = ''
    sql = None
    return note

'''
# Add new broach button
@app.callback(
    Output('broach-datatable', 'data'),
    Input('note-submit-button', 'n_clicks'),
    State('broach-datatable', 'data'),
    State('broach-datatable', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        # Add new line when button cliked at least once
        rows.append({c['id']: '' for c in columns}) 
        return rows
    else:  
        # Refreshing table on reload
        sql = sql_connection.SQLConnection(debug=True)
        table_columns_names = convert_sql_output(sql.get_table_names('Broach'))
        params_df = pd.DataFrame(sql.get_table_values('Broach'))
        params_df.columns = table_columns_names
        rows = params_df.to_dict('records') 
        sql = None
        return rows 
'''
@app.callback(
    Output('regeneration_II', 'style_data_conditional'),
    [Input('scan-simple-datatable', 'data'),
    Input('params-simple-datatable', 'data')]
)
def create_output(scan_data,type_data):
    lenght = scan_data[0]['Tooth min lenght [mm]']
    #min_acceptable_lenght = type_data[0]['MinimalToothLenght']*1000 #[μm] 
    min_acceptable_lenght = 1200 #[μm]
    lenght*=1000 #[μm]
    return conditional_table_cells_formating(type_II[3],lenght, min_acceptable_lenght, 50, type_II[5], 80, 40, 20)

@app.callback(
    Output('regeneration_I', 'style_data_conditional'),
    [Input('scan-simple-datatable', 'data'),
    Input('params-simple-datatable', 'data')]
)
def create_output(scan_data,type_data):
    lenght = scan_data[0]['Tooth min lenght [mm]']
    lenght*=1000 #[μm]
    min_acceptable_lenght = 1200 #[μm] 
    return conditional_table_cells_formating(type_I[3],lenght, min_acceptable_lenght, 50, None, 80, 40, 20)

@app.callback(
    Output('regeneration_z_gory', 'style_data_conditional'),
    [Input('scan-simple-datatable', 'data'),
    Input('params-simple-datatable', 'data')]
)
def create_output(scan_data,type_data):
    lenght = scan_data[0]['Tooth min lenght [mm]']
    lenght*=1000 #[μm]
    min_acceptable_lenght = 1200 #[μm] 
    min_acceptable_height = 80 #[μm]
    return conditional_table_cells_formating(None,lenght, min_acceptable_lenght, 50, type_top[3], 80, 40, 20)


def conditional_table_cells_formating(field,value,border,offest,field2,value2,border2,offest2):
    style_data_conditional=[
        {
        'if': {
            'filter_query': '{{{}}} > {}'.format('Front surface [μm]',value-border-offest),
            'column_id': '{}'.format(field)
        },
        'backgroundColor': '#FFFF00',
        },
        {
        'if': {
            'filter_query': '{{{}}} > {}'.format('Front surface [μm]',value-border),
            'column_id': '{}'.format(field)
        },
        'backgroundColor': '#FF4500',
        },
        {
        'if': {
            'filter_query': '{{{}}} > {}'.format('Top surface [μm]',value2-border2-offest2),
            'column_id': '{}'.format(field2)
        },
        'backgroundColor': '#FFFF00',
        },
        {
        'if': {
            'filter_query': '{{{}}} > {}'.format('Top surface [μm]',value2-border2),
            'column_id': '{}'.format(field2)
        },
        'backgroundColor': '#FF4500',
        }
    ]
    return style_data_conditional    

##---------------------------------Options---------------------------------------##

# Add new broach type button
@app.callback(
    Output('params-datatable', 'data'),
    Input('editing-rows-button', 'n_clicks'),
    State('params-datatable', 'data'),
    State('params-datatable', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        # Add new line when button cliked at least once
        rows.append({c['id']: '' for c in columns}) 
        return rows
    else:  
        # Refreshing table on reload
        sql = sql_connection.SQLConnection(debug=True)
        table_columns_names = convert_sql_output(sql.get_table_names('TypeOfBroach'))
        table_values = sql.get_table_values('TypeOfBroach')
        params_df = pd.DataFrame(data= table_values, columns=table_columns_names)
        rows = params_df.to_dict('records') 
        sql = None
        return rows     

# Info alert - params   
@app.callback(
    [Output('info-alert', 'color'),
    Output('info-alert', 'children'),
    Output('broach-datatable','dropdown')],
    Input('submit-rows-button', 'n_clicks'),
    State('params-datatable', 'data')
    )
def upload_to_sql(n_clicks,data):
    print('nclicks',n_clicks)
    if n_clicks > 0:
        sql = sql_connection.SQLConnection(debug=True)
        success = sql.update_broach_params(data)
        table_columns_names = convert_sql_output(sql.get_table_names('TypeOfBroach'))
        table_values = sql.get_table_values('TypeOfBroach')
        params_df = pd.DataFrame(data = table_values, columns = table_columns_names)
        params_data = params_df.to_dict('records')
        sql = None
        if(success == 1):return 'success',"Data updated successfuly.",update_types_dropdown()
        else: return 'danger',"Data doesn't match predefined types or there are some blank columns.",update_types_dropdown()  
    else: return 'info',"Input all template parameters and press submit to save record in the database.",update_types_dropdown()

# Add new broach button
@app.callback(
    Output('broach-datatable', 'data'),
    Input('editing-broach-button', 'n_clicks'),
    State('broach-datatable', 'data'),
    State('broach-datatable', 'columns'))
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        # Add new line when button cliked at least once
        rows.append({c['id']: '' for c in columns}) 
        return rows
    else:  
        # Refreshing table on reload
        sql = sql_connection.SQLConnection(debug=True)
        table_columns_names = convert_sql_output(sql.get_table_names('Broach'))
        table_values = sql.get_table_values('Broach')
        params_df = pd.DataFrame(data = table_values, columns = table_columns_names)
        rows = params_df.to_dict('records') 
        sql = None
        return rows   

# Info alert - broach 
@app.callback(
    [Output('info-alert2', 'color'),
    Output('info-alert2', 'children')],
    Input('submit-broach-button', 'n_clicks'),
    State('broach-datatable', 'data')
    )
def upload_to_sql(n_clicks,data):
    print('nclicks',n_clicks)
    if n_clicks > 0:
        sql = sql_connection.SQLConnection(debug=True)
        success = sql.define_new_broach(data)
        table_columns_names = convert_sql_output(sql.get_table_names('Broach'))
        table_values = sql.get_table_values('Broach')
        params_df = pd.DataFrame(data = table_values, columns = table_columns_names)
        params_data = params_df.to_dict('records')
        sql = None
        if(success == 1):return 'success',"Data updated successfuly."
        else: return 'danger',"Data doesn't match predefined types."
    else: return 'info',"Input broach parameters and press submit to save record in the database."

##---------------------------------Scanning---------------------------------------##

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
    if c == 'stepienie':    
        color_scale_pick = ["lime","yellow","tomato","red"] 
        showscale = True
    elif (c == 'zatarcie' or c == 'wykruszenie'): 
        color_scale_pick = [[0, "lime"], 
                            [0.51, "lime"],
                            [0.51, "tomato"],  
                            [1, "tomato"]]
        showscale = False
    elif (c == 'narost'): 
        color_scale_pick = [[0, "yellow"], 
                            [0.51, "yellow"],
                            [0.51, "navy"],  
                            [1, "navy"]] 
        showscale = False
    else: 
        color_scale_pick = px.colors.sequential.Inferno
        showscale = True

    fig.add_trace(
        go.Scatter(
            x = x, 
            y = y, 
            marker = dict(
                color = v,
                colorscale = color_scale_pick,
                showscale = showscale
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
     State('input-on-submit-row', 'value'),
     State('broach','value')])
def update_output(n_clicks, blunt_value,row_number,default_broach):
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
                      yaxis=dict(title_text = "Flank wear", titlefont=dict(size=20)),
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
            ("Parameter", ["Lenght", "Width", "Coordinate x", "Coordinate y", "Tooth flank wear", "Cumulated flank wear"]),
            ("Value [mm]", [display[0], display[1], display[2], display[3], display[4], st]),
        ]
    )
    dat = pd.DataFrame(OrderedDict([(name, col_data) for (name, col_data) in data.items()]))
    dat = dat.to_dict('records')

    sql = None
    return dat, IMAGE_NAME


if __name__ == '__main__':
    app.run_server(debug=False,port=8080,host='0.0.0.0')
