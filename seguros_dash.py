import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import plotly.express as px
import os
#---------------------------------------------Data-------------------------------------------------------------


#Lectura y definición de tipos de variables
df = pd.read_csv('data/seguros_v2.csv')
df2 = pd.read_csv('data/clean_data.csv')
#Variables Target
trg = ['TARGET_FLAG', 'TARGET_AMT']
#Variables Numericas
con = ['AGE','BLUEBOOK','CAR_AGE','CLM_FREQ','HOMEKIDS','HOME_VAL','INCOME','MVR_PTS','OLDCLAIM','TIF','TRAVTIME','YOJ']
#Variables Cateoricas
cat = ['CAR_TYPE','CAR_USE','EDUCATION','JOB','KIDSDRIV','MSTATUS','PARENT1','RED_CAR','REVOKED','SEX','URBANICITY']

#Lctura guardado de modelos
modelos = pd.DataFrame(columns = ['modelo','graph','accuracy','variables','columnas','pesos'])
for file in os.listdir("data/modelos/"):
    save = np.load("data/modelos/"+file,allow_pickle=True)
    modelos = modelos.append(save.item(),ignore_index=True)
#----------------------------------------Listas Desplegabes-----------------------------------------
pie_drop_l = [{'label':'TARGET_FLAG','value':trg[0]},
              {'label':'CAR_TYPE','value':cat[0]},
              {'label':'CAR_USE','value':cat[1]},
              {'label':'EDUCATION','value':cat[2]},
              {'label':'JOB','value':cat[3]},
              {'label':'KIDSDRIV','value':cat[4]},
              {'label':'MSTATUS','value':cat[5]},
              {'label':'PARENT1','value':cat[6]},
              {'label':'RED_CAR','value':cat[7]},
              {'label':'REVOKED','value':cat[8]},
              {'label':'SEX','value':cat[9]},
              {'label':'URBANICITY','value':cat[10]}]
#---------------------------------------------------------------------
hist_drop_l = [{'label':'TARGET_AMT','value':trg[1]},
               {'label': 'AGE','value':con[0]},
               {'label': 'BLUEBOOK','value':con[1]},
               {'label': 'CAR_AGE','value':con[2]},
               {'label': 'CLM_FREQ','value':con[3]},
               {'label': 'HOMEKIDS','value':con[4]},
               {'label': 'HOME_VAL','value':con[5]},
               {'label': 'INCOME','value':con[6]},
               {'label': 'MVR_PTS','value':con[7]},
               {'label': 'OLDCLAIM','value':con[8]},
               {'label': 'TIF','value':con[9]},
               {'label': 'TRAVTIME','value':con[10]},
               {'label': 'YOJ','value':con[11]}]
#----------------------------------------------------------------------
media_drop_l = [{'label':'CAR_TYPE','value':cat[0]},
              {'label':'CAR_USE','value':cat[1]},
              {'label':'EDUCATION','value':cat[2]},
              {'label':'JOB','value':cat[3]},
              {'label':'KIDSDRIV','value':cat[4]},
              {'label':'MSTATUS','value':cat[5]},
              {'label':'PARENT1','value':cat[6]},
              {'label':'RED_CAR','value':cat[7]},
              {'label':'REVOKED','value':cat[8]},
              {'label':'SEX','value':cat[9]},
              {'label':'URBANICITY','value':cat[10]}]
#--------------------------------------------------------------------------
map_drop_l = [{'label':'TARGET_FLAG','value':trg[0]},
                {'label':'TARGET_AMT','value':trg[1]}]
#--------------------------------------------------------------------------
logm_drop_l = [{'label':'Modelo 1','value':0},
               {'label':'Modelo 2','value':4},
               {'label':'Modelo 3','value':3}]
#--------------------------------------------------------------------------
regm_drop_l = [{'label':'Modelo 1','value':5},
               {'label':'Modelo 2','value':1},
               {'label':'Modelo 3','value':2}]
modelolog = "El modelo de regesión logistica se usa para predecir sobre la variable {}, para este modelo se corrieron tres versiones distintas del modelo.".format(trg[0])
modelolog1 = "* Modelo 1: variables {} y {} con un error de {:.4f} en binary cross entropy.".format(modelos['variables'][0][0],modelos['variables'][0][1],modelos['accuracy'][0])
modelolog2 = "* Modelo 2: variables {} , {} , {} , {} y {} con un error de {:.4f} en binary cross entropy.".format(modelos['variables'][4][0],modelos['variables'][4][1],modelos['variables'][4][2],modelos['variables'][4][3],modelos['variables'][4][4],modelos['accuracy'][4])
modelolog3 = "* Modelo 3: variables {} , {} , {} , {} , {} y {} con un error de {:.4f} en binary cross entropy.".format(modelos['variables'][3][0],modelos['variables'][3][1],modelos['variables'][3][2],modelos['variables'][3][3],modelos['variables'][3][4],modelos['variables'][3][6],modelos['accuracy'][3])
modeloreg = "El modelo de regesión lineal se usa para predecir sobre la variable {}, para eset modelo se corrieron tres versiones distintas del modelo.".format(trg[1])
modeloreg1 = "* Modelo 1: variables {} y {} con un error de {:.4f} en MSE.".format(modelos['variables'][0][0],modelos['variables'][0][1],modelos['accuracy'][5])
modeloreg2 = "* Modelo 2: variables {} , {} , {} , {} y {} con un error de {:.4f} en MSE.".format(modelos['variables'][4][0],modelos['variables'][4][1],modelos['variables'][4][2],modelos['variables'][4][3],modelos['variables'][4][4],modelos['accuracy'][1])
modeloreg3 = "* Modelo 3: variables {} , {} , {} , {} , {} y {} con un error de {:.4f} en MSE.".format(modelos['variables'][3][0],modelos['variables'][3][1],modelos['variables'][3][2],modelos['variables'][3][3],modelos['variables'][3][4],modelos['variables'][3][6],modelos['accuracy'][3])
clean_text = 'El proceso de limpieza se llevó a cabo de la siguiente manera. Primero se escogieron las variables con una correlación superior ({}, {}, {}, {}, {}, {}) al resto y se buscó eliminar registros atípicos eliminando todos aquellos que sobrepasaron el cuartil 99.5 pasando de 8161 registros a 7098. Esto además ayudó a eliminar registros nulos. Segundo se transformaron las variables categóricas a variables dummy. Tercero, todas las variables que se usaron eventualmente en el modelo fueron normalizadas. Para el caso del modelo de regresión lineal la variable Y también se transformó con el uso de ln(). También se eliminaron algunos registros que no concordaban con el tipo de data como edades de caros negativas, también se estandarizaron todos los labels de la variables categóricas antes de su transformación.'.format('TARGET_AMT','BLUEBOOK','HOME_VAL','INCOME','OLDCLAIM','TRAVTIME')
#-------------------------------------------------Dash-------------------------------------------


app = dash.Dash()
app.layout= html.Div([
    #Encabezado
    html.Br(),
    html.Div([
        #Logo
        html.Div([
            html.Img(src=app.get_asset_url("images/Deloitte.png"),
             className='six columns',
             style={'height': '15%',
                    'width': '15%',
                    'float': 'right',
                    'position': 'relative',
                    }) 
        ],className='row'),
        #Titulo
        html.Div([
            html.H1(children='Modelo de Seguros de Autos',style={'text-align': 'center'}),
        ],className='row'),
    ]),
    
    
    
    
    #Exploración
    html.Br(),
    html.Div([
        html.H2(children='Exploración y análisis',style={'text-aling':'rigth'}),
        html.Br(),
        
        #Primera Linea de 2 Graficos
        html.Div([
            #Pie            
            html.Div([
                html.H3(children='Proporción de etiquetas',style={'text-align': 'center'}),
                dcc.Dropdown(
                    id = 'pie_drop',
                    options = pie_drop_l,
                    placeholder='elegir una variable categorica',
                    #value = pie_drop_l[0],
                ),
                dcc.Graph(id='pie',figure={}),
            ],className='six columns'),
            #Hist
            html.Div([
                html.H3(children='Histograma variables continuas',style={'text-align': 'center'}),
                dcc.Dropdown(
                    id = 'hist_drop',
                    options = hist_drop_l,
                    placeholder='elegir una variable continua',
                    value = hist_drop_l[0],
                ),
                dcc.RadioItems(
                    id = 'hist_check',
                    options = [{'label': 'Ambos','value':'ambos'},
                               {'label': 'Colición','value':'crash'},
                              {'label': 'No colición','value':'libre'}],
                    value = 'ambos'
                ),
                
                dcc.Graph(id='hist',figure={}),
            ],className='six columns'),            
            
        ],className='row'),
        
        
        
        #Segunda Linea de 2 Graficos
        html.Div([
            
            #Dispersión de puntos
            html.Div([
                html.H3(children='Mapa de dispersión de puntos',style={'text-align': 'center'}),
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id = 'dist_drop1',
                            options = hist_drop_l,
                            placeholder='variable eje X',
                            #value = media_drop_l[0],
                        ),
                        
                    ],className='six columns'),
                    html.Div([
                        dcc.Dropdown(
                            id = 'dist_drop2',
                            options = hist_drop_l,
                            placeholder='variable eje Y',
                            #value = media_drop_l[0],
                        ),
                        
                        
                    ],className='six columns'),
                ],className='row'),
                dcc.RadioItems(
                    id = 'dist_check',
                    options = [{'label': 'Ambos','value':'ambos'},
                               {'label': 'Colición','value':'crash'},
                               {'label': 'No colición','value':'libre'}],
                    value = 'ambos'
                ),                
                
                dcc.Graph(id='dist',figure={}),
            ],className='six columns'),
            
            #BoxPlot
            html.Div([
                html.H3(children='Grafica de caja',style={'text-align': 'center'}),
                dcc.Dropdown(
                    id = 'box_drop',
                    options = media_drop_l,
                    placeholder='elegir una variable continua',
                    #value = map_drop_l[0],
                ),
                dcc.RadioItems(
                    id = 'box_check',
                    options = [{'label':'Outliers' ,'value':'outliers'},
                              {'label':'Todos los puntos' ,'value':'all'},
                              {'label':'Ningun punto' ,'value':False}],
                    value = False
                ),
                
                dcc.Graph(id='box',figure={}),
            ],className='six columns'),            
            
        ],className='row'),
        
        #Tercera Linea de 2 Graficos
        html.Div([
            
            #Media según Label
            html.Div([
                html.H3(children='Media del valor reclamado según etiqueta',style={'text-align': 'center'}),
                dcc.Dropdown(
                    id = 'media_drop',
                    options = media_drop_l,
                    placeholder='elegir una variable categorica',
                    #value = media_drop_l[0],
                ),
                dcc.Graph(id='media',figure={}),
            ],className='six columns'),
            
            #Mapa de calor
            html.Div([
                html.H3(children='Mapa de correlaciones',style={'text-align': 'center'}),
                dcc.Dropdown(
                    id = 'map_drop',
                    options = map_drop_l,
                    placeholder='elegir una variable continua',
                    #value = map_drop_l[0],
                ),
                
                dcc.Graph(id='map',figure={}),
            ],className='six columns'),            
            
        ],className='row'),
    ]),
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #Preparación de la data
    html.Br(),
    html.Br(),
    html.Div([
        html.H2(children='Limpieza y preparación',style={'text-aling':'rigth'}),
        html.Br(),
               
        #Primera Linea de 2 Graficos
        html.Div([
            
            
            #Distribución despues de limpieza
            html.Div([
                html.H3(children='Distribución despues de limpieza',style={'text-align': 'center'}),
                dcc.Dropdown(
                    id = 'hist2_drop',
                    options = hist_drop_l,
                    placeholder='elegir una variable continua',
                    value = hist_drop_l[0],
                ),
                dcc.RadioItems(
                    id = 'hist2_check',
                    options = [{'label': 'Ambos','value':'ambos'},
                               {'label': 'Colición','value':'crash'},
                              {'label': 'No colición','value':'libre'}],
                    value = 'ambos'
                ),
                
                dcc.Graph(id='hist2',figure={}),
            ],className='six columns'),  
            #Clean Text
            html.Div([
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.H6(children=clean_text,style={'text-align': 'left'}),
            ],className='six columns'),            
            
        ],className='row'),
    ]),
    
    
    
    #Modelos
    html.Br(),
    html.Br(),
    html.Div([
        html.H2(children='Modelos',style={'text-aling':'rigth'}),
        html.Br(),
        #Logos
        html.Div([
            #Python
            html.Div([
                html.Img(src=app.get_asset_url("images/python.png"),
                         style={'height': '15%',
                                'width': '15%',
                                'float': 'right',
                                'position': 'relative',
                                }),
            ],className='three columns'),
            #Jupyter
            html.Div([
                html.Img(src=app.get_asset_url("images/jupyter.png"),
                         style={'height': '15%',
                                'width': '15%',
                                'float': 'right',
                                'position': 'relative',
                                }),
            ],className='three columns'),
            #Torch
            html.Div([
                html.Img(src=app.get_asset_url("images/pytorch.png"),
                         style={'height': '15%',
                                'width': '15%',
                                'float': 'right',
                                'position': 'relative',
                                }),
            ],className='three columns'),
            #Numpy
            html.Div([
                html.Img(src=app.get_asset_url("images/numpy.png"),
                         style={'height': '15%',
                                'width': '15%',
                                'float': 'right',
                                'position': 'relative',
                                }),
            ],className='three columns'),
            
            
        ],className='row'),
               
        #Primera Linea de 2 Graficos
        html.Br(),
        html.Div([
            
            
            #Modelo de regresión logistica
            html.Div([
                html.H3(children='Modelo de regresión logística',style={'text-align': 'center'}),
                #html.
                dcc.Dropdown(
                    id = 'logm_drop',
                    options = logm_drop_l,
                    placeholder='elegir un modelo',
                    multi=True,
                ),
                
                dcc.Graph(id='logm',figure={}),
                html.H6(children=modelolog,style={'text-align': 'left'}),
                html.H6(children=modelolog1,style={'text-align': 'left'}),
                html.H6(children=modelolog2,style={'text-align': 'left'}),
                html.H6(children=modelolog3,style={'text-align': 'left'}),
            ],className='six columns'),  
            
            
            #Modelo de regresión lineal
            html.Div([
                html.H3(children='Modelo de regresión lineal',style={'text-align': 'center'}),
                #html.
                dcc.Dropdown(
                    id = 'regm_drop',
                    options = regm_drop_l,
                    placeholder='elegir un modelo',
                    multi=True,
                ),
                
                dcc.Graph(id='regm',figure={}),
                html.H6(children=modelolog,style={'text-align': 'left'}),
                html.H6(children=modeloreg1,style={'text-align': 'left'}),
                html.H6(children=modeloreg2,style={'text-align': 'left'}),
                html.H6(children=modeloreg3,style={'text-align': 'left'}),
            ],className='six columns'),             
            
        ],className='row'),
    ]),
html.Br(),
html.Br(),
html.Br(),
html.Br(),
], className='ten columns offset-by-one')

#-------------------------------------------Graficas---------------------------------------------------------------

@app.callback(Output(component_id='pie', component_property='figure'),
    [Input(component_id='pie_drop', component_property='value')]
)
def plot_pie(option_slctd):
    df_gf = df[option_slctd].value_counts()
    fig = go.Figure(data=[go.Pie(labels=df_gf.index, values=df_gf, textinfo='label+percent',insidetextorientation='auto',title=option_slctd)])
    fig.update_layout(showlegend=False)
    return fig
#--------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='hist', component_property='figure'),
    [Input(component_id='hist_drop', component_property='value'),
     Input(component_id='hist_check', component_property='value')]
)
def plot_histogram(X,opt):
    if opt == 'ambos':
        #Dataset completo
        x = df[X]
    elif opt == 'crash':
        #Dataset filtrado por coque
        x = df[df[trg[0]]==1][X]
    elif opt == 'libre':
        #Dataset filtrado por libre de accidente
        x = df[df[trg[0]]==0][X]
    fig = go.Figure(data=[go.Histogram(x=x)])
    fig.update_layout(title_text=X)
    return fig
#---------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='media', component_property='figure'),
    [Input(component_id='media_drop', component_property='value')]
)
def plot_mean_by_label(i):
    group = df.groupby([i]).mean()[trg[1]]
    fig = go.Figure([go.Bar(x=group.index, y=group.values)])
    fig.update_layout(title='Media de {} agrupada por {}'.format(trg[1],i))
    return fig
#----------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='map', component_property='figure'),
    [Input(component_id='map_drop', component_property='value')]
)
def plot_heatmap(variable):
    if variable == trg[0]:
        corr = df.drop(labels=trg[1],axis=1).corr()
    elif variable == trg[1]:
        corr = df[df[trg[0]]==1].drop(labels=trg[0],axis=1).corr()
    label_x,label_y = [],[]
    k = len(corr.index)-1
    for i in range(len(corr.index)):
        label_x.append(corr.index[i])
        label_y.append(corr.index[k])
        k-=1
    corr = np.asarray(corr)
    matrix,ax = [],[]
    for i in range(corr.shape[0]):
        fila = []
        k = corr.shape[1]-1
        for j in range(corr.shape[1]):
            fila.append(round(corr[i,k],2))
            k-=1
        matrix.append(fila)
    fig = ff.create_annotated_heatmap(z=matrix,x=label_x,y=label_y)
    return fig
#-------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='dist', component_property='figure'),
    [Input(component_id='dist_drop1', component_property='value'),
     Input(component_id='dist_drop2', component_property='value'),
     Input(component_id='dist_check', component_property='value')]
)
def plot_scatter_target_vs_continous(X,Y,opt):
    if opt == 'ambos':
        #Dataset completo
        df2 = df.copy()
    elif opt == 'crash':
        #Dataset filtrado por coque
        df2 = df[df[trg[0]]==1]
    elif opt == 'libre':
        #Dataset filtrado por libre de accidente
        df2 = df[df[trg[0]]==0]
    fig = go.Figure(data=go.Scatter(x=df2[X], y=df2[Y], mode='markers'))
    fig.update_layout(title='{} VS {}'.format(X,Y))
    return fig
#-------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='box', component_property='figure'),
    [Input(component_id='box_drop', component_property='value'),
     Input(component_id='box_check', component_property='value')]
)
def plot_cat_boxplot(x,point):
    y=trg[1]
    df2 = df[df[trg[0]]==1]
    fig = px.box(df2, x=x, y=y,points=point)
    return fig
#-------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='hist2', component_property='figure'),
    [Input(component_id='hist2_drop', component_property='value'),
     Input(component_id='hist2_check', component_property='value')]
)
def cambio_distribucion(X,opt):
    if opt == 'ambos':
        #Dataset completo
        x1 = df[X]
        x2 = df2[X]
    elif opt == 'crash':
        #Dataset filtrado por coque
        x1 = df[df[trg[0]]==1][X]
        x2 = df2[df2[trg[0]]==1][X]
    elif opt == 'libre':
        #Dataset filtrado por libre de accidente
        x1 = df[df[trg[0]]==0][X]
        x2 = df2[df2[trg[0]]==0][X]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x1,name='Cruda'))
    fig.add_trace(go.Histogram(x=x2,name='Limpia'))
    fig.update_layout(barmode='overlay')
    return fig
#------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='logm', component_property='figure'),
    [Input(component_id='logm_drop', component_property='value')]
)
def plot_model(lis):
    fin = 35
    fig = go.Figure()
    for i in lis:
        gr = pd.DataFrame(modelos['graph'][i])
        fig.add_trace(go.Scatter(x=np.arange(0,fin),y=gr.iloc[:,0],name=modelos['modelo'][i][4:]))
    return fig
#------------------------------------------------------------------------------------------------------------------
@app.callback(Output(component_id='regm', component_property='figure'),
    [Input(component_id='regm_drop', component_property='value')]
)
def plot_model(lis):
    fin = 30
    fig = go.Figure()
    for i in lis:
        gr = pd.DataFrame(modelos['graph'][i])
        fig.add_trace(go.Scatter(x=np.arange(0,fin),y=gr.iloc[:,0],name=modelos['modelo'][i][4:]))
    return fig
#-----------------------------------------------Main---------------------------------------------------
if __name__ == "__main__":
    app.run_server(debug=False)
