import haversine as hs
from utils_data import get_airport_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

dict_dia_semana = {
    'Lunes': 0,
    'Martes': 1,
    'Miercoles': 2,
    'Jueves': 3,
    'Viernes': 4,
    'Sabado': 5,
    'Domingo': 6,
}

periodo_dict = {0: 'mañana', 1: 'tarde', 2: 'noche'}  # [5:00 - 12:00(  [12:00 a 19:00( [19:00 a 05:00(

RELEVANT_COUNTRIES=['AR', 'AU', 'BO', 'BR', 'CA', 'CL', 'CO', 'DO', 'EC', 'ES', 'FK', 'FR', 'GB', 'IT', 'MX', 'NZ', 'PA', 'PE', 'PY', 'US', 'UY']

def get_labels_as_dict(to_use,key,val):
    aux=to_use[[key,val]].drop_duplicates()
    assert(not aux.duplicated(subset=[key]).any()),'Duplicated index in {0} {1}'.format(key,val)
    dict_with_labels=aux.set_index(key)[val].to_dict()
    return dict_with_labels

def is_high_season(x):
    day,month=x.day,x.month
    if month in [12,1,2,3]: # 15 dic a 3 marzo
        if month in [1,2]:
            return True
        elif month == 12:
            return day >= 15
        elif month == 3:
            return day <= 3
    if month in [12,1,2,3]: # 15 dic a 3 marzo
        return True
    elif month in [7]: # 15 jul a 31 jul
        return 15 <= day <= 31
    elif month in [9]: # 11 sept a 30 sep
        return 11 <= day <= 30
    else:
        return False

def get_periodo(time_obj): # [5:00 - 12:00(  [12:00 a 19:00( [19:00 a 05:00(
    h,m=time_obj.hour,time_obj.minute
    if h >= 5 and h < 12:
        return 0
    elif h >= 12 and h < 19:
        return 1
    else:
        return 2





def get_processed_data():
    """
    Obtiene el dataframe con los datos a usar en el modelo
    """
    rt = 'dataset_SCL.csv'
    df = pd.read_csv(rt)

    # agrega 'dest_country','dest_dist'
    df_airports = get_airport_data()
    x0, y0 = df_airports[df_airports['icao'] == 'SCEL'][['lat', 'lon']].iloc[0].values.tolist()
    distancias_a_arturo_benites = [hs.haversine((lat, lon), (x0, y0)) for lat, lon in
                                   df_airports[['lat', 'lon']].values.tolist()]
    df_airports['dist'] = distancias_a_arturo_benites
    df_airports = df_airports[['icao', 'country', 'dist']].rename(
        columns={'country': 'dest_country', 'dist': 'dest_dist'})
    df = df.merge(df_airports, left_on='Des-I', right_on='icao', how='left').drop('icao', axis=1)


    df['Fecha-I'] = pd.to_datetime(df['Fecha-I'])
    df['Fecha-O'] = pd.to_datetime(df['Fecha-O'])


    df['temporada_alta'] = ([int(is_high_season(x)) for x in df['Fecha-I'].dt.to_pydatetime()])
    df['dif_min'] = (df['Fecha-O'] - df['Fecha-I']).dt.total_seconds() / 60
    df['atraso_15'] = (df['dif_min'] > 15).astype(int)
    df['periodo_dia'] = [get_periodo(x) for x in df['Fecha-I'].dt.time.values.tolist()]

    # como el origen es siempre el mismo aeropuerto podemos quitar
    assert ((df['Ori-I'] != df['Ori-O']).sum() == 0)
    df = df.drop('Ori-O', axis=1)

    # solo hay datos de un anno practicamente
    df = df.drop('AÑO', axis=1)

    # EL dia y el mes se trataran de otra forma
    df = df.drop('DIA', axis=1)
    df = df.drop('MES', axis=1)

    # cambiar el tipo de vuelo a I o N a 0
    dict_tipo = {'N': 0, "I": 1}
    df['TIPOVUELO'] = df['TIPOVUELO'].map(dict_tipo)

    # obtener siglas y luego borrar de dataframe
    key, val = 'Ori-I', 'SIGLAORI'
    origin_dict = get_labels_as_dict(df, key, val)

    key, val = 'Emp-I', 'OPERA'
    company_dict = get_labels_as_dict(df, key, val)

    key, val = 'Des-O', 'SIGLADES'
    dest_dict = get_labels_as_dict(df, key, val)

    airports_dict = {**dest_dict, **origin_dict}

    df = df.drop(['SIGLAORI', 'OPERA', 'SIGLADES'], axis=1)

    # ya que es un puro aeropuerto de destino podemos quitar tambien esta columna
    df = df.drop('Ori-I', axis=1)

    # para comparar esta variable hay que tener cuidado con el parse
    #CUIDADO CON df['Vlo-O'] y df['Vlo-I'] al inferir los tipos hay que tener cuidado. no se puede llegar y pasar a int
    #Tampoco sirve float ya que hay strings. Para que los valores coincidan (400.0 vs 400) hay que intentar parse como int y si no es numero a string
    df['Vlo-O'] = [str(int(float(x))) if x.replace('.', '').isdecimal() else x for x in
                   df['Vlo-O'].astype(str).values.tolist()]
    df['Vlo-I'] = [str(int(float(x))) if x.replace('.', '').isdecimal() else x for x in
                   df['Vlo-I'].astype(str).values.tolist()]

    # cambiar el dia de la semana a numero
    df['DIANOM'] = df['DIANOM'].map(dict_dia_semana)

    # para efectos de cross validation necesito ordenarlos por fecha
    df=df.sort_values('Fecha-O')
    df['id'] = np.arange(df.shape[0])


    df['cambio_empresa'] = (df['Emp-I'] != df['Emp-O']).astype(int)

    # TODO definir temporada alta v2 ?
    # TODO agregar temperatura
    # todo agregar semana del ano percentil ???? (estare sobre ajustando?)

    # TODO codificar aeropuerto destino
    # TODO codificar empresa operando

    # TODO codificar demanda ultimos 10 dias ?
    # TODO crear variables que midan saturacion de red. Ejmplo vuelos interacionales X dias atras

    # ONE HOT de paises destino
    enc = OneHotEncoder()
    res=enc.fit_transform(df['dest_country'].values.reshape(-1, 1)).todense()
    df_encoded_countries=pd.DataFrame(res,columns=enc.categories_[0])
    df_encoded_countries['id']=df['id'].values.tolist()
    df=df.merge(df_encoded_countries, on='id', how='left')

    # por cada empresa colocar cantidad vuelos ultimos X dias
    df=df.set_index('Fecha-I').sort_index().reset_index()
    days=30
    agg='count'

    # rolling carpa por empresa
    aux=df.groupby('Emp-I').rolling(window='{0}D'.format(days),
                                    on='Fecha-I').aggregate({'id': agg, 'id': lambda x: x[-1]}).rename(columns={'id':'rolling_emp_carga'})
    aux['rolling_emp_carga']=np.clip(aux['rolling_emp_carga']/days,0,1)
    df=df.merge(aux,on='id',how='left')

    # rolling carga del destino
    aux=df.groupby('Des-I').rolling(window='{0}D'.format(days),
                                    on='Fecha-I').aggregate({'id': agg, 'id': lambda x: x[-1]}).rename(columns={'id':'rolling_dest_carga'})
    aux['rolling_dest_carga']=np.clip(aux['rolling_dest_carga']/days,0,1)
    df=df.merge(aux,on='id',how='left')


    # dias rankeados segun demanda
    dict_calidad={
        'Viernes': 6,
        'Jueves': 5,
        'Lunes': 4,
        'Domingo': 3,
        'Miercoles': 2,
        'Martes': 1,
        'Sabado':0
    }
    inv_code_to_day={v:k for k,v in dict_dia_semana.items()}
    df['calidad_dia']=np.array([dict_calidad[inv_code_to_day[x]] for x in df['DIANOM'].values.tolist()]) / 6

    df['calidad_year_dia']=[x for x in df['Fecha-I'].dt.dayofyear.values.tolist()]

    # one hot encoding de los dias
    for i in range(7):
        df['dia_{0}'.format(i)]=[int(x==i) for x in df['DIANOM'].values.tolist()]


    # variables de dias dada distinta frecuencia.
    df['x0'] = np.array([x%7 for x in df['Fecha-I'].dt.dayofyear.values.tolist()])#/7
    df['x1'] = np.array([x%30 for x in df['Fecha-I'].dt.dayofyear.values.tolist()])#/30
    df['x2'] = np.array([x%(30*4) for x in df['Fecha-I'].dt.dayofyear.values.tolist()])#/(30*4)


    return df,company_dict