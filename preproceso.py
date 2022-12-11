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


"""
Cosas interesantes

* Esta mal definida el label de OPERA. Opera es realmente el label de la empresa del vuelo programado
Emp-I							OPERA
ejemplo:
df[df['Emp-O'] == 'AUT'][['Emp-I','Emp-O','OPERA']].drop_duplicates()

* CUIDADO CON df['Vlo-O'] y df['Vlo-I'] al inferir los tipos hay que tener cuidado. no se puede llegar y pasar a int
ya que faltan valores. Tampoco sirve float ya que hay strings. Para que los valores coincidan (400.0 vs 400) hay que intentar parse como int y si no es numero a strin

* pocas veces cambian de vuelo el numero
(df['Vlo-O'] != df['Vlo-I']).sum()

* Es bastante comun cambiar de empresa (notar que no cambia numeros de vuelo)
(df['Vlo-O'] != df['Vlo-I']).sum()

* MUy poco comun cambiar destino
(df['Des-I'] != df['Des-O']).sum()

* Los dias de vuelo mas comunes son viernes y jueves el menos comun sabado
df['DIANOM'].value_counts()

* fechas de un anno de 2017 a 2018.
    si se grafica por quincenas 

        Si bien el peek de dic-marzo , julio es claro. El otro peek parece ser octubre y no septiembre como sale en temporada alta
        df['temporada_alta']*3000
        ax = df.groupby(pd.Grouper(freq='SM', key='Fecha-I'))['Fecha-I'].count().plot(style='*--')
        (df.set_index('Fecha-I')['temporada_alta']*3000).plot(ax=ax, x='Fecha-I', y='temporada_alta')
        plt.show()
        print('f')

* Algo importante. Dado que estoy trabajando con un anno es bastante posible que sobre ajuste los datos. Se requiere mayores datos para validar
estacionalidad.

* La temporada alta concentra el 66% de los viajes
df['temporada_alta'].value_counts(normalize=True)

* el rate de atraso tambien crece en temporada alta
    es entonces importante la demanda del aeropuerto
    df.groupby(pd.Grouper(freq='SM',key='Fecha-I'))['atraso_15'].mean().plot(style='*--')

* todos del mismo origen
    Cuidado con buscar generalizar fuera de este aeropuerto

* vuelos internacionales y nacionales siguen mismo patron de demanda
key_to_use='TIPOVUELO'
res=df.groupby([pd.Grouper(key='Fecha-I',freq='SM'),key_to_use])['Vlo-I'].count().reset_index()
ax=None
for key,df_key in res.groupby(key_to_use):
    ax=df_key.plot(x='Fecha-I',y='Vlo-I',ax=ax,label=key)
plt.show()


* Vuelos muy concentrados entre GRUPO lata y sky (80 % de los vuelos)
df['Emp-O'].value_counts(normalize=True)
df['emp']=[company_dict[x] for x in df['Emp-O'].values.tolist()]
df['emp'].value_counts(normalize=True)

* Ligeramente ams vuelos nacionales 54% vs 45%
df['TIPOVUELO'].value_counts(normalize=True)

* El atraso en minutos es generalmente de 4 minutos. Hay que llegar al percentil 90 para tener un atraso de 15 minutos (eventos raros)
df['dif_min'].quantile([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
df['atraso_15'].value_counts(normalize=True)

* Los vuelos estan mas o menos equitativos en manana y dia. La noche tiene un poco menos
df['periodo_dia'].value_counts(normalize=True)

"""

def procesar_dataframe(df):


    df['Fecha-I'] = pd.to_datetime(df['Fecha-I'])
    df['Fecha-O'] = pd.to_datetime(df['Fecha-O'])

    df['temporada_alta'] = ([int(is_high_season(x)) for x in df['Fecha-I'].dt.to_pydatetime()])
    df['dif_min'] = (df['Fecha-O'] - df['Fecha-I']).dt.total_seconds() / 60
    df['atraso_15'] = (df['dif_min'] > 15).astype(int)

    periodo_dict = {0: 'mañana', 1: 'tarde', 2: 'noche'}  # [5:00 - 12:00(  [12:00 a 19:00( [19:00 a 05:00(

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
    df['Vlo-O'] = [str(int(float(x))) if x.replace('.', '').isdecimal() else x for x in
                   df['Vlo-O'].astype(str).values.tolist()]
    df['Vlo-I'] = [str(int(float(x))) if x.replace('.', '').isdecimal() else x for x in
                   df['Vlo-I'].astype(str).values.tolist()]

    # cambiar el dia de la semana a numero

    df['DIANOM'] = df['DIANOM'].map(dict_dia_semana)


    # para efectos de cross validation necesito ordenarlos por fecha
    df=df.sort_values('Fecha-O')
    df['id'] = np.arange(df.shape[0])
    return df,dict_tipo,periodo_dict,company_dict,airports_dict


def calc_features(df):
    df['cambio_empresa'] = (df['Emp-I'] != df['Emp-O']).astype(int)

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
                                    on='Fecha-I').aggregate({'atraso_15': agg, 'id': lambda x: x[-1]}).rename(columns={'atraso_15':'rolling_emp_carga'})
    aux['rolling_emp_carga']=np.clip(aux['rolling_emp_carga']/days,0,1)
    df=df.merge(aux,on='id',how='left')

    # rolling carga del destino
    aux=df.groupby('Des-I').rolling(window='{0}D'.format(days),
                                    on='Fecha-I').aggregate({'atraso_15': agg, 'id': lambda x: x[-1]}).rename(columns={'atraso_15':'rolling_dest_carga'})
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
        df['dia_{0}'.format(i)]=[x==i for x in df['DIANOM'].values.tolist()]


    df['x0'] = np.array([x%7 for x in df['Fecha-I'].dt.dayofyear.values.tolist()])#/7
    df['x1'] = np.array([x%30 for x in df['Fecha-I'].dt.dayofyear.values.tolist()])#/30
    df['x2'] = np.array([x%(30*4) for x in df['Fecha-I'].dt.dayofyear.values.tolist()])#/(30*4)


    return df


def a_few_plots(df):
    key_to_use = 'Emp-I'
    norm = False
    res = df.groupby([pd.Grouper(key='Fecha-I', freq='SM'), key_to_use])['Vlo-I'].count().reset_index()
    ax = None
    for key, df_key in res.groupby(key_to_use):
        fact = df_key['Vlo-I'].iloc[0] if norm else 1
        ax = (df_key['Vlo-I'] / fact).plot(x='Fecha-I', y='Vlo-I', ax=ax, label=key)
    plt.show()

    key_to_use = 'Des-I'
    norm = True
    res = df.groupby([pd.Grouper(key='Fecha-I', freq='SM'), key_to_use])['Vlo-I'].count().reset_index()
    ax = None
    for key, df_key in res.groupby(key_to_use):
        fact = df_key['Vlo-I'].iloc[0] if norm else 1
        ax = (df_key['Vlo-I'] / fact).clip(-3, 3).plot(x='Fecha-I', y='Vlo-I', ax=ax, label=key)
    plt.show()

    # efecto del destino en atraso TODO algo interesante?
    rr = df.groupby('Des-I').agg(n=('atraso_15', 'count'), m=('atraso_15', 'mean'))
    rr['enought']=rr['n'] > 100
    print(rr[rr.enought]['m'].quantile([0.1 * i for i in range(10)]))

    # efecto de empresa en atraso efecto igual importante. del percentil 0.7 es bastante alto
    rr = df.groupby('Emp-I').agg(n=('atraso_15', 'count'), m=('atraso_15', 'mean'))
    rr['enought']=rr['n'] > 100
    print(rr[rr.enought]['m'].quantile([0.1 * i for i in range(10)]))


    # efecto de tiempo del dia en atraso
    # no mucho, pero en la manana tiene de ser menor atraso
    df[['periodo_dia', 'atraso_15']].groupby('periodo_dia').agg(n=('atraso_15', 'count'), m=('atraso_15', 'mean'))

    # efecto del tipo de viaje en atraso. Efectoi mportante
    rr = df.groupby('TIPOVUELO').agg(n=('atraso_15', 'count'), m=('atraso_15', 'mean'))
    print(rr)

    # efecto del mes en atraso. HAy importancia a periodos de alta actividad veer como sube atraso
    rr = df.groupby(pd.Grouper(freq='M', key='Fecha-I')).agg(n=('atraso_15', 'count'), m=('atraso_15', 'mean'))
    print(rr)

    # efecto del dia de la semana. a mayor cantidad de gente sube algo el % de atraso.
    rr = df.groupby(pd.Grouper(freq='M', key='Fecha-I')).agg(n=('atraso_15', 'count'), m=('atraso_15', 'mean'))
    print(rr)

    # efecto de temporada. Ligereamente mas alta segun temporada alta o no
    rr = df.groupby('temporada_alta').agg(n=('atraso_15', 'count'), m=('atraso_15', 'mean'))
    print(rr)