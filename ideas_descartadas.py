"""
Ideas descartadas. Quizas pueden ser utiles al futuro

"""


#======== rolling carga del aeropuerto
aux = df.rolling(window='{0}D'.format(days),
                 on='Fecha-I').aggregate({'atraso_15': agg, 'id': lambda x: x[-1]}).rename(
    columns={'atraso_15': 'rolling_glob_carga'})
aux['rolling_glob_carga'] = np.clip(aux['rolling_glob_carga'] / 4000, 0, 1)
df = df.merge(aux, on='id', how='left')
# df['rolling_glob_carga']=1 # TODO MUY NEGATIVO EL div 6000 negativo con 4000 efecto muy bajo

# ===== rolling carga sobre el tipo de viaje
aux = df.groupby('TIPOVUELO').rolling(window='{0}D'.format(days),
                                      on='Fecha-I').aggregate({'atraso_15': agg, 'id': lambda x: x[-1]}).rename(
    columns={'atraso_15': 'rolling_international_carga'})
aux['rolling_international_carga'] = np.clip(aux['rolling_international_carga'] / days, 0, 1)
df = df.merge(aux, on='id', how='left')
# df['rolling_international_carga'] = 1 # TODO negativo


# ====== por empresa estima confiabilidad =========
days = 30
aux = df.groupby('Emp-I').rolling(window='{0}D'.format(days),
                                  on='Fecha-I').aggregate({'atraso_15': 'mean', 'id': lambda x: x[-1]}).rename(
    columns={'atraso_15': 'emp_confianza'})
aux['emp_confianza'] = np.clip(aux['emp_confianza'], 0, 1)
df = df.merge(aux, on='id', how='left')


# empresa confiabilidad v2
days = 30
aux = df.groupby('Emp-I').rolling(window='{0}D'.format(days),
                                  on='Fecha-I').aggregate({'atraso_15': 'mean', 'id': lambda x: x[-1]}).rename(
    columns={'atraso_15': 'emp_confianza'})
aux['emp_confianza'] = np.clip(aux['emp_confianza'], 0, 1)
df = df.merge(aux, on='id', how='left')

# empresa confiabilidad v3 (cuidado usa historia full)
aux = df.groupby('Emp-I').agg(emp_confianza=('atraso_15', 'mean'), id=('id', 'first'))
df = df.merge(aux, on='id', how='left').fillna(0.16)


# cambios de avion y destino muy raros
df['cambio_destino'] = (df['Des-I'] != df['Des-O']).astype(int)
df['cambio_avion'] = (df['Vlo-I'] != df['Vlo-O']).astype(int)

# cantidad de vuelos pro dia
df['vuelos_ese_dia'] = df.groupby(pd.Grouper(freq='D', key='Fecha-I'))['dest_country'].transform('count')
df['vuelos_4_dia'] = df.groupby(pd.Grouper(freq='4D', key='Fecha-I'))['dest_country'].transform('count')

# atrasos en los ultimos 20 vuelos
df['rolling_atraso'] = df.groupby('Des-I').rolling(20).aggregate({'atraso_15': 'sum'})['atraso_15'].fillna(
    0).values.tolist()


# normalizar la carga por empresa
map_daily_load_company = df.groupby(['Emp-I', pd.Grouper(key='Fecha-I', freq='D')])['Vlo-I'].count().groupby(
    'Emp-I').median().to_dict()
aux['rolling_emp_norm'] = [(x / map_daily_load_company[emp]) / 50 for emp, x in
                           aux.reset_index()[['Emp-I', 'rolling_emp_carga']].values.tolist()]
aux['rolling_emp_norm'] = aux['rolling_emp_norm'].clip(0, 1)