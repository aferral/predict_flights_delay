
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
