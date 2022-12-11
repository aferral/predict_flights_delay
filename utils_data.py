import pandas as pd
import os

PATH_DATA_AIRPORTS='airport_data.csv'
def get_airport_data():

    if not os.path.exists(PATH_DATA_AIRPORTS):
        df_airports = pd.read_csv('https://raw.githubusercontent.com/mborsetti/airportsdata/main/airportsdata/airports.csv')

        row={'icao':'SEQU','iata':None,'name':'Aeropuerto Internacional Mariscal Sucre',
             'city':'Quito','subd':'Pichincha','country':'EC',
             'elevation':9228,'lat':-0.141111,'lon':-78.488056,'tz':'America/Guayaquil','lid':None}
        df_airports=df_airports.append(row,ignore_index=True)

        row={'icao':'SCQP','iata':None,'name':'Temuco Airport',
             'city':'Temuco','subd':'Araucania','country':'CL',
             'elevation':321,'lat':-38.925,'lon':-72.651389,'tz':'America/Santiago','lid':None}
        df_airports=df_airports.append(row,ignore_index=True)
        df_airports.to_csv(PATH_DATA_AIRPORTS,index=False)
    df=pd.read_csv(PATH_DATA_AIRPORTS)
    return df

if __name__ == '__main__':
    df=get_airport_data()
    print(df)