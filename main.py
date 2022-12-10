import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from preproceso import procesar_dataframe, calc_features, a_few_plots
from utils_eval import evaluate_predicted_df, plot_roc, get_metric_and_best_threshold_from_roc_curve

rt = 'dataset_SCL.csv'

df = pd.read_csv(rt)
df,dict_tipo,periodo_dict,company_dict,airports_dict=procesar_dataframe(df)

df=calc_features(df)

# a_few_plots(df) # TODO plot in jupyer

# Train test split
# TODO split by fecha
# TODO mucho cuidado como separar datos
# TODO idea mostrar como sobreajusta del sampleo incorrecto de los datos.
"""
https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
http://francescopochetti.com/pythonic-cross-validation-time-series-pandas-scikit-learn/
"""
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)

# evaluacion de modelo random para tener commparativa (no se usa train-test split ya que el modelo es literalmente ignorante de los datos)
df['y_score']=[random.random() for x in df['id'].values.tolist()]
df['y_pred']=df['y_score']>0.8
evaluate_predicted_df(df,'random_model')
df.drop(['y_score','y_pred'],axis=1)


indexs_to_use=df['id'].values.tolist()
for i, (train_index, test_index) in enumerate(tscv.split(indexs_to_use)):
    print(f"Fold {i}:")

    df_train=df.iloc[train_index]
    df_test=df.iloc[test_index]

    # TODO test de sanidad esto es practicamente pasar el target en entrenamiento
    FEATURES=['dif_min'] # 'TIPOVUELO','temporada_alta','periodo_dia','cambio_destino','cambio_empresa','cambio_avion',
    FEATURES=['TIPOVUELO','temporada_alta','periodo_dia','cambio_destino','cambio_empresa','cambio_avion','calidad_dia']

    TARGET_COL='atraso_15'

    x_train,y_train=df_train[FEATURES].values,df_train[TARGET_COL].astype(int).values
    x_val,y_val=df_test[FEATURES].values,df_test[TARGET_COL].astype(int).values


    # modelo simple
    from sklearn.linear_model import LogisticRegression

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression(fit_intercept=False)
    clf= KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
    clf.fit(x_train, y_train)

    df_train['y_score']=clf.predict_proba(x_train)[:,1]

    best_score, TH_TO_USE=get_metric_and_best_threshold_from_roc_curve(y_train, df_train['y_score'].values, calc='f1')

    df_train['y_pred']=df_train['y_score']>=TH_TO_USE
    evaluate_predicted_df(df_train, 'logistic regressor_TRAIN')

    df_test['y_score']=clf.predict_proba(x_val)[:,1]
    df_test['y_pred']=df_test['y_score']>=TH_TO_USE
    evaluate_predicted_df(df_test,'logistic regressor')



"""
Ideas cambios de aerolina (se pueden saber con cuanto)

Puntos importantes:
    Las variables se saben con cuanto tiempo de antelacion?
"""