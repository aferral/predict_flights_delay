from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit,KFold
import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from preproceso import a_few_plots, RELEVANT_COUNTRIES, get_processed_data
from utils_eval import evaluate_predicted_df, plot_roc, get_metric_and_best_threshold_from_roc_curve

df,company_dict=get_processed_data()

# Train test split
# TODO split by fecha
# TODO mucho cuidado como separar datos
# TODO idea mostrar como sobreajusta del sampleo incorrecto de los datos.
# TODO importnte revisar posibles combinaciones de variables
"""
https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
http://francescopochetti.com/pythonic-cross-validation-time-series-pandas-scikit-learn/
"""
tscv = TimeSeriesSplit(n_splits=3)
# tscv = KFold(n_splits=3)

# evaluacion de modelo random para tener commparativa (no se usa train-test split ya que el modelo es literalmente ignorante de los datos)
df['y_score']=[random.random() for x in df['id'].values.tolist()]
df['y_pred']=df['y_score']>0.8
evaluate_predicted_df(df,'random_model')
df.drop(['y_score','y_pred'],axis=1)

plot=False
indexs_to_use=df['id'].values.tolist()
global_results=[]
for i, (train_index, test_index) in enumerate(tscv.split(indexs_to_use)):
    print(f"Fold {i}:")

    df_train=df.iloc[train_index]
    df_test=df.iloc[test_index]

    FEATURES=[
        'TIPOVUELO',
              'periodo_dia',

              'cambio_empresa',

              'calidad_dia',
              'calidad_year_dia',

              'x0',
              'x1',
              'x2',

              'rolling_emp_carga',
              'rolling_dest_carga',

        'dest_dist'
              ]
    FEATURES=FEATURES+['dia_{0}'.format(i) for i in range(7)]
    FEATURES=FEATURES+RELEVANT_COUNTRIES


    TARGET_COL='atraso_15'

    x_train,y_train=df_train[FEATURES].values,df_train[TARGET_COL].astype(int).values
    x_val,y_val=df_test[FEATURES].values,df_test[TARGET_COL].astype(int).values


    # modelo simple

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression()
    clf= KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
    clf = RandomForestClassifier(max_depth=None, random_state=0)
    clf.fit(x_train, y_train)

    df_train['y_score']=clf.predict_proba(x_train)[:,1]

    best_score, TH_TO_USE=get_metric_and_best_threshold_from_roc_curve(y_train, df_train['y_score'].values,calc='f1')

    df_test=df_test.assign(y_score=clf.predict_proba(x_val)[:,1])
    df_test=df_test.assign(y_pred=df_test['y_score']>=TH_TO_USE)
    print("Pred distribution: {0}".format(df_test.y_pred.value_counts().to_dict()))
    roc_area,f1_val=evaluate_predicted_df(df_test,'logistic regressor',plot=plot)

    row={'roc_area':roc_area,'f1_mean':f1_val}
    global_results.append(row)

df_full=pd.DataFrame(global_results)
print(df_full)
print(df_full['f1_mean'].mean())

"""
Ideas cambios de aerolina (se pueden saber con cuanto)

Puntos importantes:
    Las variables se saben con cuanto tiempo de antelacion?
"""