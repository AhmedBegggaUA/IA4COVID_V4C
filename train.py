# Librerías necesarias
import numpy as np
import pandas as pd
import os
import sys
import argparse
import valencia_predictor
from importlib import reload
reload(valencia_predictor)
from valencia_predictor import ValenciaPredictor
from valencia_predictor import ModelPreset
import warnings
warnings.filterwarnings("ignore") # Let's get rid of those warnings
import pandas as pd
from sklearn.metrics import mean_absolute_error
# Vamos a hacer una lista que tenga fechas desde el 1 de enero de 2021 hasta el 31 de julio de 2021
# donde cada elemento de la lista es una tupla con el inicio y el fin del mes
fechas_prediccion = []
for i in range(4, 5):
    # Hay que tener cuidado con los meses que tienen 31 días y los que tienen 30
    if i in [1, 3, 5, 7]:
        fecha_inicio = f"2021-0{i}-01"
        fecha_fin = f"2021-0{i}-31"
    elif i in [4, 6]:
        fecha_inicio = f"2021-0{i}-01"
        fecha_fin = f"2021-0{i}-30"
    else:
        fecha_inicio = f"2021-0{i}-01"
        fecha_fin = f"2021-0{i}-28"
    fechas_prediccion.append((fecha_inicio, fecha_fin))
    fechas_entrenamiento = [("2020-09-01", "2021-03-31")]
#fechas_entrenamiento = [("2020-09-01", "2020-12-31"), ("2020-09-01", "2021-01-31"), ("2020-09-01", "2021-02-28"), ("2020-09-01", "2021-03-31"), ("2020-09-01", "2021-04-30"), ("2020-09-01", "2021-05-31"), ("2020-09-01", "2021-06-30")]
# Ahora vamos a realizar 100 entrenamientos, uno por cada mes
df = pd.read_csv('data/OxCGRT_latest.csv')
training_geos = ["United States", "Portugal", "United Kingdom", "Italy", "Spain",
                 "Germany", "Israel", "France", "Greece", "Hungary"]
train_preset = ModelPreset.VAC_H7_SUS
LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv'
GEO_FILE = "data/countries_regions.csv"
DATA_FILE = 'data/OxCGRT_latest.csv'
# Creamos un dataframe que tenga el mae, el mes y el trial
df_mae = pd.DataFrame(columns=['mae', 'trial', 'mes'])
for i in range(20):
    print("Trial: ", i)
    for j in range(len(fechas_entrenamiento)):
        print("Entrenando con el mes: ", fechas_entrenamiento[j])
        start_date_str = fechas_entrenamiento[j][0]
        end_date_str = fechas_entrenamiento[j][1]
        path_to_model = f"model_08_03_23/valencia_model_weights_"+str(i)+"_"+str(j)+".h5" # La i indica el trial y la j el mes
        predictor = ValenciaPredictor(model_preset=train_preset)
        predictor_model = predictor.train(geos=training_geos, start_date_str=start_date_str, end_date_str=end_date_str) #, min_vaccination_percentage=min_vaccination)
        
        print('Saving model to', path_to_model)
        predictor_model.save_weights(path_to_model)
        print("Testeando con el mes: ", fechas_prediccion[j])
        pred_start_date_str = fechas_prediccion[j][0]
        pred_end_date_str = fechas_prediccion[j][1]
        pred_start_date = pd.to_datetime(pred_start_date_str)
        pred_end_date = pd.to_datetime(pred_end_date_str)
        preds_df = predictor.predict_df(pred_start_date, pred_end_date, LATEST_DATA_URL)
        # Obtenemos el mae de la predicción
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        ground_truth_df = df[df['CountryName'].isin(training_geos)]
        ground_truth_df = ground_truth_df[ground_truth_df['Date'] >= pred_start_date]
        ground_truth_df = ground_truth_df[ground_truth_df['Date'] <= pred_end_date]
        ground_truth_df = ground_truth_df[['CountryName', 'Date', 'ConfirmedCases']]
        # Creamos una nueva columna, que tenga el valor de la columna 'ConfirmedCases' respecto al día anterior
        ground_truth_df['ConfirmedCases_Diff'] = ground_truth_df.groupby('CountryName')['ConfirmedCases'].diff()
        #print("Number of rows in ground truth: {}".format(len(ground_truth_df)))
        # Place 0 in NaNs
        ground_truth_df['ConfirmedCases_Diff'] = ground_truth_df['ConfirmedCases_Diff'].fillna(0)
        # Cogemos las predicciones de los países que nos interesan
        pred = preds_df[preds_df['CountryName'].isin(training_geos)]
        y_true = ground_truth_df['ConfirmedCases_Diff']
        y_pred = pred['PredictedDailyNewCases']
        mae = mean_absolute_error(y_true, y_pred).round(2)
        print("MAE: ", mae)
        
        # Guardamos las predicciones
        preds_df.to_csv("predictions/08_03_23/robojudge_test_H7_SUS_"+str(i)+"_"+str(j)+".csv", index=False)
        # Guardamos el mae, el mes y el trial en un dataframe
        df_mae = df_mae.append({'mae': mae, 'trial': i, 'mes': j}, ignore_index=True)
        del predictor # Borramos el predictor para que no se acumulen en memoria
        print("Guardando el dataframe con los mae")
        df_mae.to_csv("maes_08.csv", index=False)