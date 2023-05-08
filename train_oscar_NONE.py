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

fechas_prediccion = [("2020-12-28","2021-1-11"),("2021-1-11","2021-1-25"),("2021-1-25","2021-2-8"),("2021-2-8","2021-2-22"),("2021-2-22","2021-3-8"),("2021-3-8","2021-3-22"),("2021-3-22","2021-4-5"),("2021-4-5","2021-4-19"),
("2021-4-19","2021-5-3"), ("2021-5-3","2021-5-17"),("2021-5-17","2021-5-31"),("2021-5-31","2021-6-14"),("2021-6-14","2021-6-28"), ("2021-6-28","2021-7-12"),("2021-7-12","2021-7-26"),("2021-7-26","2021-8-9"),
("2021-08-09","2021-08-23"), ("2021-08-23","2021-09-06"), ("2021-09-06","2021-09-20"), ("2021-09-20","2021-10-04"), ("2021-10-04","2021-10-18"), ("2021-10-18","2021-11-01"), ("2021-11-01","2021-11-15"),
("2021-11-15","2021-11-29"), ("2021-11-29","2021-12-13"), ("2021-12-13","2021-12-27"), ("2021-12-27","2021-12-31")]


fechas_entrenamiento = [("2020-09-01", "2020-12-27"), ("2020-09-01", "2021-1-10"), ("2020-09-01", "2021-1-24"), ("2020-09-01", "2021-2-7"), ("2020-09-01", "2021-2-21"), ("2020-09-01", "2021-3-7"), ("2020-09-01", "2021-3-21"),
("2020-09-01", "2021-4-4"), ("2020-09-01", "2021-4-18"), ("2020-09-01", "2021-5-2"), ("2020-09-01", "2021-5-16"), ("2020-09-01", "2021-5-30"), ("2020-09-01", "2021-6-13"), ("2020-09-01", "2021-6-27"), ("2020-09-01", "2021-7-11"),
("2020-09-01", "2021-7-25"), ("2020-09-01", "2021-08-08"), ("2020-09-01", "2021-08-22"), ("2020-09-01", "2021-09-05"), ("2020-09-01", "2021-09-19"), ("2020-09-01", "2021-10-03"), ("2020-09-01", "2021-10-17"), 
("2020-09-01", "2021-10-31"), ("2020-09-01", "2021-11-14"), ("2020-09-01", "2021-11-28"), ("2020-09-01", "2021-12-12"), ("2020-09-01", "2021-12-26")]  

# Ahora vamos a realizar 100 entrenamientos, uno por cada mes
df = pd.read_csv('data/OxCGRT_latest.csv')
training_geos = ['Argentina', 'Austria', 'Belgium', 'Bulgaria', 'Canada', 'Croatia',
       'Cyprus', 'Czech Republic', 'Denmark', 'Ecuador', 'Estonia', 'Finland',
       'France', 'Germany', 'Hungary', 'Ireland', 'Italy', 'Latvia',
       'Lithuania', 'Luxembourg', 'Netherlands',
       'Norway', 'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain',
       'Sweden', 'Switzerland', 'United States']

train_preset = ModelPreset.VAC_NONE
#LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv'
GEO_FILE = "data/countries_regions.csv"
DATA_FILE = 'data/OxCGRT_latest.csv'
# Creamos un dataframe que tenga el mae, el mes y el trial
df_mae = pd.DataFrame(columns=['mae', 'trial', 'mes'])
for i in range(12, 21):
    print("Trial: ", i)
    for j in range(len(fechas_entrenamiento)):
        print("Entrenando con el mes: ", fechas_entrenamiento[j])
        print("el j es" ,j)
        start_date_str = fechas_entrenamiento[j][0]
        end_date_str = fechas_entrenamiento[j][1]
    #   # path_to_model = f"model_23_03_23/valencia_model_weights_"+str(i)+"_"+str(j)+".h5" # La i indica el trial y la j el mes
        path_to_model = f"oscar_NONE_waning_casos/models/valencia_model_weights_"+str(i)+"_"+str(j)+".h5" # La i indica el trial y la j el mes

        #predictor = ValenciaPredictor(model_preset=train_preset)
        predictor = ValenciaPredictor(model_preset = train_preset)
        predictor_model = predictor.train(geos=training_geos, start_date_str=start_date_str, end_date_str=end_date_str) #, min_vaccination_percentage=min_vaccination)

        print('Saving model to', path_to_model)
        predictor_model.save_weights(path_to_model)
        print("Testeando con el mes: ", fechas_prediccion[j])
        pred_start_date_str = fechas_prediccion[j][0]
        pred_end_date_str = fechas_prediccion[j][1]
        pred_start_date = pd.to_datetime(pred_start_date_str)
        pred_end_date = pd.to_datetime(pred_end_date_str)
        preds_df = predictor.predict_df(pred_start_date, pred_end_date, 'data/OxCGRT_latest.csv')
        # Obtenemos el mae de la predicción
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        #ground_truth_df = df[df['CountryName'].isin(training_geos)]
        ground_truth_df = df
        ground_truth_df = ground_truth_df[ground_truth_df['Date'] >= pred_start_date]
        ground_truth_df = ground_truth_df[ground_truth_df['Date'] <= pred_end_date]
        ground_truth_df = ground_truth_df[['CountryName', 'Date', 'ConfirmedCases']]
        # Creamos una nueva columna, que tenga el valor de la columna 'ConfirmedCases' respecto al día anterior
        ground_truth_df['ConfirmedCases_Diff'] = ground_truth_df.groupby('CountryName')['ConfirmedCases'].diff()
        #print("Number of rows in ground truth: {}".format(len(ground_truth_df)))
        # Place 0 in NaNs
        ground_truth_df['ConfirmedCases_Diff'] = ground_truth_df['ConfirmedCases_Diff'].fillna(0)
        # Cogemos las predicciones de los países que nos interesan
        #pred = preds_df[preds_df['CountryName'].isin(training_geos)]
        pred = preds_df
        y_true = ground_truth_df['ConfirmedCases_Diff']
        y_pred = pred['PredictedDailyNewCases']
        mae = mean_absolute_error(y_true, y_pred).round(2)
        print("MAE: ", mae)
        
        # Guardamos las predicciones
   #     #preds_df.to_csv("predictions/23_03_23/robojudge_test_"+str(i)+"_"+str(j)+".csv", index=False)
        preds_df.to_csv("oscar_NONE_waning_casos/predictions/robojudge_test_"+str(i)+"_"+str(j)+".csv", index=False)
        # Guardamos el mae, el mes y el trial en un dataframe
        df_mae = df_mae.append({'mae': mae, 'trial': i, 'mes': j}, ignore_index=True)
        del predictor # Borramos el predictor para que no se acumulen en memoria
        print("Guardando el dataframe con los mae")
   #     #df_mae.to_csv("maes_23.csv", index=False)
        df_mae.to_csv("oscar_NONE_waning_casos/maes_NONE_waning_casos_vacunas.csv", index=False)
