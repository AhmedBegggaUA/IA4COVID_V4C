from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D #[NEW]
from keras.callbacks import EarlyStopping #[NEW]
from keras.layers import Bidirectional
from keras.models import Model
#from keras.optimizers import Adam
import tensorflow as tf
Adam = tf.keras.optimizers.Adam
from keras.constraints import Constraint
from keras.constraints import NonNeg

import os
import csv 
import ast
import pandas as pd
import numpy as np

import datetime


# Suppress noisy Tensorflow debug logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

INCLUDE_CV_PREDICTION = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#Ruta donde se encuentran los datos
DATA_PATH = os.path.join(ROOT_DIR, 'data')
#Donde se encuentran los csv con los datos necesarios
DATA_FILE_PATH = os.path.join(DATA_PATH, 'OxCGRT_latest.csv')
DATA_FILE_CV_PATH = os.path.join(DATA_PATH, 'OxfordComunitatValenciana.csv')
ADDITIONAL_CONTEXT_FILE = os.path.join(DATA_PATH, "Additional_Context_Data_Global.csv")
ADDITIONAL_US_STATES_CONTEXT = os.path.join(DATA_PATH, "US_states_populations.csv")
ADDITIONAL_UK_CONTEXT = os.path.join(DATA_PATH, "uk_populations.csv")
ADDITIONAL_BRAZIL_CONTEXT = os.path.join(DATA_PATH, "brazil_populations.csv")
#[NEW] Ruta para las vacunas
ADDITIONAL_VAC_CV_URL = 'https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_vacunas.csv'
ADDITIONAL_VAC_FILE = os.path.join(DATA_PATH, "vaccinations.csv")
#Ruta donde se encuentran los modelos
MODEL_PATH = os.path.join(ROOT_DIR, 'models')
MODEL_WEIGHTS_CLUSTER_FILE = os.path.join(MODEL_PATH, "weightscluster{}_280traineddays.h5")
MODEL_WEIGHTS_SCO_V0_FILE = os.path.join(MODEL_PATH, "sco_v0_trained_model_weights.h5")
MODEL_WEIGHTS_SCO_V1_FILE = os.path.join(MODEL_PATH, "sco_v1_trained_model_weights.h5")
MODEL_WEIGHTS_SCO_V2_FILE = os.path.join(MODEL_PATH, "sco_v2_trained_model_weights.h5")

#[NEW] Nuevos Ficherps para la vacunación
MODEL_WEIGHTS_VAC_H7_FILE = os.path.join(MODEL_PATH, "valencia_vac_h7_model_weights.h5")
MODEL_WEIGHTS_VAC_H7_SUS_FILE = os.path.join(MODEL_PATH, "valencia_vac_h7_sus_model_weights.h5")
MODEL_WEIGHTS_VAC_SUS_FILE = os.path.join(MODEL_PATH, "valencia_vac_sus_model_weights.h5")
MODEL_WEIGHTS_VAC_NONE_FILE = os.path.join(MODEL_PATH, "valencia_vac_none_model_weights.h5")

ADDITIONAL_CV_POPULATION = 5003769

# Identificadores de las columnas
ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
# Una nueva cplumna con el valor de la media de los ultimos 7 dias
CASES_COL = ['NewCases']
#[NEW] En la última versión se añade la columna con la población inmunizada
IMMUNIZED_COL = ['ProportionImmunized']
#Más columnas que se añaden al dataset
CONTEXT_COLUMNS = ['CountryName',
                   'RegionName',
                   'GeoID',
                   'Date',
                   'ConfirmedCases',
                   'ConfirmedDeaths',
                   'Population',
                   'ProportionImmunized'] #[NEW]
#NPIs (non-pharmaceutical interventions) columns
NPI_COLUMNS_OLD = ["C1M_School closing",
            "C2M_Workplace closing",
            "C3M_Cancel public events",
            "C4M_Restrictions on gatherings",
            "C5M_Close public transport",
            "C6M_Stay at home requirements",
            "C7M_Restrictions on internal movement",
            "C8EV_International travel controls"]
# [NEW] Introduccción de nuevas NPIs
#Esta primera lista, tiene las NPis antiguas
NPI_COLUMNS_8C = ["C1M_School closing",
            "C2M_Workplace closing",
            "C3M_Cancel public events",
            "C4M_Restrictions on gatherings",
            "C5M_Close public transport",
            "C6M_Stay at home requirements",
            "C7M_Restrictions on internal movement",
            "C8EV_International travel controls"]
# [NEW] Esta lista incluye las restricciones Hs (Intervenciones políticas)
NPI_COLUMNS_8C_4H = ["C1M_School closing",
            "C2M_Workplace closing",
            "C3M_Cancel public events",
            "C4M_Restrictions on gatherings",
            "C5M_Close public transport",
            "C6M_Stay at home requirements",
            "C7M_Restrictions on internal movement",
            "C8EV_International travel controls",
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6M_Facial Coverings']
# [NEW] La misma que la de antes, pero se incluye la política de vacunación
NPI_COLUMNS_8C_5H = ["C1M_School closing",
            "C2M_Workplace closing",
            "C3M_Cancel public events",
            "C4M_Restrictions on gatherings",
            "C5M_Close public transport",
            "C6M_Stay at home requirements",
            "C7M_Restrictions on internal movement",
            "C8EV_International travel controls",
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6M_Facial Coverings',
            'H7_Vaccination policy'] 
NB_LOOKBACK_DAYS = 21
WINDOW_SIZE = 7
LSTM_SIZE = 32
US_PREFIX = "United States / "
# [NEW] Nuevas constantes , TODO revisar
NB_TEST_DAYS = 14
NUM_TRIALS = 1# 5 # 10
MAX_NB_COUNTRIES = 40
PARTIAL_VAC_IMMUNIZATION_PERIOD = 12
PARTIAL_VAC_IMMUNIZATION_PROB = 0.7
FULLY_VAC_IMMUNIZATION_PERIOD = 14
FULLY_VAC_IMMUNIZATION_PROB = 0.9
REINFECTION_RATE = 0


Cluster_1 = [('Central African Republic', ''),('Chile', ''),('China', ''),('Lithuania', ''),('Niger', ''),('Panama', ''),
             ('Sweden', ''),('Switzerland', ''),('United States', 'Arizona'),('United States', 'Hawaii'),
             ('United States', 'Maine'),('United States', 'Rhode Island')]
Cluster_2 = [('Bahrain', ''),('Bangladesh', ''),('El Salvador', ''),('Estonia', ''),('Japan', ''),('Kosovo', ''),
             ('Luxembourg', ''),('Moldova', ''),('Peru', ''),('Vietnam', '')]
Cluster_3 = [('Andorra', ''),('Aruba', ''),('Australia', ''),('Belarus', ''),('Belgium', ''),('Bolivia', ''),
             ('Bulgaria', ''),('Burkina Faso', ''),('Croatia', ''),("Cote d'Ivoire", ''),('Czech Republic', ''),
             ('Dominican Republic', ''),('Finland', ''),('France', ''),('Greece', ''),('Guatemala', ''),('Iceland', ''),
             ('India', ''),('Ireland', ''),('Israel', ''),('Kosovos', ''),('Latvia', ''),('Mongolia', ''),('Myanmar', ''),
             ('Nepal', ''),('Norway', ''),('Oman', ''),('Puerto Rico', ''),('Romania', ''),('Russia', ''),('Saudi Arabia', ''),
             ('Slovenia', ''),('Tajikistan', ''),('Trinidad and Tobago', ''),('Uganda', ''),('Ukraine', ''),
             ('United Arab Emirates', ''),('United States', 'California'),('United States', 'Georgia'),
             ('United States', 'Idaho'),('United States', 'New Hampshire'),('United States', 'North Carolina'),('Uruguay', ''),
             ('Venezuela', ''),('Zambia', '')] 
Cluster_4 = [('United States', 'South Carolina')]
Cluster_6 = [('Cameroon', ''),('Ethiopia', ''),('Jordan', ''),('Uzbekistan', ''),('Zimbabwe', '')]
Cluster_7 = [('Eswatini', ''),('Kenya', ''),('Libya', ''),('Singapore', ''),('Suriname', ''),('United States', 'Illinois')]
Cluster_10 = [('Algeria', ''), ('Iran', ''), ('Morocco', ''), ('United States', 'Texas')]
Cluster_11 = [('United States', 'Florida')]
Cluster_v0 = [ ('Afghanistan', ''), ('Bahamas', ''), ('Azerbaijan', ''), ('Burundi', ''), ('Comoros', ''), 
            ('Democratic Republic of Congo', ''), ('Hong Kong', ''), ('Indonesia', ''), ('Kazakhstan', ''), 
            ('Kyrgyz Republic', ''), ('Mauritius', ''), ('New Zealand', ''), ('Nicaragua', ''), ('Sudan', ''), 
            ('Taiwan', '')]


class Positive(Constraint):

    def __call__(self, w):
        return K.abs(w)
# [NEW] Nueva clase para establecer que NPIs se pueden activar o no
class ModelPreset(object):
    VAC_H7 = 1
    VAC_H7_SUS = 2
    VAC_SUS = 3
    VAC_NONE = 4
    VAC_REINF = 5
    


class ValenciaPredictor(object):
    """
    A class that computes a fitness for Prescriptor candidates.
    """
    # [NEW] En el nuevo código, el predictor recibe más parámetros
    #    * vac_model_weights = Los pesos del modelo con vacunación
    #    * model_presets = Los NPIs a usar del modelo
    #    * load_default_weights = Si se cargan los pesos por defecto
    def __init__(self, vac_model_weights = None, model_preset = ModelPreset.VAC_H7, load_default_model=False):
    #def __init__(self):
        # Carga el modelo y sus pesos
        #self.model = self._create_model_default(MODEL_WEIGHTS_DEFAULT_FILE)
        # [NEW] Ahora es un atributo las NPIS a usar
        self.set_model_preset(model_preset)
        nb_context = 1  # Only time series of new cases rate is used as context
        nb_action = len(NPI_COLUMNS) #[NEW] Ahora NPI_COLUMNS pasa a ser una variable global que toma el valor según el preset que se haya elegido (Las NPIS a usar) 
        
        #[NEW] Ahora se carga el modelo con los pesos por defecto, es decir se acabo eso de crar 40 modelos con diferentes pesos y diferentes  clusters 
        if(vac_model_weights is not None):
            self.model, _ = self.create_model(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
            self.model.load_weights(vac_model_weights)
        elif load_default_model:
            self.model, _ = self.create_model(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
            self.model.load_weights(MODEL_WEIGHTS_DEFAULT_FILE)      
        """
        self.model_v0 = self._create_model_sco_v0(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
        self.model_v0.load_weights(MODEL_WEIGHTS_SCO_V0_FILE)
        self.model_v1 = self._create_model_sco_v1(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
        self.model_v1.load_weights(MODEL_WEIGHTS_SCO_V1_FILE)
        self.model_v2 = self._create_model_sco_v2(nb_context=nb_context, nb_action=nb_action, lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
        self.model_v2.load_weights(MODEL_WEIGHTS_SCO_V2_FILE)
        self.cluster_dict = self._load_clusters()
        """
        self.df = self._prepare_dataframe()
        print("ValenciaPredictor ready")    
    #"""

    #[NEW] Ahora tenemos un modelo estatico que se encarga de modificar las variables en funcion de los NPIs que se quieran usar
    #esto lo hace gracias a que usa varibles globales. OJO con estas variables
    @staticmethod
    def set_model_preset(preset):
        global NPI_COLUMNS, USE_VAC_PREDICTION_RATIO, MODEL_WEIGHTS_DEFAULT_FILE, REINFECTION_RATE

        if preset == ModelPreset.VAC_H7:
            NPI_COLUMNS = NPI_COLUMNS_8C_5H
            USE_VAC_PREDICTION_RATIO = False
            MODEL_WEIGHTS_DEFAULT_FILE = MODEL_WEIGHTS_VAC_H7_FILE
        elif preset == ModelPreset.VAC_H7_SUS:
            NPI_COLUMNS = NPI_COLUMNS_8C_5H
            USE_VAC_PREDICTION_RATIO = True
            MODEL_WEIGHTS_DEFAULT_FILE = MODEL_WEIGHTS_VAC_H7_SUS_FILE
        elif preset == ModelPreset.VAC_SUS:
            NPI_COLUMNS = NPI_COLUMNS_8C_4H
            USE_VAC_PREDICTION_RATIO = True #change
            MODEL_WEIGHTS_DEFAULT_FILE = MODEL_WEIGHTS_VAC_SUS_FILE
        else: # NONE y REINF
            NPI_COLUMNS = NPI_COLUMNS_8C_4H
            USE_VAC_PREDICTION_RATIO = False
            MODEL_WEIGHTS_DEFAULT_FILE = MODEL_WEIGHTS_VAC_NONE_FILE
        if preset == ModelPreset.VAC_REINF:
            print("Ha cambiado el valor de reinfection_rate")
            REINFECTION_RATE = 0.75
        else:
            REINFECTION_RATE = 0.1

    #"""

    def predict_df(self, start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
        # Load historical intervention plans, since inception
        hist_ips_df = self._load_original_data(path_to_ips_file)
        return self.predict_from_df(start_date_str, end_date_str, hist_ips_df, verbose=verbose)


    def predict_from_df(self,
                        start_date_str: str,
                        end_date_str: str,
                        npis_df: pd.DataFrame, 
                        verbose=False) -> pd.DataFrame:
        """
        Generates a file with daily new cases predictions for the given countries, regions and npis, between
        start_date and end_date, included.
        :param start_date_str: day from which to start making predictions, as a string, format YYYY-MM-DDD
        :param end_date_str: day on which to stop making predictions, as a string, format YYYY-MM-DDD
        :param path_to_ips_file: path to a csv file containing the intervention plans between inception_date and end_date
        :param verbose: True to print debug logs
        :return: a Pandas DataFrame containing the predictions
        """
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        nb_days = (end_date - start_date).days + 1

        # Load historical intervention plans, since inception
        hist_ips_df = npis_df
        # Fill any missing NPIs by assuming they are the same as previous day
        for npi_col in NPI_COLUMNS:
            hist_ips_df.update(hist_ips_df.groupby(['CountryName', 'RegionName'])[npi_col].ffill().fillna(0))

        # Intervention plans to forecast for: those between start_date and end_date
        ips_df = hist_ips_df[(hist_ips_df.Date >= start_date) & (hist_ips_df.Date <= end_date)]
        
        reinfect = pd.DataFrame({},[])
        
        # Make predictions for each country,region pair
        geo_pred_dfs = []
        for g in ips_df.GeoID.unique():
            
            sus_vector = []
            
            if verbose:
                print('\nPredicting for', g)
                        
            # Pull out all relevant data for country c
            ips_gdf = ips_df[ips_df.GeoID == g]
            hist_ips_gdf = hist_ips_df[hist_ips_df.GeoID == g]
            hist_cases_gdf = self.df[self.df.GeoID == g]
            last_known_date = hist_cases_gdf.Date.max()
            
            # Start predicting from start_date, unless there's a gap since last known date
            current_date = min(last_known_date + np.timedelta64(1, 'D'), start_date)

            past_cases_gdf = hist_cases_gdf[hist_cases_gdf.Date < current_date]
            past_ips_gdf = hist_ips_gdf[hist_ips_gdf.Date < current_date]
            future_ips_gdf = hist_ips_gdf[(hist_ips_gdf.Date >= current_date) & (hist_ips_gdf.Date <= end_date)]
            
            past_cases = np.array(past_cases_gdf[CASES_COL]).flatten()
            past_npis = np.array(past_ips_gdf[NPI_COLUMNS])
            future_npis = np.array(future_ips_gdf[NPI_COLUMNS])
            #"""
        #[NEW] Ahora controlamos el ratio de vacunación e inmunidad de la población
            if USE_VAC_PREDICTION_RATIO:
                if verbose:
                    print(' ... Using vaccination info')
                #Aqui falla, porque no esta la columna de inmunizados, para solucionarlo, he añadido la columna de inmunizados #TODO
                past_ips_gdf[IMMUNIZED_COL] = self.df[IMMUNIZED_COL]
                future_ips_gdf[IMMUNIZED_COL] = self.df[IMMUNIZED_COL]
                past_prop_immunized = np.array(past_ips_gdf[IMMUNIZED_COL]).flatten()
                future_prop_immunized = np.array(future_ips_gdf[IMMUNIZED_COL]).flatten()
            else:
                if verbose:
                    print(' ... Setting immunization proportion to default value')
                past_prop_immunized = np.array([0] * past_npis.shape[0])
                future_prop_immunized = np.array([0] * future_npis.shape[0])
       # """
        
            pop_size = hist_cases_gdf.Population.max()
            
            past_cum_cases = np.cumsum(past_cases)
            zn = np.array(compute_7days_mean(past_cases))
            # [NEW] Ahora controlamos el ratio inmunidad de la población
            rn = np.array(compute_rns(past_cum_cases, zn, pop_size, past_prop_immunized))
            #rn = np.array(compute_rns(past_cum_cases, zn, pop_size))
            # [NEW] Tambien controlamos el ratio de reinfección
            #SUS_REINFECTION = REINFECTION_RATE*past_cum_cases.sum() # ~580000 en CV
            #print('Susceptibles de reinfeccion', SUS_REINFECTION)
            #[NEW] Se acabaron los clusters el if con el dict, se fue
            current_model = self.model
            """
            # Loads custom model
            cluster_id = self.cluster_dict.get(g)
            if cluster_id is None:
                current_model = self.model_v1
            elif cluster_id == -1:
                current_model = self.model_v0
            else:
                file_name = MODEL_WEIGHTS_CLUSTER_FILE.format(cluster_id)
                current_model = self._create_model(file_name)
            """
            # Make prediction for each day
            geo_preds = []
            geo_ratios = []
            days_ahead = 0
            try:
                SUS_REINFECTION = REINFECTION_RATE*(past_cum_cases.sum()) # ~580000 en CV
                #SUS_REINFECTION = REINFECTION_RATE*(past_cum_cases[-1]) # ~580000 en CV
            except:
                SUS_REINFECTION = 0
            #TODO 19/01/2023
            #SUS_REINFECTION = REINFECTION_RATE*past_cum_cases.sum() # ~580000 en CV 
            while current_date <= end_date:
                
                #SUS_REINFECTION = REINFECTION_RATE*past_cum_cases.sum() # ~580000 en CV
                
                # Prepare data
                #X_rns = rn[-NB_LOOKBACK_DAYS:].reshape(1, 21, 1)
                #X_npis = past_npis[-NB_LOOKBACK_DAYS:].reshape(1, 21, 8)
                #[NEW] Ahora la preparacion de los datos es diferente
                X_rns = rn[-NB_LOOKBACK_DAYS:].reshape(1, NB_LOOKBACK_DAYS, 1)
                X_npis = past_npis[-NB_LOOKBACK_DAYS:].reshape(1, NB_LOOKBACK_DAYS, len(NPI_COLUMNS))
                
                # Make the prediction (reshape so that sklearn is happy)
                pred_rn = current_model.predict([X_rns, X_npis])[0][0]
                #[NEW] SUS_REINFECTION es el numero de susceptibles de reinfeccion y se tiene en cuenta ala hora de predecir los casos. Además se tiene en cuenta tambien la inmunidad de la población
                #print(pop_size,SUS_REINFECTION)
                pred_cases = int(((((pred_rn * ((pop_size - past_cum_cases[-1] + SUS_REINFECTION) / pop_size) * (1 - past_prop_immunized[-1])) - 1.0) * 7.0 * zn[-1])) + past_cases[-7])
                
                #Cambio acumulacion 19/01/23
                #pred_cases = int(((((pred_rn * ((pop_size - past_cum_cases[-1] + SUS_REINFECTION) / pop_size) * (1 - past_prop_immunized[-1])) - 1.0) * 7.0 * zn[-1])) + past_cases[-7])#cambiar el past_prop_immunized[-1] con el sum
                
                #pred_cases = int(((((pred_rn * ((pop_size - past_cum_cases[-1]) / pop_size)) - 1.0) * 7.0 * zn[-1])) + past_cases[-7])

                pred = max(0, pred_cases)  # Do not allow predicting negative cases
                # Add if it's a requested date
                if current_date >= start_date:
                    geo_preds.append(pred)
                    geo_ratios.append(pred_rn)
                    if verbose:
                        print(f"{current_date.strftime('%Y-%m-%d')}: {pred}")
                else:
                    if verbose:
                        print(f"{current_date.strftime('%Y-%m-%d')}: {pred} - Skipped (intermediate missing daily cases)")

                # Append the prediction and npi's for next day
                # in order to rollout predictions for further days.
                past_cases = np.append(past_cases, pred)
                past_npis = np.append(past_npis, future_npis[days_ahead:days_ahead + 1], axis=0)
                past_cum_cases = np.append(past_cum_cases, past_cum_cases[-1] + pred)
                #[NEW] Ahora controlamos el ratio inmunidad de la población
                past_prop_immunized = np.append(past_prop_immunized, future_prop_immunized[days_ahead:days_ahead + 1], axis=0)
                zn = np.append(zn, compute_last_7days_mean(past_cases))
                rn = np.append(rn, pred_rn) # compute_last_rn(past_cum_cases, zn, pop_size)

                # Move to next day
                current_date = current_date + np.timedelta64(1, 'D')
                days_ahead += 1
                #print("Saldran los diarios", SUS_REINFECTION)

                #Vector de susceptibles a reinferctarse
                sus_vector.append(SUS_REINFECTION)
                #print(sus_vector)
                #print("El vector de reinfectados es", sus_vector)

            
            # we don't have historical data for this geo: return zeroes
            if len(geo_preds) != nb_days:
                geo_preds = [0] * nb_days
                geo_ratios = [0] * nb_days
 
            if g=='Mauritania':
                geo_preds = [140] * nb_days
                geo_ratios = [0] * nb_days
            if g=='Yemen':
                geo_preds = [5] * nb_days
                geo_ratios = [0] * nb_days

            # Create geo_pred_df with pred column
            geo_pred_df = ips_gdf[ID_COLS].copy()
            geo_pred_df['PredictedDailyNewCases'] = geo_preds
            geo_pred_df['PredictedDailyNewRatios'] = geo_ratios
            geo_pred_dfs.append(geo_pred_df)
            #Ponerlo en un data
            #if len(sus_vector)!= 0:
                #reinfect.insert(0, g, sus_vector)
            #print(reinfect)

            
        # Combine all predictions into a single dataframe
        pred_df = pd.concat(geo_pred_dfs)

        # Drop GeoID column to match expected output format
        pred_df = pred_df.drop(columns=['GeoID'])
        
        #Pasarlo a csv
        reinfect.to_csv("reinfectados.csv")


        return pred_df


    def _load_cluster(self, country_list, cluster_id, cluster_dict):
        for country, region in country_list:
            geo_id = country if region=='' else "{} / {}".format(country, region)
            cluster_dict[geo_id] = cluster_id
 

    def _load_clusters(self):
        cluster_dict = dict()
        self._load_cluster(Cluster_1, 1, cluster_dict)
        self._load_cluster(Cluster_2, 2, cluster_dict)
        self._load_cluster(Cluster_3, 3, cluster_dict)
        self._load_cluster(Cluster_4, 4, cluster_dict)
        self._load_cluster(Cluster_6, 6, cluster_dict)
        self._load_cluster(Cluster_7, 7, cluster_dict)
        self._load_cluster(Cluster_10, 10, cluster_dict)
        self._load_cluster(Cluster_11, 11, cluster_dict)
        self._load_cluster(Cluster_v0, -1, cluster_dict)

        return cluster_dict
    
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """
        Loads the Oxford dataset, cleans it up and prepares the necessary columns. Depending on options, also
        loads the Johns Hopkins dataset and merges that in.
        :return: a Pandas DataFrame with the historical data
        """
        # Original df from Oxford
        df1 = self._load_original_data(DATA_FILE_PATH)
        
        if INCLUDE_CV_PREDICTION:
            # Adds CV data
            df1_cv = self._load_original_data(DATA_FILE_CV_PATH)
            df1 = df1.append(df1_cv)

        # Additional context df (e.g Population for each country)
        df2 = self._load_additional_context_df()
        #[NEW] Ahora cargamos el fichero de datos de vacunación
        # Vaccination data df
        df3 = self._load_vaccination_data_df()
        if INCLUDE_CV_PREDICTION:
        #    # Adds CV data
            df3_cv = self._load_vaccination_data_cv_df()
            df3 = df3.append(df3_cv)
        # Merge the 2 DataFrames
        df = df1.merge(df2, on=['GeoID'], how='left', suffixes=('', '_y'))
        #[NEW]
        #"""
        # Add vaccination column
        df = df.merge(df3, on=['GeoID', 'Date'], how='left', suffixes=('', '_z'))
        min_date = df.Date.min()
        df.loc[df.Date == min_date, 'VaccinationsPerHundred'] = 0
        df.loc[df.Date == min_date, 'FullVaccinationsPerHundred'] = 0
        df.VaccinationsPerHundred = df.VaccinationsPerHundred.ffill()
        df.FullVaccinationsPerHundred = df.FullVaccinationsPerHundred.ffill()

        # Add immunization columns
        df['PartialImmunizedPeople'] = df.groupby('GeoID')['VaccinationsPerHundred'].shift(periods=PARTIAL_VAC_IMMUNIZATION_PERIOD, fill_value=0)
        df['FullyImmunizedPeople'] = df.groupby('GeoID')['FullVaccinationsPerHundred'].shift(periods=FULLY_VAC_IMMUNIZATION_PERIOD, fill_value=0)
        df['ProportionImmunized'] = ((df['PartialImmunizedPeople'] - df['FullyImmunizedPeople']) * PARTIAL_VAC_IMMUNIZATION_PROB + df['FullyImmunizedPeople'] * FULLY_VAC_IMMUNIZATION_PROB) / 100.0
        #"""
        
        # Drop countries with no population data
        df.dropna(subset=['Population'], inplace=True)

        #  Keep only needed columns
        columns = CONTEXT_COLUMNS + NPI_COLUMNS
        df = df[columns]

        # Fill in missing values
        self._fill_missing_values(df)

        # Compute number of new cases and deaths each day
        df['NewCases'] = df.groupby('GeoID').ConfirmedCases.diff().fillna(0)
        df['NewDeaths'] = df.groupby('GeoID').ConfirmedDeaths.diff().fillna(0)

        # Replace negative values (which do not make sense for these columns) with 0
        df['NewCases'] = df['NewCases'].clip(lower=0)
        df['NewDeaths'] = df['NewDeaths'].clip(lower=0)

        # Compute smoothed versions of new cases and deaths each day
        df['SmoothNewCases'] = df.groupby('GeoID')['NewCases'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)
        df['SmoothNewDeaths'] = df.groupby('GeoID')['NewDeaths'].rolling(
            WINDOW_SIZE, center=False).mean().fillna(0).reset_index(0, drop=True)

        # Compute percent change in new cases and deaths each day
        df['CaseRatio'] = df.groupby('GeoID').SmoothNewCases.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1
        df['DeathRatio'] = df.groupby('GeoID').SmoothNewDeaths.pct_change(
        ).fillna(0).replace(np.inf, 0) + 1
        #[NEW]
       # """
        # Add column for proportion of population infected
        df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

        # Create column of value to predict
        df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])
        df['PredictionRatioVac'] = df['PredictionRatio'] / (1 - df['ProportionImmunized'])
        #"""
        # Add column for proportion of population infected
        df['ProportionInfected'] = df['ConfirmedCases'] / df['Population']

        # Create column of value to predict
        df['PredictionRatio'] = df['CaseRatio'] / (1 - df['ProportionInfected'])

        return df

    @staticmethod
    def _load_original_data(data_url):
        latest_df = pd.read_csv(data_url,
                                parse_dates=['Date'],
                                encoding="ISO-8859-1",
                                dtype={"RegionName": str,
                                       "RegionCode": str},
                                error_bad_lines=False)
        # GeoID is CountryName / RegionName
        # np.where usage: if A then B else C
        latest_df["GeoID"] = np.where(latest_df["RegionName"].isnull(),
                                      latest_df["CountryName"],
                                      latest_df["CountryName"] + ' / ' + latest_df["RegionName"])
        return latest_df

    @staticmethod
    def _fill_missing_values(df):
        """
        # Fill missing values by interpolation, ffill, and filling NaNs
        :param df: Dataframe to be filled
        """
        df.update(df.groupby('GeoID').ConfirmedCases.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of cases is available
        df.dropna(subset=['ConfirmedCases'], inplace=True)
        df.update(df.groupby('GeoID').ConfirmedDeaths.apply(
            lambda group: group.interpolate(limit_area='inside')))
        # Drop country / regions for which no number of deaths is available
        df.dropna(subset=['ConfirmedDeaths'], inplace=True)
        for npi_column in NPI_COLUMNS:
            df.update(df.groupby('GeoID')[npi_column].ffill().fillna(0))

    @staticmethod
    def _load_additional_context_df():
        # File containing the population for each country
        # Note: this file contains only countries population, not regions
        additional_context_df = pd.read_csv(ADDITIONAL_CONTEXT_FILE,
                                            usecols=['CountryName', 'Population'])
        additional_context_df['GeoID'] = additional_context_df['CountryName']

        # US states population
        additional_us_states_df = pd.read_csv(ADDITIONAL_US_STATES_CONTEXT,
                                              usecols=['NAME', 'POPESTIMATE2019'])
        # Rename the columns to match measures_df ones
        additional_us_states_df.rename(columns={'POPESTIMATE2019': 'Population'}, inplace=True)
        # Prefix with country name to match measures_df
        additional_us_states_df['GeoID'] = US_PREFIX + additional_us_states_df['NAME']

        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_us_states_df)

        # UK population
        additional_uk_df = pd.read_csv(ADDITIONAL_UK_CONTEXT)
        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_uk_df)

        # Brazil population
        additional_brazil_df = pd.read_csv(ADDITIONAL_BRAZIL_CONTEXT)
        # Append the new data to additional_df
        additional_context_df = additional_context_df.append(additional_brazil_df)

        if INCLUDE_CV_PREDICTION:
            additional_cv_df = pd.DataFrame(data = {'CountryName': ['Spain'], 'Population': [5003769], 'GeoID': ['Spain / ComunidadValenciana']})
            additional_context_df = additional_context_df.append(additional_cv_df)
        
        return additional_context_df

    #[NEW] Se han añadido funciones estaticas para la carga de vacunación 
    @staticmethod
    def _load_vaccination_data_df():
        # File containing the population for each country
        # Note: this file contains only countries population, not regions
        #TODO cambiar el nombre de la variable
        vaccination_df = pd.read_csv("https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv?raw=true",
                        parse_dates=['date'],
                        usecols=['location', 'date', 'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred'])
        vaccination_df.rename(columns={'location' : 'GeoID', 
                                       'date': 'Date',
                                       'people_vaccinated_per_hundred': 'VaccinationsPerHundred',
                                       'people_fully_vaccinated_per_hundred': 'FullVaccinationsPerHundred'}, 
                              inplace=True)

        country_mapping_list = {'GeoID': { 'England' : 'United Kingdom / England',
                    'Northern Ireland' : 'United Kingdom / Northern Ireland', 
                    'Scotland' : 'United Kingdom / Scotland',
                    'Wales' : 'United Kingdom / Wales', 
                    'Kyrgyzstan' : 'Kyrgyz Republic', 
                    'Timor' : 'Timor-Leste',
                    'Slovakia' : 'Slovak Republic', 
                    'Czechia' : 'Czech Republic' } }
        vaccination_df.replace(to_replace=country_mapping_list, inplace=True)
        return vaccination_df

    @staticmethod
    def _load_vaccination_data_cv_df():
        vaccination_df = pd.read_csv(ADDITIONAL_VAC_CV_URL,
                        parse_dates=['Fecha publicación'], thousands='.',
                        usecols=['Fecha publicación', 'cod_ine', 'Personas con al menos una dosis', 'Personas con pauta completa'])
        vaccination_df = vaccination_df[vaccination_df.cod_ine==10]
        vaccination_df.fillna(0, inplace=True)
        
        vaccination_df['VaccinationsPerHundred'] = vaccination_df['Personas con al menos una dosis'] * 100 / ADDITIONAL_CV_POPULATION
        vaccination_df['FullVaccinationsPerHundred'] = vaccination_df['Personas con pauta completa'] * 100 / ADDITIONAL_CV_POPULATION
                                                                
        vaccination_df['GeoID'] = 'Spain / ComunidadValenciana'
        vaccination_df.rename(columns={'Fecha publicación' : 'Date'}, inplace=True)
        vaccination_df.drop(columns=['cod_ine', 'Personas con al menos una dosis', 'Personas con pauta completa'],inplace=True)

        # Parche para evitar que las parciales sean menores que las completas
        vaccination_df['VaccinationsPerHundred'] = vaccination_df[['VaccinationsPerHundred', 'FullVaccinationsPerHundred']].max(axis=1)
        
        return vaccination_df
    @staticmethod
    def _create_model_sco_v0(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((nb_lookback_days, nb_context))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _lstm_rn = LSTM(units=lstm_size)(_conv1d_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm_rn)
        model_rn = Model(_input_rn, _output_rn)
        #print(model_rn.summary())
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])

        _input_hn = Input((nb_lookback_days, nb_action))
        _conv1d_hn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_hn)
        _lstm_hn = LSTM(units=lstm_size, kernel_constraint=Positive(), recurrent_constraint=Positive(), bias_constraint=Positive(),return_sequences=False)(_conv1d_hn)
        _output_hn_med = Dense(10, activation="sigmoid")(_lstm_hn)
        _output_hn = Dense(1, activation="sigmoid")(_output_hn_med)
        model_hn = Model(_input_hn, _output_hn)
        #print(model_hn.summary())
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])

        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])
    
        return combined


    @staticmethod
    def _create_model_sco_v1(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((nb_lookback_days, nb_context))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _lstm_rn = Bidirectional(LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(), return_sequences=False))(_conv1d_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm_rn)
        model_rn = Model(_input_rn, _output_rn)
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        _input_hn = Input((nb_lookback_days, nb_action))
        _lstm_hn = LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(),return_sequences=False)(_input_hn)
        _output_hn_med = Dense(10, activation="sigmoid")(_lstm_hn)
        _output_hn = Dense(1, activation="sigmoid")(_output_hn_med)
        model_hn = Model(_input_hn, _output_hn)
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])
    
        return combined

    @staticmethod
    def _create_model_sco_v2(nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((nb_lookback_days, nb_context))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _lstm_rn = Bidirectional(LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(), return_sequences=False))(_conv1d_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm_rn)
        model_rn = Model(_input_rn, _output_rn)
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        _input_hn = Input((nb_lookback_days, nb_action))
        _lstm_hn = LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(),return_sequences=False)(_input_hn)
        _output_hn_med = Dense(10, activation="sigmoid")(_lstm_hn)
        _output_hn = Dense(1, activation="sigmoid")(_output_hn_med)
        model_hn = Model(_input_hn, _output_hn)
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])
    
        return combined
    #[NEW] Se ha creado otro modelo con capas diferentes
    def create_model(self, nb_context, nb_action, lstm_size=32, nb_lookback_days=21):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((nb_lookback_days, nb_context))
        _conv1d_rn = Conv1D(filters=64, kernel_size=5, activation="relu")(_input_rn)
#     SI ACTIVAMOS UNA SEGUNDA CAPA DE COVOLUCIONAL CAMBIAMOS LOS FILTROS DE LA PRIMERA A 32.
#     PRUEBA PARA ALDO. LAS DOS CAPAS DEBERÍAN SACAR MÁS INFORMACIÓN TEMPORAL DE LA SERIE
#        _dropout = Dropout(0.1)(_conv1d_rn)
#        _conv1d2_rn = Conv1D(filters=64, kernel_size=5, activation="relu")(_dropout)
        _maxpool_rn = MaxPooling1D(pool_size=2)(_conv1d_rn)
        _lstm_rn = Bidirectional(LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(), return_sequences=False))(_maxpool_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm_rn)
        model_rn = Model(_input_rn, _output_rn)
        #print(model_rn.summary())
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        _input_hn = Input((nb_lookback_days, nb_action))
 #       _conv1d_hn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_hn)
        _lstm_hn = LSTM(units=lstm_size, kernel_constraint=NonNeg(), recurrent_constraint=NonNeg(), bias_constraint=NonNeg(),return_sequences=False, name='lstm_hn')(_input_hn)
        _output_hn_med = Dense(10, activation="sigmoid", name='output_hn_med')(_lstm_hn)
        _output_hn = Dense(1, activation="sigmoid", name='output_hn')(_output_hn_med)
        model_hn = Model(_input_hn, _output_hn)
        #print(model_hn.summary())
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["mae"])

        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])
    
        # Create training model, which includes loss to measure
        # variance of action_output predictions
        training_model = Model(inputs=[_input_rn, _input_hn],outputs=[lambda_layer])
        training_model.compile(loss='mae', optimizer='adam')

        return combined, training_model

    @staticmethod
    def _create_model(path_to_model_weights):
        def join_layer(tensor):
            rn, an = tensor
            result = (1 - abs(an)) * rn
            return (result)

        _input_rn = Input((21, 1))
        _conv1d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_input_rn)
        _conv2d_rn = Conv1D(filters=64, kernel_size=8, activation="relu")(_conv1d_rn)
        _lstm_rn = Bidirectional(LSTM(32, return_sequences=True))(_conv2d_rn)
        _lstm2_rn = Bidirectional(LSTM(32))(_lstm_rn)
        _output_rn = Dense(1, activation="softplus")(_lstm2_rn)

        model_rn = Model(_input_rn, _output_rn)
        model_rn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])
        _input_hn = Input((21, 8))
        _lstm_hn = LSTM(32)(_input_hn)
        _output_hn = Dense(1, activation="sigmoid")(_lstm_hn)

        model_hn = Model(_input_hn, _output_hn)
        model_hn.compile(loss="mae", optimizer=Adam(), metrics=["accuracy"])
        lambda_layer = Lambda(join_layer, name="lambda_layer")([_output_rn, _output_hn])

        combined = Model(inputs=[_input_rn, _input_hn], outputs=lambda_layer)
        combined.compile(loss="mean_absolute_error", optimizer=Adam(), metrics=["mean_absolute_error"])

        combined.load_weights(path_to_model_weights)

        return combined
    



#[NEW]
# ---------------------------------------------------------
    #   Training code
    # ---------------------------------------------------------

# Train model
    def _train_model(self, training_model, X_context, X_action, y, epochs=1, verbose=0):
        early_stopping = EarlyStopping(patience=20,restore_best_weights=True)
        history = training_model.fit([X_context, X_action], [y], epochs=epochs, batch_size=32,  validation_split=0.1, callbacks=[early_stopping], verbose=verbose)
        return history

    def train(self, geos = None, start_date_str = "2020-03-01", end_date_str = "2020-12-20", min_vaccination_percentage=None):
        print("Creating numpy arrays for Keras for each country...")
        if geos is None:
            if min_vaccination_percentage is not None:
                countries_with_vaccination_data = self.df[self.df.ProportionImmunized > min_vaccination_percentage].GeoID.unique()
                #print("Countries with vaccination data:", countries_with_vaccination_data)
                self.df = self.df[self.df.GeoID.isin(countries_with_vaccination_data)]
            geos = self._most_affected_geos(self.df, MAX_NB_COUNTRIES, NB_LOOKBACK_DAYS)
        print("Columnas del entrenamiento:", self.df.columns)
        country_samples = self._create_country_samples(self.df, geos, start_date_str, end_date_str)
        print("Training for geos:", geos)
        print("Numpy arrays created from", start_date_str, "to", end_date_str)

        # Aggregate data for training
        all_X_context_list = [country_samples[c]['X_train_context'] for c in country_samples]
        #print("all_X_context_list shape:", all_X_context_list)
        all_X_action_list = [country_samples[c]['X_train_action'] for c in country_samples]
        all_y_list = [country_samples[c]['y_train'] for c in country_samples]
        X_context = np.concatenate(all_X_context_list)
        #print("X_context shape:", X_context)
        X_action = np.concatenate(all_X_action_list)
        y = np.concatenate(all_y_list)

        # Clip outliers
        MIN_VALUE = 0.
        MAX_VALUE = 2.
        X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
        y = np.clip(y, MIN_VALUE, MAX_VALUE)

        # Aggregate data for testing only on top countries
        test_all_X_context_list = [country_samples[g]['X_train_context'] for g in geos]
        test_all_X_action_list = [country_samples[g]['X_train_action'] for g in geos]
        test_all_y_list = [country_samples[g]['y_train'] for g in geos]
        test_X_context = np.concatenate(test_all_X_context_list)
        test_X_action = np.concatenate(test_all_X_action_list)
        test_y = np.concatenate(test_all_y_list)

        test_X_context = np.clip(test_X_context, MIN_VALUE, MAX_VALUE)
        test_y = np.clip(test_y, MIN_VALUE, MAX_VALUE)

        # Run full training several times to find best model
        # and gather data for setting acceptance threshold
        models = []
        train_losses = []
        val_losses = []
        test_losses = []
        for t in range(NUM_TRIALS):
            print('Trial', t)
            X_context, X_action, y = self._permute_data(X_context, X_action, y, seed=t)
            model, training_model = self.create_model(nb_context=X_context.shape[-1], nb_action=X_action.shape[-1], lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
            history = self._train_model(training_model, X_context, X_action, y, epochs=1000, verbose=0)
            top_epoch = np.argmin(history.history['val_loss'])
            train_loss = history.history['loss'][top_epoch]
            val_loss = history.history['val_loss'][top_epoch]
            test_loss = training_model.evaluate([test_X_context, test_X_action], [test_y])
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            models.append(model)
            print('Train Loss:', train_loss)
            print('Val Loss:', val_loss)
            print('Test Loss:', test_loss)

        # Gather test info
        country_indeps = []
        country_predss = []
        country_casess = []
        for model in models:
            country_indep, country_preds, country_cases = self._lstm_get_test_rollouts(model, self.df, geos, country_samples)
            country_indeps.append(country_indep)
            country_predss.append(country_preds)
            country_casess.append(country_cases)

        # Compute cases mae
        test_case_maes = []
        for m in range(len(models)):
            total_loss = 0
            for g in geos:
                true_cases = np.sum(np.array(self.df[self.df.GeoID == g].NewCases)[-NB_TEST_DAYS:])
                pred_cases = np.sum(country_casess[m][g][-NB_TEST_DAYS:])
                total_loss += np.abs(true_cases - pred_cases)
            test_case_maes.append(total_loss)

        # Select best model
        best_model = models[np.argmin(test_case_maes)]
        self.model = best_model
        print("Done")
        return best_model
        

    def train_two_phases(self, geos = None, start_date_str = "2020-03-31", end_date_str = "2021-03-31", 
                        start_date_vac_str = "2021-02-15", end_date_vac_str = "2021-06-15", min_vaccination_percentage=None):
        print("Creating numpy arrays for Keras for each country...")
        if geos is None:
            if min_vaccination_percentage is not None:
                countries_with_vaccination_data = self.df[self.df.ProportionImmunized > min_vaccination_percentage].GeoID.unique()
                self.df = self.df[self.df.GeoID.isin(countries_with_vaccination_data)]
            geos = self._most_affected_geos(self.df, MAX_NB_COUNTRIES, NB_LOOKBACK_DAYS)

        # ---------------
        # Phase 1
        # ---------------
        country_samples = self._create_country_samples(self.df, geos, start_date_str, end_date_str)
        print("Training phase-1 for geos:", geos)
        print("Numpy arrays created from", start_date_str, "to", end_date_str)

        # Aggregate data for training
        all_X_context_list = [country_samples[c]['X_train_context'] for c in country_samples]
        all_X_action_list = [country_samples[c]['X_train_action'] for c in country_samples]
        all_y_list = [country_samples[c]['y_train'] for c in country_samples]
        X_context = np.concatenate(all_X_context_list)
        X_action = np.concatenate(all_X_action_list)
        y = np.concatenate(all_y_list)

        # Clip outliers
        MIN_VALUE = 0.
        MAX_VALUE = 2.
        X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
        y = np.clip(y, MIN_VALUE, MAX_VALUE)

        # Aggregate data for testing only on top countries
        test_all_X_context_list = [country_samples[g]['X_train_context'] for g in geos]
        test_all_X_action_list = [country_samples[g]['X_train_action'] for g in geos]
        test_all_y_list = [country_samples[g]['y_train'] for g in geos]
        test_X_context = np.concatenate(test_all_X_context_list)
        test_X_action = np.concatenate(test_all_X_action_list)
        test_y = np.concatenate(test_all_y_list)

        test_X_context = np.clip(test_X_context, MIN_VALUE, MAX_VALUE)
        test_y = np.clip(test_y, MIN_VALUE, MAX_VALUE)

        # Run full training several times to find best model
        # and gather data for setting acceptance threshold
        freeze_layers = {'lstm_hn', 'output_hn_med', 'output_hn'}
        models = []
        train_models = []
        train_losses = []
        val_losses = []
        test_losses = []
        for t in range(NUM_TRIALS):
            print('Trial', t)
            X_context, X_action, y = self._permute_data(X_context, X_action, y, seed=t)
            model, training_model = self.create_model(nb_context=X_context.shape[-1], nb_action=X_action.shape[-1], lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
            history = self._train_model(training_model, X_context, X_action, y, epochs=1000, verbose=0)
            top_epoch = np.argmin(history.history['val_loss'])
            train_loss = history.history['loss'][top_epoch]
            val_loss = history.history['val_loss'][top_epoch]
            test_loss = training_model.evaluate([test_X_context, test_X_action], [test_y])
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            models.append(model)

            # Freeze action model
            for layer in freeze_layers:
                training_model.get_layer(layer).trainable = False
            training_model.compile(loss='mae', optimizer='adam')

            train_models.append(training_model)
            print('Train Loss:', train_loss)
            print('Val Loss:', val_loss)
            print('Test Loss:', test_loss)

        # Gather test info
        country_indeps = []
        country_predss = []
        country_casess = []
        for model in models:
            country_indep, country_preds, country_cases = self._lstm_get_test_rollouts(model, self.df, geos, country_samples)
            country_indeps.append(country_indep)
            country_predss.append(country_preds)
            country_casess.append(country_cases)

        # Compute cases mae
        test_case_maes = []
        for m in range(len(models)):
            total_loss = 0
            for g in geos:
                true_cases = np.sum(np.array(self.df[self.df.GeoID == g].NewCases)[-NB_TEST_DAYS:])
                pred_cases = np.sum(country_casess[m][g][-NB_TEST_DAYS:])
                total_loss += np.abs(true_cases - pred_cases)
            test_case_maes.append(total_loss)

        # Select best model
        model = models[np.argmin(test_case_maes)]
        training_model = train_models[np.argmin(test_case_maes)]

        # ---------------
        # Phase 2
        # ---------------
        country_samples = self._create_country_samples(self.df, geos, start_date_vac_str, end_date_vac_str)
        print("Training phase-2 for geos:", geos)
        print("Numpy arrays created from", start_date_str, "to", end_date_str)

        # Aggregate data for training
        all_X_context_list = [country_samples[c]['X_train_context'] for c in country_samples]
        all_X_action_list = [country_samples[c]['X_train_action'] for c in country_samples]
        all_y_list = [country_samples[c]['y_train'] for c in country_samples]
        X_context = np.concatenate(all_X_context_list)
        X_action = np.concatenate(all_X_action_list)
        y = np.concatenate(all_y_list)

        # Clip outliers
        MIN_VALUE = 0.
        MAX_VALUE = 2.
        X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
        y = np.clip(y, MIN_VALUE, MAX_VALUE)

        # Aggregate data for testing only on top countries
        test_all_X_context_list = [country_samples[g]['X_train_context'] for g in geos]
        test_all_X_action_list = [country_samples[g]['X_train_action'] for g in geos]
        test_all_y_list = [country_samples[g]['y_train'] for g in geos]
        test_X_context = np.concatenate(test_all_X_context_list)
        test_X_action = np.concatenate(test_all_X_action_list)
        test_y = np.concatenate(test_all_y_list)

        test_X_context = np.clip(test_X_context, MIN_VALUE, MAX_VALUE)
        test_y = np.clip(test_y, MIN_VALUE, MAX_VALUE)

        X_context, X_action, y = self._permute_data(X_context, X_action, y, seed=t)
        history = self._train_model(training_model, X_context, X_action, y, epochs=1000, verbose=0)
        top_epoch = np.argmin(history.history['val_loss'])
        train_loss = history.history['loss'][top_epoch]
        val_loss = history.history['val_loss'][top_epoch]
        test_loss = training_model.evaluate([test_X_context, test_X_action], [test_y])
        print('Train Loss:', train_loss)
        print('Val Loss:', val_loss)
        print('Test Loss:', test_loss)

        self.model = training_model
        print("Done")
        return training_model


    def multitrain_dates_and_countries(self, train_geos, test_geos,
                    train_dateranges=[('2020-07-01', '2021-03-31'), ('2020-12-01', '2021-03-31')],
                    test_daterange=('2021-04-01', '2021-04-30')):
        print("Creating numpy arrays for Keras for each country...")
    #        if geos is None:
    #            geos = self._most_affected_geos(self.df, MAX_NB_COUNTRIES, NB_LOOKBACK_DAYS)
        NB_TEST_DAYS = 14 # No es necesaria esta línea, lo vamos a poner a 0
        old_nb_test_days = NB_TEST_DAYS
        NB_TEST_DAYS = 0
        models = []
        for geos in train_geos:
            for train_daterange in train_dateranges:
                start_date_str = train_daterange[0]
                end_date_str = train_daterange[1]
                print("geos: ", geos)
                print("type(geos): ", type(geos))
                print("len(geos): ", len(geos))
                print("type(geos[0]): ", type(geos[0]))
                print("type(start_date_str): ", type(start_date_str))
                print("type(end_date_str): ", type(end_date_str))
                print("start_date_str: ", start_date_str)
                print("end_date_str: ", end_date_str)

                country_samples = self._create_country_samples(self.df, geos, start_date_str, end_date_str)
                print("Numpy arrays created from", start_date_str, "to", end_date_str)
                
                # Aggregate data for training
                all_X_context_list = [country_samples[c]['X_train_context'] for c in country_samples]
                all_X_action_list = [country_samples[c]['X_train_action'] for c in country_samples]
                all_y_list = [country_samples[c]['y_train'] for c in country_samples]
                X_context = np.concatenate(all_X_context_list)
                X_action = np.concatenate(all_X_action_list)
                y = np.concatenate(all_y_list)

                # Clip outliers
                MIN_VALUE = 0.
                MAX_VALUE = 2.
                X_context = np.clip(X_context, MIN_VALUE, MAX_VALUE)
                y = np.clip(y, MIN_VALUE, MAX_VALUE)
                
                # Aggregate data for testing only on top countries
                test_all_X_context_list = [country_samples[g]['X_train_context'] for g in geos]
                test_all_X_action_list = [country_samples[g]['X_train_action'] for g in geos]
                test_all_y_list = [country_samples[g]['y_train'] for g in geos]
                test_X_context = np.concatenate(test_all_X_context_list)
                test_X_action = np.concatenate(test_all_X_action_list)
                test_y = np.concatenate(test_all_y_list)

                test_X_context = np.clip(test_X_context, MIN_VALUE, MAX_VALUE)
                test_y = np.clip(test_y, MIN_VALUE, MAX_VALUE)

                # Run full training several times to find best model
                # and gather data for setting acceptance threshold
    #                models = []
    # =============================================================================
    #                 train_losses = []
    #                 val_losses = []
    #                 test_losses = []
    # =============================================================================
                for t in range(NUM_TRIALS):
                    print('Trial', t)
                    X_context, X_action, y = self._permute_data(X_context, X_action, y, seed=t)
                    model, training_model = self.create_model(nb_context=X_context.shape[-1], nb_action=X_action.shape[-1], lstm_size=LSTM_SIZE, nb_lookback_days=NB_LOOKBACK_DAYS)
                    history = self._train_model(training_model, X_context, X_action, y, epochs=1000, verbose=0)

    # =============================================================================
    #                     top_epoch = np.argmin(history.history['val_loss'])
    #                     train_loss = history.history['loss'][top_epoch]
    #                     val_loss = history.history['val_loss'][top_epoch]
    #                     test_loss = training_model.evaluate([test_X_context, test_X_action], [test_y])
    #                     train_losses.append(train_loss)
    #                     val_losses.append(val_loss)
    #                     test_losses.append(test_loss)
    # =============================================================================
                    models.append(model)
    # =============================================================================
    #                     print('Train Loss:', train_loss)
    #                     print('Val Loss:', val_loss)
    #                     print('Test Loss:', test_loss)
    # =============================================================================
        # Test
        start_date_str = test_daterange[0]
        end_date_str = test_daterange[1]

        # Gets number of test days
        start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
        end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
        NB_TEST_DAYS = (end_date - start_date).days

        # Add previous lookback days to test range
        corrected_start_date = start_date - np.timedelta64(NB_LOOKBACK_DAYS, 'D')
        start_date_str = corrected_start_date.strftime('%Y-%m-%d')

        country_samples = self._create_country_samples(self.df, test_geos, start_date_str, end_date_str)
        
        # Gather test info
        country_indeps = []
        country_predss = []
        country_casess = []
        for model in models:
            country_indep, country_preds, country_cases = self._lstm_get_test_rollouts(model, self.df, test_geos, country_samples)
            country_indeps.append(country_indep)
            country_predss.append(country_preds)
            country_casess.append(country_cases)

        # Compute cases mae
        test_case_maes = []
        for m in range(len(models)):
            print('Model', m)
            print(' -> Training geos: ', train_geos[m // (len(train_dateranges) * NUM_TRIALS)])
            print(' -> Test geos: ', test_geos)
            print(' -> Date range', train_dateranges[(m % (len(train_dateranges) * NUM_TRIALS)) // NUM_TRIALS])
            print(' -> Trial: ', m % NUM_TRIALS)
            total_loss = 0
            for g in test_geos:
                true_cases = np.sum(np.array(self.df[self.df.GeoID == g].NewCases)[-NB_TEST_DAYS:])
                pred_cases = np.sum(country_casess[m][g][-NB_TEST_DAYS:])
                total_loss += np.abs(true_cases - pred_cases)
            print(' -> MAE', total_loss)
            test_case_maes.append(total_loss)
        # Select best model
        best_model = models[np.argmin(test_case_maes)]
        self.model = best_model
        print("Done")

        # Restore number of test days
        NB_TEST_DAYS = old_nb_test_days

        return best_model


    #[NEW]

    @staticmethod
    def _most_affected_geos(df, nb_geos, min_historical_days):
        """
        Returns the list of most affected countries, in terms of confirmed deaths.
        :param df: the data frame containing the historical data
        :param nb_geos: the number of geos to return
        :param min_historical_days: the minimum days of historical data the countries must have
        :return: a list of country names of size nb_countries if there were enough, and otherwise a list of all the
        country names that have at least min_look_back_days data points.
        """
        # By default use most affected geos with enough history
        gdf = df.groupby('GeoID')['ConfirmedDeaths'].agg(['max', 'count']).sort_values(by='max', ascending=False)
        filtered_gdf = gdf[gdf["count"] > min_historical_days]
        geos = list(filtered_gdf.head(nb_geos).index)
        return geos
    #[NEW]
    @staticmethod
    def _most_affected_geos(df, nb_geos, min_historical_days):
        """
        Returns the list of most affected countries, in terms of confirmed deaths.
        :param df: the data frame containing the historical data
        :param nb_geos: the number of geos to return
        :param min_historical_days: the minimum days of historical data the countries must have
        :return: a list of country names of size nb_countries if there were enough, and otherwise a list of all the
        country names that have at least min_look_back_days data points.
        """
        # By default use most affected geos with enough history
        gdf = df.groupby('GeoID')['ConfirmedDeaths'].agg(['max', 'count']).sort_values(by='max', ascending=False)
        filtered_gdf = gdf[gdf["count"] > min_historical_days]
        geos = list(filtered_gdf.head(nb_geos).index)
        return geos

    @staticmethod
    def _create_country_samples(df: pd.DataFrame, geos: list, 
                                start_time_str, end_time_str) -> dict:
        """
        For each country, creates numpy arrays for Keras
        :param df: a Pandas DataFrame with historical data for countries (the "Oxford" dataset)
        :param geos: a list of geo names
        :return: a dictionary of train and test sets, for each specified country
        """
        _start_date = pd.to_datetime(start_time_str, format='%Y-%m-%d')
        _end_date = pd.to_datetime(end_time_str, format='%Y-%m-%d')

        df = df[(df.Date >= _start_date) & (df.Date <= _end_date)]
        action_columns = NPI_COLUMNS
        if USE_VAC_PREDICTION_RATIO:
            context_column = 'PredictionRatioVac'
            outcome_column = 'PredictionRatioVac'
        else:
            context_column = 'PredictionRatio'
            outcome_column = 'PredictionRatio'
        country_samples = {}
        for g in geos:
            cdf = df[df.GeoID == g]
            cdf = cdf[cdf.ConfirmedCases.notnull()]
            context_data = np.array(cdf[context_column])
            action_data = np.array(cdf[action_columns])
            outcome_data = np.array(cdf[outcome_column])
            context_samples = []
            action_samples = []
            outcome_samples = []
            nb_total_days = outcome_data.shape[0]
    #            nb_total_days2 = nb_total_days + 60
            for d in range(NB_LOOKBACK_DAYS, nb_total_days):
                context_samples.append(context_data[d - NB_LOOKBACK_DAYS:d])
                action_samples.append(action_data[d - NB_LOOKBACK_DAYS:d])
                outcome_samples.append(outcome_data[d])
            if len(outcome_samples) > 0:
                X_context = np.expand_dims(np.stack(context_samples, axis=0), axis=2)
                X_action = np.stack(action_samples, axis=0)
                y = np.stack(outcome_samples, axis=0)
                country_samples[g] = {
                    'X_context': X_context,
                    'X_action': X_action,
                    'y': y,
                    'X_train_context': X_context[:-NB_TEST_DAYS],
                    'X_train_action': X_action[:-NB_TEST_DAYS],
                    'y_train': y[:-NB_TEST_DAYS],
                    'X_test_context': X_context[-NB_TEST_DAYS:],
                    'X_test_action': X_action[-NB_TEST_DAYS:],
                    'y_test': y[-NB_TEST_DAYS:],
                }
        return country_samples

    # Shuffling data prior to train/val split
    def _permute_data(self, X_context, X_action, y, seed=301):
        np.random.seed(seed)
        p = np.random.permutation(y.shape[0])
        X_context = X_context[p]
        X_action = X_action[p]
        y = y[p]
        return X_context, X_action, y

    # Functions for computing test metrics
    def _lstm_roll_out_predictions(self, model, initial_context_input, initial_action_input, future_action_sequence):
        nb_test_days = future_action_sequence.shape[0]
        pred_output = np.zeros(nb_test_days)
        context_input = np.expand_dims(np.copy(initial_context_input), axis=0)
        action_input = np.expand_dims(np.copy(initial_action_input), axis=0)
        for d in range(nb_test_days):
            action_input[:, :-1] = action_input[:, 1:]
            action_input[:, -1] = future_action_sequence[d]
            pred = model.predict([context_input, action_input])
            pred_output[d] = pred
            context_input[:, :-1] = context_input[:, 1:]
            context_input[:, -1] = pred
        return pred_output
        
    def _lstm_get_test_rollouts(self, model, df, top_geos, country_samples):
        country_indep = {}
        country_preds = {}
        country_cases = {}
        for g in top_geos:
            X_test_context = country_samples[g]['X_test_context']
            X_test_action = country_samples[g]['X_test_action']
            country_indep[g] = model.predict([X_test_context, X_test_action])

            initial_context_input = country_samples[g]['X_test_context'][0]
            initial_action_input = country_samples[g]['X_test_action'][0]
            y_test = country_samples[g]['y_test']

            nb_test_days = y_test.shape[0]
            nb_actions = initial_action_input.shape[-1]

            future_action_sequence = np.zeros((nb_test_days, nb_actions))
            future_action_sequence[:nb_test_days] = country_samples[g]['X_test_action'][:, -1, :]
            current_action = country_samples[g]['X_test_action'][:, -1, :][-1]
            future_action_sequence[14:] = current_action
        
            preds = self._lstm_roll_out_predictions(model,initial_context_input,initial_action_input,future_action_sequence)
            country_preds[g] = preds

            prev_confirmed_cases = np.array(df[df.GeoID == g].ConfirmedCases)[:-nb_test_days]
            prev_new_cases = np.array(df[df.GeoID == g].NewCases)[:-nb_test_days]
            initial_total_cases = prev_confirmed_cases[-1]
            pop_size = np.array(df[df.GeoID == g].Population)[0]

            pred_new_cases = self._convert_ratios_to_total_cases(preds, WINDOW_SIZE, prev_new_cases, initial_total_cases, pop_size)
            country_cases[g] = pred_new_cases

        return country_indep, country_preds, country_cases

    # Functions for converting predictions back to number of cases
    @staticmethod
    def _convert_ratio_to_new_cases(ratio,
                                    window_size,
                                    prev_new_cases_list,
                                    prev_pct_infected):
        return (ratio * (1 - prev_pct_infected) - 1) * \
                (window_size * np.mean(prev_new_cases_list[-window_size:])) \
                + prev_new_cases_list[-window_size]

    def _convert_ratios_to_total_cases(self,
                                        ratios,
                                        window_size,
                                        prev_new_cases,
                                        initial_total_cases,
                                        pop_size):
        new_new_cases = []
        prev_new_cases_list = list(prev_new_cases)
        curr_total_cases = initial_total_cases
        for ratio in ratios:
            new_cases = self._convert_ratio_to_new_cases(ratio,
                                                            window_size,
                                                            prev_new_cases_list,
                                                            curr_total_cases / pop_size)
            # new_cases can't be negative!
            new_cases = max(0, new_cases)
            # Which means total cases can't go down
            curr_total_cases += new_cases
            # Update prev_new_cases_list for next iteration of the loop
            prev_new_cases_list.append(new_cases)
            new_new_cases.append(new_cases)
        return new_new_cases
        
#[NEW] FUnciones auxiliares
# ---------------------------------------------------------
#   Auxiliar functions
# ---------------------------------------------------------
def compute_7days_mean(casos_diarios):
    '''
    Function to compute the 7 days mean for daily cases (Zt), to smooth the reporting anomalies.
    Input:
        - daily cases list
    Output:
        - daily Zt (7 days mean, current day and the 6 prior it)
    '''
    zn = []
    for i in range(len(casos_diarios)):
        if i == 0:
            zn.append(casos_diarios[i])
        elif i > 0 and i < 6:
            acc = 0
            for j in range(i):
                acc += casos_diarios[i-j]
            zn.append(acc/i)
        else:
            acc = 0
            for j in range(7):
                acc += casos_diarios[i-j]
            zn.append(acc/7)
    return zn

def compute_last_7days_mean(casos_diarios):
    '''
    Function to compute the 7 days mean for the last day (Zt), to smooth the reporting anomalies.
    Input:
        - daily cases list
    Output:
        - last day Zt (7 days mean, current day and the 6 prior it)
    '''
    i = len(casos_diarios) - 1
    if i == 0:
        zn=casos_diarios[i]
    elif i > 0 and i < 6:
        acc = 0
        for j in range(i):
            acc += casos_diarios[i-j]
        zn=acc/i
    else:
        acc = 0
        for j in range(7):
            acc += casos_diarios[i-j]
        zn=acc/7

    return zn



def compute_rns(casos_acumulados, zn, population, prop_immunized=None):#[NEW] added new parameter prop_immunized
    '''
    Function to take into account population size and immunity when calculating Rt.
    Input:
        - cummulated cases list
        - daily Zt means over 7 days (Zn)
        - population for the given country or region
    Output:
        - Rns list
    '''
    # [NEW] added new parameter prop_immunized
    if prop_immunized is None:
        prop_immunized = [0] * len(casos_acumulados)
    rn = []
    for i in range(len(casos_acumulados)):
        if i == 0:
            num = population * zn[i]
            denom = population
            rn.append(num/denom)
        else:
            if zn[i-1] == 0:
                rn.append(0)
            else:
                num = population * zn[i]
                #denom = (population - casos_acumulados[i-1]) * zn[i-1]  # En xprize utilizan casos_acumulados[i]
                #[NEW] added new parameter prop_immunized
                denom = (population - casos_acumulados[i-1]) * (1-prop_immunized[i-1]) * zn[i-1]  # En xprize utilizan casos_acumulados[i]
                rn.append(num/denom)
    rn = [2 if x>2 else x for x in rn] # En xprize no acotan a 2 en la prediccion, solo en el train
    return rn


def compute_last_rn(casos_acumulados, zn, population,prop_immunized=0):#[NEW] added new parameter prop_immunized
    '''
    Function to take into account population size and immunity when calculating Rt.
    Input:
        - cummulated cases list
        - daily Zt means over 7 days (Zn)
        - population for the given country or region
    Output:
        - Last Rn
    '''
    i = len(casos_acumulados) - 1
    if i == 0:
        num = population * zn[i]
        denom = population
        rn=num/denom
    else:
        if zn[i-1] == 0:
            rn=0
        else:
            num = population * zn[i]
            #denom = (population - casos_acumulados[i-1]) * zn[i-1]  # En xprize utilizan casos_acumulados[i]
            #[NEW] added new parameter prop_immunized
            denom = (population - casos_acumulados[i-1]) * (1-prop_immunized) * zn[i-1]  # En xprize utilizan casos_acumulados[i]
            rn=num/denom
    rn = 2 if rn>2 else rn # En xprize no acotan a 2 en la prediccion, solo en el train
    return rn