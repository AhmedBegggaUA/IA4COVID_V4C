import os

import sys
sys.path.append('/Users/ahmedbegga/Desktop/UPV/trabajo/valencia-ia4covid-xprize')

import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from covid_xprize.standard_predictor.xprize_predictor import NPI_COLUMNS
from covid_xprize.scoring.prescriptor_scoring import weight_prescriptions_by_cost
from covid_xprize.scoring.prescriptor_scoring import generate_cases_and_stringency_for_prescriptions
from covid_xprize.scoring.prescriptor_scoring import compute_domination_df
from covid_xprize.scoring.prescriptor_scoring import compute_pareto_set
from covid_xprize.validation.prescriptor_validation import validate_submission

# Can set these longer for better evaluation. Will increase eval time
START_DATE = "2021-3-01"
END_DATE = "2021-6-30"

from covid_xprize.scoring.predictor_scoring import load_dataset
from covid_xprize.validation.scenario_generator import generate_scenario

LATEST_DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_nat_latest.csv'
GEO_FILE = "countries_regions.csv"

latest_df = load_dataset(LATEST_DATA_URL, GEO_FILE)

IP_FILE = "prescriptions/robojudge_test_scenario.csv"
os.makedirs(os.path.dirname(IP_FILE), exist_ok=True)
countries = None
scenario_df = generate_scenario(START_DATE, END_DATE, latest_df, countries, scenario="Freeze")
scenario_df.to_csv(IP_FILE, index=False)
print("Cargados los datos!")
# Cost weightings for each IP for each geo
TEST_COST = "covid_xprize/validation/data/socioeconomic_costs.csv"
# greedy
# Generate feat_greedy prescriptions
from prescribe_feat import prescribe as prescribe_feat
output_file = "prescriptions/feat_greedy_3_6.csv"
print("Ejecutando feat_greedy...")
prescribe_feat(start_date_str=START_DATE,
          end_date_str=END_DATE,
          path_to_hist_file=IP_FILE,
          path_to_cost_file = TEST_COST,
          output_file_path = output_file)
# Generate VALENCIA IA4COVID19 prescriptions
from prescribe import prescribe as prescribe
output_file = "prescriptions/v4c_standar_3_6.csv"
print("Ejecutando v4c_standar_3_6...")
prescribe(start_date_str=START_DATE,
          end_date_str=END_DATE,
          path_to_prior_ips_file=IP_FILE,
          path_to_cost_file = TEST_COST,
          output_file_path = output_file)
prescription_files = {
#     'NeatExample': 'covid_xprize/examples/prescriptors/neat/test_prescriptions/pres.csv',
#    'Random1': 'covid_xprize/examples/prescriptors/random/prescriptions/random_presc_1.csv',
#    'Random2': 'covid_xprize/examples/prescriptors/random/prescriptions/random_presc_2.csv',
#    'BlindGreedy': 'covid_xprize/examples/prescriptors/blind_greedy/prescriptions/blind_greedy.csv',
    'FeatGreedy': 'prescriptions/feat_greedy_3_6.csv',
    'V4C': 'prescriptions/v4c_standar_3_6.csv'
}
# Validate the prescription files
for prescriptor_name, output_file in prescription_files.items():
    errors = validate_submission(START_DATE, END_DATE, IP_FILE, output_file)
    if errors:
        for error in errors:
            print(f"{prescriptor_name}: {error}")
    else:
        print(f"{prescriptor_name}: All good!")

print(TEST_COST)
print(IP_FILE)
# Collect case and stringency data for all prescriptors
dfs = []
for prescriptor_name, prescription_file in sorted(prescription_files.items()):
    print("Generating predictions for", prescriptor_name)
    df, preds = generate_cases_and_stringency_for_prescriptions(START_DATE, END_DATE, prescription_file, TEST_COST, IP_FILE)
    df['PrescriptorName'] = prescriptor_name
    dfs.append(df)
df = pd.concat(dfs)
df.to_csv('stringency_v4c_standar_3_6.csv', index=False)
# Store the predictions for the prescriptors
print("TODO BIEN")
