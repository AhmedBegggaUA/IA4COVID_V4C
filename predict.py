# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os

import numpy as np
import pandas as pd


def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path,
            model_preset=None,
            force_cluster=None) -> None:
    """
    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    preds_df = predict_df(start_date, end_date, path_to_ips_file, verbose=False, model_preset=model_preset, force_cluster=force_cluster)
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")    


def predict_df(start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False, model_preset=None, force_cluster=None):
    """
    Generates a file with daily new cases predictions for the given countries, regions and npis, between
    start_date and end_date, included.
    :param start_date_str: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date_str: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception_date and end_date
    :param verbose: True to print debug logs
    :return: a Pandas DataFrame containing the predictions
    """

    # Load model
    if model_preset is None:
        predictor = ValenciaPredictor(force_cluster)
    else:
        predictor = ValenciaPredictor(model_preset=model_preset, load_default_model=True)

    # Make predictions
    pred_df = predictor.predict_df(start_date_str, end_date_str, path_to_ips_file, verbose)

    return pred_df


# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    parser.add_argument("-m", "--model",
                        dest="model",
                        type=str,
                        required=False,
                        default='nan',
                        help="The path to the CSV file where predictions should be written")
    parser.add_argument("-c", "--force_cluster",
                        dest="force_cluster",
                        type=int,
                        required=False,
                        default=None,
                        help="The path to the CSV file where predictions should be written")

    args = parser.parse_args()
    print(f"Generating predictions from {args.start_date} to {args.end_date}...")

    cluster = None
    if args.model == 'nan':
        print('Loading VALENCIA NAN model')
        from valencia_predictor import ValenciaPredictor
        preset = None
    else:
        print('Loading VALENCIA VAC model')
        from valencia_predictor import ValenciaPredictor
        from valencia_predictor import ModelPreset
        if args.model == 'vac_h7':
            preset = ModelPreset.VAC_H7
        elif args.model == 'vac_h7_sus':
            preset = ModelPreset.VAC_H7_SUS
        elif args.model == 'vac_sus':
            preset = ModelPreset.VAC_SUS
        elif args.model == 'vac_none':
            preset = ModelPreset.VAC_NONE
        else:
            preset = ModelPreset.VAC_REINF

    
    predict(args.start_date, args.end_date, args.ip_file, args.output_file, model_preset=preset, force_cluster=args.force_cluster)
    print("Done!")


