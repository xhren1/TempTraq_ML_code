import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def deal_with_missing_points(temp_data_path,cohort):
    '''
    This function deals with the missing data points (no nan recored in the file) in the temperature data.
    The function will fill the missing data points with nan and output the data to a csv file ("./full_ttemp.csv").
    The function will also record the original data positions in the data.

    Parameters:
    temp_data_path (str): The path to the temperature data file.
    cohort (str): The cohort to be analyzed ("HCT"or"CART").

    Returns:
    final_df (pd.DataFrame): The data with missing points filled with nan and the original data positions recorded.
    
    '''
    # Load the data
    data = pd.read_csv(temp_data_path, sep=',')

    data = data[data['Cohort'] == cohort]
    data = data=data[["MaskID","Time_DPI","TTemp"]]
    data["fever_start"] = 0

    # make time in days to minutes
    data["Time_DPI"] = data["Time_DPI"]*24*60
    
    # make sure all the time points are even
    data["Time_DPI"] = data["Time_DPI"].round(0).apply(lambda x: x if x % 2 == 0 else x+1)

    # reset the index
    data = data.reset_index(drop=True)

    final_df = pd.DataFrame(columns=['MaskID', 'Time_DPI', 'TTemp', 'fever_start'])

    # Loop through each maskid and fill the missing values with nan
    for maskid in data['MaskID'].unique():
        print(maskid)
        time_dpi = data.loc[data['MaskID'] == maskid, 'Time_DPI'].tolist()
        ttemp = data.loc[data['MaskID'] == maskid, 'TTemp'].tolist()
        new_index = range(int(time_dpi[0]), int(time_dpi[-1])+2, 2)
        df = pd.DataFrame({'MaskID': maskid, 'Time_DPI': new_index, 'TTemp': np.nan, 'fever_start': 0, 'Cohort': maskid.split('-')[0]})
        for i, time in enumerate(tqdm(df["Time_DPI"])):
            if time in time_dpi:
                temp_t = ttemp[time_dpi.index(time)]
                df.iloc[i,2] = temp_t
        final_df = pd.concat([final_df, df], axis=0)
        print('-'*80)

    # record the original data positions
    final_df["Orignial_data"]= ~final_df["TTemp"].isna().astype(int) + 2

    # output the data
    final_df.to_csv("../full_ttemp.csv", index=False)

    return final_df


def preprocess(filled_temp_data,fever_start_data_path,cohort):
    '''
    This function preprocess the temperature data by interpolating the missing data points and smoothing the data.
    The function will also mark the fever start points in the data.

    Parameters:
    filled_temp_data (pd.DataFrame): The data with missing points filled with nan and the original data positions recorded.
    fever_start_data_path (str): The path to the fever start data file.
    cohort (str): The cohort to be analyzed ("HCT"or"CART").

    Returns:
    filled_temp_data (pd.DataFrame): The preprocessed data.
    '''
    print("Start preprocessing the data...")
    # Load the data
    fever_start = pd.read_csv(fever_start_data_path, sep=',')
    fever_start = fever_start[fever_start['Cohort'] == cohort]
    fever_start = fever_start[["MaskID","Time_DPI"]]
    # make time in days to minutes
    fever_start["Time_DPI"] = fever_start["Time_DPI"]*24*60
    # make sure all the time points are even
    fever_start["Time_DPI"] = fever_start["Time_DPI"].round(0).apply(lambda x: x if x % 2 == 0 else x+1)

    for maskid in filled_temp_data['MaskID'].unique():
        individual_data = filled_temp_data[filled_temp_data['MaskID'] == maskid][["Time_DPI","TTemp"]]
        individual_data["TTemp"] = individual_data['TTemp'].mask(individual_data['TTemp'] < 35.5, np.nan)
        
        # interpolate the missing data points(nan)
        y_known = np.array(individual_data["TTemp"])
        x_known = np.array(range(len(y_known)))
        known_indices = ~np.isnan(y_known)
        interp_func = interp1d(x_known[known_indices], y_known[known_indices], kind='nearest',fill_value="extrapolate")
        y_interp = interp_func(x_known)
        
        # smooth the data
        temp_filtered = savgol_filter(y_interp, 11, 2)
        filled_temp_data.loc[filled_temp_data['MaskID'] == maskid,"TTemp"] = np.array(temp_filtered)

    # mark the fever start
    for i in range(len(fever_start)):
        mask_id = fever_start.iloc[i,0]
        dpi = fever_start.iloc[i,1]
        filled_temp_data.loc[(filled_temp_data['MaskID'] == mask_id) & (filled_temp_data['Time_DPI'] == dpi), 'fever_start'] = 1 
    print("fever start number check:")
    print(filled_temp_data[filled_temp_data["fever_start"] == 1].shape[0] == fever_start.shape[0])
    print("Finish preprocessing the data!")

    return filled_temp_data

if __name__ == "__main__":
    temp_data_path = "../input/TempTraq_Dataset.csv"
    fever_start_data_path = "../input/TFeverStarts.csv"
    cohort = "HCT"
    temp_df = deal_with_missing_points(temp_data_path,cohort)
    temp_df = preprocess(temp_df,fever_start_data_path,cohort)
    print("Done!")