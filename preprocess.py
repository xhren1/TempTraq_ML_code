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
    print("Start dealing with missing points...")
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
    final_df.to_csv("./full_ttemp.csv", index=False)
    print("Finish dealing with missing points!")
    print('-'*80)

    return final_df


def interpolation_smooth(filled_temp_data,fever_start_data_path,cohort):
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
    print('-'*80)

    return filled_temp_data


def filter_data(data,before,after,original_percent_threshold,fever_cause_data_path):
    '''
    This function picks up the data with fever event hours before and after and filter out the data that has more than 70% original data.
    The function will also convert the fever causes to binary labels, infection:1, non-infection/unclear:0.

    Parameters:
    data (pd.DataFrame): The preprocessed data.
    before (int): The hours before the fever event.
    after (int): The hours after the fever event.
    original_percent_threshold (float): The threshold of the percentage of original data (0-1).
    fever_cause_data_path (str): The path to the fever cause data file.

    Returns:
    temp_array (np.array): The input data for clustering.
    qualified_temp_label (list): The fever event label in (maskid, days after infusion) format.
    qualified_temp_percent (list): The percentage of original data.
    true_label (np.array): The binary labels for fever causes.
    qualified_fever_causes (pd.Series): The fever causes.
    '''
    print("Start picking up and filtering the data...")
    print(f"{before} hours before and {after} hours after fever event was chosen.")
    
    # pick the data with fever event hours before and after
    temp_list = []
    fever_event_label = []
    original_data = []
    
    for i in range(len(data)):
        # find the fever event and the time have to be with in 30 days
        if data.iloc[i,3] == 1 and data.iloc[i,1] < 30*1440:
            temp_list.append(list(data.iloc[i-before*30:i+after*30,2]))
            # record the original data positions
            original_data.append(list(data.iloc[i-before*30:i+after*30,5]))
            # record the fever event label in (maskid, days after infusion) format
            fever_event_label.append((data.iloc[i,0],round(data.iloc[i,1]/1440,3)))

    # calculate the percentage of original data
    original_percent = []
    for original in original_data:
        percent = round(sum(original)/len(original),4)
        original_percent.append(percent)

    # filter out the data that has more than 70% original data
    qualified_temp_data = []
    qualified_temp_label = []
    qualified_temp_percent = []
    
    for i,percent in enumerate(original_percent):
        if percent >= original_percent_threshold:
            qualified_temp_percent.append(percent)
            qualified_temp_data.append(temp_list[i])
            qualified_temp_label.append(fever_event_label[i])

    result = pd.read_csv(fever_cause_data_path, sep=',')
    result = result[['MaskID','Time_DPI','Category']]

    qualified_fever_causes = result["Category"]

    # convert the fever causes to binary labels, infection:1, non-infection/unclear:0
    true_label = []
    for i in qualified_fever_causes:
        if i == "infection":
            true_label.append(1)
        else:
            true_label.append(0)
    true_label = np.array(true_label)
    
    # check the length of the data
    print(f'Check data length before clustering: {len(qualified_temp_data) == len(qualified_temp_label) == len(qualified_temp_percent) == len(qualified_fever_causes)}')

    temp_array = np.array(qualified_temp_data)
    
    print("Finish picking up and filtering the data!")
    print('-'*80)
    print(f'The data size for clustering is {temp_array.shape}.')

    return temp_array, qualified_temp_label, qualified_temp_percent, true_label, qualified_fever_causes

if __name__ == "__main__":
    
    temp_data_path = "../input/TempTraq_Dataset.csv"
    fever_start_data_path = "../input/TFeverStarts.csv"
    fever_cause_data_path = "../input/4-17-19_With_PHI_HCT_result_with_exact_time_clinical_categories.csv"
    cohort = "HCT"
    before = 4
    after = 4
    original_percent_threshold = 0.7

    temp_df = deal_with_missing_points(temp_data_path,cohort)
    temp_df = interpolation_smooth(temp_df,fever_start_data_path,cohort)
    temp_array, qualified_temp_label, qualified_temp_percent, true_label, qualified_fever_causes = filter_data(temp_df,before,after,original_percent_threshold,fever_cause_data_path)
    print("Done!")