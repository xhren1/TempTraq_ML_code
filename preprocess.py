import pandas as pd
import numpy as np
from tqdm import tqdm

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

    return final_df

if __name__ == "__main__":
    temp_data_path = "../input/TempTraq_Dataset.csv"
    fever_start_data_path = "../input/TFeverStarts.csv"
    cohort = "HCT"
    temp_df = deal_with_missing_points(temp_data_path,fever_start_data_path,cohort)
    print("Done!")