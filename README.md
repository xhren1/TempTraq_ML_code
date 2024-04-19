# TempTraq_ML_code
## Input files
### Data files may contain patients' PHI data. If you want to have the data files, please contact the first author or the corresponding author.
### 1. TempTraq temperature data file (csv) - must have the following columns
      a. Cohort – study population (HCT or CART)
      b. MaskID – Masked ID
      c. Time_DPI – Time masked as Minutes Post-Infusion
      d. TTemp – Temperature recorded by TempTraq in Celsius
### 2. Fever start data file (csv) (only have fever start points) - must have the following columns
      a. Cohort – study population (HCT or CART)
      b. MaskID – Masked ID
      c. Time_DPI – Time masked as Minutes Post-Infusion
### 3. Fever cause data file (csv) (only 30 qualified fever events) - must have the following columns
      a.	MaskID – Masked ID
      c.	Time_DPI – Time masked as Minutes Post-Infusion (fever starting point)
      d.	Category – Fever causes determined by clinical diagnosis (Infection, Other Adverse Event, or unclear)

## Run the code

### Check dependency
```bash
pip install -r requirements.txt
```

### Setup parameters
In the main.py file, please fill up the path for the three files.

For parameter missing_points: 

    True: deal with missing points(for the first time run)

    False: not deal with missing points(for re-runs, if you have a full_ttemp.csv file in the current directory)

### Run the code
```bash
python main.py
```

## Output files
### 1. full_ttemp.csv - file filled missing points as nan
      a. Cohort – study population (HCT or CART)
      b. MaskID – Masked ID
      c. Time_DPI – Time masked as Minutes Post-Infusion
      d. TTemp – Temperature recorded by TempTraq in Celsius
      e. fever_start - 1:fever start points 0:not fever start point
      f. Orignial_data - 1:orignial data points 0:not orignial data points

### 2. result.csv - file with results
      a. MaskID – Masked ID
      b. Time_DPI – Time masked as Minutes Post-Infusion
      c. Category - Clinically determined fever causes
      d. cluster result - cluster result in number
### 3. plots in .svg format
      
