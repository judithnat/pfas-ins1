# 26 oct 2025
# modelled on D:\MyDocumentsD\ProjectsD\pfas-diss2\scripts\CIPdata\CIPprep3.py
# CIPcombine (one off done previously)
# CIP units4 - add cols to get consistent units
# diagnose problems: https://claude.ai/chat/aea9e181-f923-4368-96da-1028c6dee994


import os
os.environ['PROJ_LIB'] = r'C:/Users/judit/anaconda3/envs/geo-env/Library/share/proj'
import pyproj
import fiona
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
#import glob

run = "run3_inferCoordsGPKG"  # Change this to your desired run name

# units
units_csv_file_path = r"G:\Shared drives\P drive\1604-0 PFAS insurance\lqm work\qgis\data\pfas\CIP\CIPpfas2025.csv"  
#units_csv_file_path = r"C:\MyDocuments\experiment"
units_input_file = units_csv_file_path

base_units_output_folder = r"G:\Shared drives\P drive\1604-0 PFAS insurance\lqm work\qgis\data\pfas\CIP/"
units_output_folder = os.path.join(base_units_output_folder,run)
os.makedirs(units_output_folder, exist_ok=True)
units_output_csv = os.path.join(units_output_folder,"CIPpfas2025_units.csv")

# prep - for def prepare_data (makes tiers and log col)
input_file = units_output_csv  # Use the output from the units conversion as input for preparation  
print("debug - input_file", input_file)
base_output_folder = units_output_folder
output_folder = os.path.join(base_output_folder,run)
os.makedirs(output_folder, exist_ok=True)
output_csv = os.path.join(output_folder,"CIPpfas2025_prep.csv")

# infer coordinates
input_file_coords = output_csv
infer_coords_csv_file_path = output_csv  # Replace with your actual file path
infer_coords_output_folder = os.path.join(base_output_folder,run)
os.makedirs(infer_coords_output_folder, exist_ok=True)
infer_coords_output_csv = os.path.join(infer_coords_output_folder,"CIP_missingLocCol.csv")
infer_coords_file_path = infer_coords_csv_file_path # is needed

# manual coords (from CIPmanualAddcoords.py)
manualcoords_main_csv_path = infer_coords_output_csv
manualcoords_missing_coords_csv_path =  os.path.join(base_units_output_folder, "CIP_missingTreatmentPlants_populate.csv")
manualcoords_output_folder = os.path.join(base_units_output_folder, run)  # Use the same output folder as for units 
manualcoords_output_csv = os.path.join(manualcoords_output_folder, "CIP_with_manual_coords.csv")
manualcoords_summary_csv = os.path.join(manualcoords_output_folder, "CIP_manual_coords_summary.csv")
os.makedirs(manualcoords_output_folder, exist_ok=True)

# filter on ug/l
filtug_gpkg_file_path = manualcoords_output_csv
filtug_output_folder = os.path.join(base_units_output_folder, run)
os.makedirs(filtug_output_folder, exist_ok=True)
filtug_base_path_subset = os.path.join(filtug_output_folder) #doesnt make a new dir
os.makedirs(filtug_base_path_subset, exist_ok=True)
filtug_output_file_path = os.path.join(filtug_base_path_subset, 'CIPsubsetGDF2_ug_l.gpkg')

#standardize
standardize_input_gpkg = filtug_output_file_path  # Use the output from manual coords as input for standardization
standardize_output_folder = os.path.join(base_units_output_folder, run)
os.makedirs(standardize_output_folder, exist_ok=True)
standardize_output_gpkg = os.path.join(standardize_output_folder, "CIPgdf_standardized.gpkg")

#subset
subset_gpkg_file_path = standardize_output_gpkg  # Use the output from standardization as input for subsetting  
subset_output_folder = os.path.join(base_units_output_folder, run)
os.makedirs(subset_output_folder, exist_ok=True)
base_path_subset = os.path.join(subset_output_folder, "subsets/")
os.makedirs(base_path_subset, exist_ok=True)

#filter on selected pfas
selPFAS_input_gpkg = standardize_output_gpkg  # Input will be the standardized GPKG
selPFAS_output_folder = os.path.join(base_units_output_folder, run)
os.makedirs(selPFAS_output_folder, exist_ok=True)
selPFAS_output_gpkg_cip = os.path.join(selPFAS_output_folder, "CIP_selectedPFAS.gpkg")



print("============ F U N C T I O N S ==================")

print("======== units =========")
# scripts\CIPdata\CIPunits4.py

def sort_units(units_input_file):
    # Read the CSV file with explicit UTF-8 encoding
    df = pd.read_csv(units_input_file, encoding='utf-8')

    print("Original UnitsName values:")
    print(df['UnitsName'].head())

    # Create new columns for unit conversions
    df['µg/l'] = np.nan
    df['ng/l'] = np.nan
    df['mg/l'] = np.nan
    df['ug/kg'] = np.nan # Added new column for ug/kg

    # Conversion factors
    conversion_factors = {
        'µg/l': {
            'µg/l': 1,
            'ng/l': 1000,
            'mg/l': 0.001
        },
        'ng/l': {
            'µg/l': 0.001,
            'ng/l': 1,
            'mg/l': 0.000001
        },
        'mg/l': {
            'µg/l': 1000,
            'ng/l': 1000000,
            'mg/l': 1
        },
        # Added conversion factor for mg/kg to ug/kg
        'mg/kg': {
            'ug/kg': 1000
        }
    }



    # Process each row
    for index, row in df.iterrows():
        unit = row['UnitsName']
        value = row.get('SampleValue', 1)  # Assuming there's a value column, default to 1 if not
        
        # Copy original value if unit already matches
        for target_unit in ['µg/l', 'ng/l', 'mg/l', 'ug/kg']:
            if unit == target_unit:
                df.at[index, target_unit] = value
            elif unit in conversion_factors and target_unit in conversion_factors[unit]:
                df.at[index, target_unit] = value * conversion_factors[unit][target_unit]
            # special mg/kg 
            elif unit == 'mg/kg' and target_unit == 'ug/kg':
                df.at[index, target_unit] = value * 1000

    # For Excel compatibility, replace the micro symbol with 'u'
    df['UnitsName'] = df['UnitsName'].str.replace('µ', 'u')
    # Also rename the column with micro symbol
    df = df.rename(columns={'µg/l': 'ug/l'})

    # Save the updated dataframe with encoding that Excel can interpret correctly
    df.to_csv(units_output_csv, index=False, encoding='utf-8-sig')  # BOM helps Excel recognize UTF-8

    print("\nUnit conversion complete. Output saved to ", units_output_csv)
    return df
    
sort_units(units_input_file)

print("======== prepare =========")

def prepare_data(input_file):
    print("df S T A R T ")
    print("input_file", input_file)
    # Read the CSV file with explicit UTF-8 encoding
    df = pd.read_csv(input_file, encoding='utf-8')
    print(df.head())

    # functions
    def pfas_tiers(row):
        if row['ug/l'] < 0.01:
            return 'Tier 1'
        elif row['ug/l'] < 0.1 and row['ug/l'] >= 0.01:
            return 'Tier 2'
        elif row['ug/l'] >= 0.1:
            return 'Tier 3'
        else:
            return 'Error'
        
    def project_tiers(row):
        if row['ug/l'] < 0.001:
            return 'P-Tier 1d'
        elif row['ug/l'] < 0.0025 and row['ug/l'] >= 0.001:
            return 'P-Tier 1c'
        elif row['ug/l'] < 0.005 and row['ug/l'] >= 0.0025:
            return 'P-Tier 1b'
        elif row['ug/l'] < 0.01 and row['ug/l'] >= 0.005:
            return 'P-Tier 1a'
        elif row['ug/l'] < 0.1 and row['ug/l'] >= 0.01:
            return 'P-Tier 2'
        elif row['ug/l'] >= 0.1:
            return 'P-Tier 3'
        else:
            return 'Error'

    def eqs_tiers(row):
        if row['ug/l'] < 0.00013:
            return 'EQS(PFOS) below AA (est)'
        elif row['ug/l'] < 0.00065 and row['ug/l'] >= 0.00013:
            return 'EQS(PFOS) below AA (fresh)'
        elif row['ug/l'] >= 0.00065:
            return 'EQS(PFOS) above AA (fresh)'
        else:
            return 'Error'


    df['Tier'] = df.apply(pfas_tiers, axis=1)
    df['Project_Tier'] = df.apply(project_tiers, axis=1)
    df['EQS_Tier'] = df.apply(eqs_tiers, axis=1)
    # https://claude.ai/chat/c05bcabd-18f9-4c1a-8203-bfa9b2d13fc3 - could use log+1 transformation, but no zeros so OK (some controversys with l+1)
    # same claude np.log (ln) versus np.log10 
    df['result_log_ug/l'] = np.log(df['ug/l'])
    df['result'] = df['ug/l']

    #  Using apply with row function
    def extract_year_from_row(row):
        # Get date string 
        date_string = row['SampleDateTime']
        
        # Handle NaN, None or other non-string values
        if pd.isna(date_string) or not isinstance(date_string, str):
            print(f"Non-string date value found: {date_string}")
            return 9999
        
        try:
            # For DD/MM/YYYY format, split by '/' and take the last element before the time
            date_part = date_string.split(' ')[0]  # Get just the date part
            year = date_part.split('/')[2]  # Get the year
            
            if year == '2457':
                print('year', year)
                year = '9991'
                print('year', year)
            
            # Handle truncated years if needed
            if len(year) == 2:
                return 2000 + int(year)  # Adjust this logic if needed for your data
            else:
                return int(year)
                
        except (IndexError, ValueError) as e:
            # Print problematic value for debugging
            print(f"Problem extracting year from: {date_string}")
            return None  # Or some fallback value

    # Call it like this
    df['Year'] = df.apply(extract_year_from_row, axis=1)

    print("df OUTPUT")
    print(df.head())
    print(df.tail())

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print("Saved to", output_csv)
    return df

prepare_data(input_file)
    
def all_infer_missing_coords(infer_coords_file_path, infer_coords_output_csv):
    df = pd.read_csv(infer_coords_file_path)
    print('columns: ', df.columns)
    print(df['Latitude'].isna().sum())
    print(df['Longitude'].isna().sum())
    print('orig',df.head(2))

    # Define a function to check if a value is blank, missing, or zero      
    def is_blank_or_zero(value):
        if pd.isna(value):  # Check for NaN values
            return True
        if value == 0 or value == "0" or value == "":
            return True
        return False

    def determine_location_status(row):
        if is_blank_or_zero(row['Latitude']) and is_blank_or_zero(row['Longitude']):
            return 'no coords'
        elif is_blank_or_zero(row['Latitude']) or is_blank_or_zero(row['Longitude']):
            return 'partial coords'
        else:
            return 'coords'
        
    df['location_status'] = 'blank'
    df['location_status'] = df.apply(determine_location_status, axis=1)
    print('location_status',df.head(2))

    # infer missing coordinates from 'TreatmentPlant' and 'SampleLocationName'
    df['inferred_coordinates'] = 'blank'
    print('inferred_coordinates empty col',df.head(2))

    # Add function to infer missing coordinates
    def infer_missing_coordinates(df):
        # Create dictionary to map treatment plants to coordinates
        treatment_plant_coords = {}
        
        # Create dictionary for TreatmentPlant + SampleLocationName combinations
        # HIGHLIGHT: Using combined key of plant and location
        plant_location_coords = {}
        
        # Identify records with valid coordinates
        valid_coords_df = df[df['location_status'] == 'coords']
        
        # Build mapping from treatment plants to their coordinates
        for idx, row in valid_coords_df.iterrows():
            plant = row['TreatmentPlant']
            location = row['SampleLocationName']
            
            if not pd.isna(plant) and plant != "":
                # Store coordinates for the treatment plant
                if plant not in treatment_plant_coords:
                    treatment_plant_coords[plant] = (row['Latitude'], row['Longitude'])
                
                # HIGHLIGHT: Store coordinates for plant+location combination
                combined_key = f"{plant}_{location}"
                if not pd.isna(location) and location != "":
                    plant_location_coords[combined_key] = (row['Latitude'], row['Longitude'])
        
        # Apply mappings to fill in missing coordinates
        for idx, row in df.iterrows():
            if row['location_status'] in ['no coords', 'partial coords']:
                plant = row['TreatmentPlant']
                location = row['SampleLocationName']
                
                # HIGHLIGHT: First try plant+location combination
                if not pd.isna(plant) and plant != "" and not pd.isna(location) and location != "":
                    combined_key = f"{plant}_{location}"
                    if combined_key in plant_location_coords:
                        lat, lon = plant_location_coords[combined_key]
                        
                        if row['location_status'] == 'no coords':
                            df.at[idx, 'Latitude'] = lat
                            df.at[idx, 'Longitude'] = lon
                            df.at[idx, 'inferred_coordinates'] = f'inferred from {plant} + {location}'
                        else:  # partial coords
                            if is_blank_or_zero(row['Latitude']):
                                df.at[idx, 'Latitude'] = lat
                            if is_blank_or_zero(row['Longitude']):
                                df.at[idx, 'Longitude'] = lon
                            df.at[idx, 'inferred_coordinates'] = f'partially inferred from {plant} + {location}'
                    
                    # HIGHLIGHT: If no match for combination, try just the treatment plant
                    elif plant in treatment_plant_coords:
                        lat, lon = treatment_plant_coords[plant]
                        
                        if row['location_status'] == 'no coords':
                            df.at[idx, 'Latitude'] = lat
                            df.at[idx, 'Longitude'] = lon
                            df.at[idx, 'inferred_coordinates'] = f'inferred from {plant} only'
                        else:  # partial coords
                            if is_blank_or_zero(row['Latitude']):
                                df.at[idx, 'Latitude'] = lat
                            if is_blank_or_zero(row['Longitude']):
                                df.at[idx, 'Longitude'] = lon
                            df.at[idx, 'inferred_coordinates'] = f'partially inferred from {plant} only'
                    else:
                        # HIGHLIGHT: Cannot infer if treatment plant has no known coordinates
                        df.at[idx, 'inferred_coordinates'] = 'unable to infer (no plant match)'
                
                # If only treatment plant is available
                elif not pd.isna(plant) and plant != "" and plant in treatment_plant_coords:
                    lat, lon = treatment_plant_coords[plant]
                    
                    if row['location_status'] == 'no coords':
                        df.at[idx, 'Latitude'] = lat
                        df.at[idx, 'Longitude'] = lon
                        df.at[idx, 'inferred_coordinates'] = f'inferred from {plant} only'
                    else:  # partial coords
                        if is_blank_or_zero(row['Latitude']):
                            df.at[idx, 'Latitude'] = lat
                        if is_blank_or_zero(row['Longitude']):
                            df.at[idx, 'Longitude'] = lon
                        df.at[idx, 'inferred_coordinates'] = f'partially inferred from {plant} only'
                else:
                    # HIGHLIGHT: Cannot infer if missing or unmatched treatment plant
                    df.at[idx, 'inferred_coordinates'] = 'unable to infer (missing plant info)'
            else:
                # Already has coordinates
                df.at[idx, 'inferred_coordinates'] = 'original coordinates'
        
        # HIGHLIGHT: Create a new column 'location_status_2' instead of updating the original
        df['location_status_2'] = df.apply(determine_location_status, axis=1)
        
        # return df
        # Create dictionaries to map treatment plants and sample locations to coordinates
        treatment_plant_coords = {}
        sample_location_coords = {}
        
        # Identify records with valid coordinates
        valid_coords_df = df[df['location_status'] == 'coords']
        
        # Build mapping from treatment plants to their coordinates
        for idx, row in valid_coords_df.iterrows():
            plant = row['TreatmentPlant']
            if not pd.isna(plant) and plant != "":
                if plant not in treatment_plant_coords:
                    treatment_plant_coords[plant] = (row['Latitude'], row['Longitude'])
        
        # Build mapping from sample locations to their coordinates
        for idx, row in valid_coords_df.iterrows():
            location = row['SampleLocationName']
            if not pd.isna(location) and location != "":
                if location not in sample_location_coords:
                    sample_location_coords[location] = (row['Latitude'], row['Longitude'])
        
        # Apply mappings to fill in missing coordinates
        for idx, row in df.iterrows():
            if row['location_status'] == 'no coords':
                # Try to infer from treatment plant
                plant = row['TreatmentPlant']
                if not pd.isna(plant) and plant != "" and plant in treatment_plant_coords:
                    lat, lon = treatment_plant_coords[plant]
                    df.at[idx, 'Latitude'] = lat
                    df.at[idx, 'Longitude'] = lon
                    df.at[idx, 'inferred_coordinates'] = 'inferred from treatment plant'
                # If not found, try to infer from sample location name
                elif not pd.isna(row['SampleLocationName']) and row['SampleLocationName'] != "" and row['SampleLocationName'] in sample_location_coords:
                    lat, lon = sample_location_coords[row['SampleLocationName']]
                    df.at[idx, 'Latitude'] = lat
                    df.at[idx, 'Longitude'] = lon
                    df.at[idx, 'inferred_coordinates'] = 'inferred from sample location'
                else:
                    df.at[idx, 'inferred_coordinates'] = 'unable to infer'
            elif row['location_status'] == 'partial coords':
                # Handle partial coordinates
                plant = row['TreatmentPlant']
                if not pd.isna(plant) and plant != "" and plant in treatment_plant_coords:
                    lat, lon = treatment_plant_coords[plant]
                    if is_blank_or_zero(row['Latitude']):
                        df.at[idx, 'Latitude'] = lat
                        df.at[idx, 'inferred_coordinates'] = 'latitude inferred from plant'
                    if is_blank_or_zero(row['Longitude']):
                        df.at[idx, 'Longitude'] = lon
                        df.at[idx, 'inferred_coordinates'] = 'longitude inferred from plant'
                elif not pd.isna(row['SampleLocationName']) and row['SampleLocationName'] != "" and row['SampleLocationName'] in sample_location_coords:
                    lat, lon = sample_location_coords[row['SampleLocationName']]
                    if is_blank_or_zero(row['Latitude']):
                        df.at[idx, 'Latitude'] = lat
                        df.at[idx, 'inferred_coordinates'] = 'latitude inferred from location'
                    if is_blank_or_zero(row['Longitude']):
                        df.at[idx, 'Longitude'] = lon
                        df.at[idx, 'inferred_coordinates'] = 'longitude inferred from location'
                else:
                    df.at[idx, 'inferred_coordinates'] = 'unable to infer missing coordinate'
            else:
                # Already has coordinates
                df.at[idx, 'inferred_coordinates'] = 'original coordinates'
        
        # Update location status after inference
        df['location_status_2'] = 'blank'
        df['location_status_2'] = df.apply(determine_location_status, axis=1)
        return df


    # Apply the inference function to the dataframe
    df = infer_missing_coordinates(df)

    # Print summary of inference results

    print("Number of CIP records:", len(df))
    print("Original location status:")
    print(df['location_status'].value_counts())
    print("New location status after inference:")
    print(df['location_status_2'].value_counts())
    # print("Inference summary:")
    # print(df['inferred_coordinates'].value_counts())

    # treatment plants with no coords
    missing_treatment_plants = df[df['location_status_2'] == 'no coords']['TreatmentPlant'].unique()
    print("Treatment plants with no coordinates after inference:")
    print(missing_treatment_plants)
    print(type(missing_treatment_plants))
    # Convert the NumPy array to a DataFrame
    missing_treatment_plants_df = pd.DataFrame({'TreatmentPlant': missing_treatment_plants})
    print('LOOK LOOK',missing_treatment_plants_df.head(2))
    missing_treatment_plants_df['Latitude'] = 0
    missing_treatment_plants_df['Longitude'] = 0
    missing_treatment_plants_df['Source_quality'] = 'blank'
    missing_treatment_plants_df['Source_quality'] = 'note'
    missing_treatment_plants_df['Source'] = 'blank'
    missing_treatment_plants_df['Link'] = 'blank'

    #save to csv
    missing_treatment_plants_csv = os.path.join(infer_coords_output_folder, "CIP_missingTreatmentPlants.csv")
    missing_treatment_plants_df.to_csv(missing_treatment_plants_csv, index=False)  
    print("Treatment plants with no coordinates saved to", missing_treatment_plants_csv)

    # Create a dataframe of the summary information
    # Get the counts for original and new location status
    original_counts = df['location_status'].value_counts().reset_index()
    original_counts.columns = ['status', 'count_original']

    new_counts = df['location_status_2'].value_counts().reset_index()
    new_counts.columns = ['status', 'count_after_inference']

    # Merge the two count dataframes
    summary_df = pd.merge(original_counts, new_counts, on='status', how='outer').fillna(0)

    # Calculate the difference (how many were fixed)
    summary_df['difference'] = summary_df['count_after_inference'] - summary_df['count_original']

    # Add a total row
    total_row = pd.DataFrame({
        'status': ['Total'],
        'count_original': [len(df)],
        'count_after_inference': [len(df)],
        'difference': [0]
    })
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)

    # Convert count columns to integers
    summary_df['count_original'] = summary_df['count_original'].astype(int)
    summary_df['count_after_inference'] = summary_df['count_after_inference'].astype(int)
    summary_df['difference'] = summary_df['difference'].astype(int)

    print("\nSummary of Coordinate Inference Results:")
    print(summary_df.to_string(index=False))

    # save as gpkg change
    # df.to_csv(infer_coords_output_csv, index=False)
    # print("Saved to", infer_coords_output_csv)   
    
        # Convert to GeoDataFrame
    df_with_coords = df.dropna(subset=['Latitude', 'Longitude'])
    geometry = [Point(xy) for xy in zip(df_with_coords['Longitude'], df_with_coords['Latitude'])]
    gdf = gpd.GeoDataFrame(df_with_coords, geometry=geometry, crs='EPSG:4326')
    
    # Save as both GPKG and CSV
    infer_coords_output_gpkg = infer_coords_output_csv.replace('.csv', '.gpkg')
    gdf.to_file(infer_coords_output_gpkg, driver='GPKG')
    gdf.to_csv(infer_coords_output_csv, index=False)
    print("Saved to", infer_coords_output_gpkg)
    print("Saved to", infer_coords_output_csv)

    # Save the summary dataframe to a separate CSV

    # Save the summary dataframe to a separate CSV
    summary_csv = os.path.join(infer_coords_output_folder, "CIP_coordinate_inference_summary1.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("Main data saved to", infer_coords_output_csv)
    print("Summary statistics saved to", summary_csv)
    return df

# Run the function to infer missing coordinates
all_infer_missing_coords(infer_coords_file_path, infer_coords_output_csv)

print("======== manual coords =========")

def all_manual_add_coords(manualcoords_main_csv_path, manualcoords_missing_coords_csv_path, manualcoords_output_folder):
        # Define a function to check if a value is blank, missing, or zero      
    def is_blank_or_zero(value):
        if pd.isna(value):  # Check for NaN values
            return True
        if value == 0 or value == "0" or value == "":
            return True
        return False

    # Define function to determine location status
    def determine_location_status(row):
        if is_blank_or_zero(row['Latitude']) and is_blank_or_zero(row['Longitude']):
            return 'no coords'
        elif is_blank_or_zero(row['Latitude']) or is_blank_or_zero(row['Longitude']):
            return 'partial coords'
        else:
            return 'coords'

    # Read in the main dataframe
    print("Reading main CSV file...")
    df = pd.read_csv(manualcoords_main_csv_path)
    print(f"Loaded {len(df)} records from main CSV")

    # Read in the missing coordinates dataframe
    print("Reading missing coordinates CSV file...")
    missing_coords_df = pd.read_csv(manualcoords_missing_coords_csv_path)
    print(f"Loaded {len(missing_coords_df)} treatment plants with manually added coordinates")

    # Display a sample of the missing coordinates data
    print("\nSample of manually added coordinates:")
    print(missing_coords_df.head())

    # Initialize the manual_coordinates column
    df['manual_coordinates'] = 'none'

    # Create a dictionary for faster lookups
    coords_dict = {}
    for idx, row in missing_coords_df.iterrows():
        coords_dict[row['TreatmentPlant']] = {
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Source': row['Source']
        }

    # Count how many records we update
    update_count = 0

    # Update coordinates in the main dataframe
    print("\nUpdating coordinates...")
    for idx, row in df.iterrows():
        plant = row['TreatmentPlant']
        
        # Check if this treatment plant has manually added coordinates
        if plant in coords_dict:
            # Only update if coordinates are still missing
            if row['location_status_2'] == 'no coords' or row['location_status_2'] == 'partial coords':
                # Update latitude if it's missing
                if is_blank_or_zero(row['Latitude']):
                    df.at[idx, 'Latitude'] = coords_dict[plant]['Latitude']
                
                # Update longitude if it's missing
                if is_blank_or_zero(row['Longitude']):
                    df.at[idx, 'Longitude'] = coords_dict[plant]['Longitude']
                
                # Set the manual_coordinates field to show the source
                df.at[idx, 'manual_coordinates'] = coords_dict[plant]['Source']
                
                update_count += 1

    # Create location_status_3 column
    df['location_status_3'] = df.apply(determine_location_status, axis=1)

    print(f"Updated coordinates for {update_count} records")

    # Print summary of location status before and after updates
    print("\nLocation status before manual updates:")
    print(df['location_status_2'].value_counts())
    print("\nLocation status after manual updates:")
    print(df['location_status_3'].value_counts())

    # Create a summary dataframe
    status_before = df['location_status_2'].value_counts().reset_index()
    status_before.columns = ['status', 'count_before']

    status_after = df['location_status_3'].value_counts().reset_index()
    status_after.columns = ['status', 'count_after']

    # Merge the summary dataframes
    summary_df = pd.merge(status_before, status_after, on='status', how='outer').fillna(0)

    # Calculate differences
    summary_df['difference'] = summary_df['count_after'] - summary_df['count_before']

    # Add a total row
    total_row = pd.DataFrame({
        'status': ['Total'],
        'count_before': [len(df)],
        'count_after': [len(df)],
        'difference': [0]
    })
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)

    # Convert count columns to integers
    summary_df['count_before'] = summary_df['count_before'].astype(int)
    summary_df['count_after'] = summary_df['count_after'].astype(int)
    summary_df['difference'] = summary_df['difference'].astype(int)

    print("\nSummary of manual coordinate updates:")
    print(summary_df.to_string(index=False))

    # Save the updated DataFrame to a new CSV file
    df.to_csv(manualcoords_output_csv, index=False)

    # Save the summary dataframe to a separate CSV
    summary_df.to_csv(manualcoords_summary_csv, index=False)

    print("\nMain data saved to", output_csv)
    print("Summary statistics saved to", manualcoords_summary_csv)
    
    return df

# Run the function to manually add coordinates
all_manual_add_coords(manualcoords_main_csv_path, manualcoords_missing_coords_csv_path, manualcoords_output_folder)


print("======== filter on ug/l =========")
def filter_cip_ug_l(filtug_gpkg_file_path):
    # Read CSV file as pandas DataFrame first
    df = pd.read_csv(filtug_gpkg_file_path)
    print('df loaded', df.head(2))
    
    # Convert to GeoDataFrame by creating Point geometries from Lat/Lon
    # Filter out rows with missing coordinates first
    df_with_coords = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Create Point geometries
    geometry = [Point(xy) for xy in zip(df_with_coords['Longitude'], df_with_coords['Latitude'])]
    cip_pfas_gdf = gpd.GeoDataFrame(df_with_coords, geometry=geometry)
    
    # Set coordinate reference system (assuming WGS84)
    cip_pfas_gdf.crs = 'EPSG:4326'
    
    print('cip_pfas_gdf', cip_pfas_gdf.head(2))

    column_name = 'ug/l'  # Specify the column name to filter by
    subset_with_values = cip_pfas_gdf[cip_pfas_gdf[column_name].notna() & (cip_pfas_gdf[column_name] != '')]
    subset_with_values = subset_with_values.reset_index(drop=True)
    print('subset_with_values', subset_with_values.head(2))
    
    # Save the subset to a new GeoDataFrame     
    
    subset_with_values.to_file(filtug_output_file_path, driver='GPKG') 
    print(f"Subset saved to {filtug_output_file_path}")
    
    # Save the subset to a CSV file 
    output_csv_path = os.path.splitext(filtug_output_file_path)[0] + '.csv'
    subset_with_values.to_csv(output_csv_path, index=False, encoding='utf-8')   
    print(f"Subset saved to {output_csv_path}")

    # Record changes made to the original dataset
    # Calculate row counts**
    original_rows = len(cip_pfas_gdf)
    subset_rows = len(subset_with_values)
    removed_rows = original_rows - subset_rows

    # Create a summary DataFrame with row count information**
    row_count_summary = pd.DataFrame({
        'Dataset': ['Original CIP data', 'Filtered on ug/l', 'Number of rows removed'],
        'Row_Count': [original_rows, subset_rows, removed_rows],
        'Filter_Applied': ['None', f'Non-null/non-empty values in {column_name} column', 'NA']
    })

    # Export row count summary to CSV**
    row_count_csv_path = os.path.join(filtug_base_path_subset, 'CIP_row_count_summary.csv')
    row_count_summary.to_csv(row_count_csv_path, index=False, encoding='utf-8')
    print(f"Row count summary saved to {row_count_csv_path}")

    # print original number of rows
    print(f"Original dataset: {len(cip_pfas_gdf)} rows")    
    # print number of rows in subset
    print(f"Subset dataset: {len(subset_with_values)} rows")    
    
    return subset_with_values
filter_cip_ug_l(filtug_gpkg_file_path)

print("======== standardize =========")

def standardize_pfas_combine_acid_salt(standardize_input_gpkg, column_name='NameDeterminandName'):
    """
    Standardize PFAS names to abbreviations, combining acids and their corresponding salts.
    
    Parameters:
    gdf: GeoDataFrame containing PFAS data
    column_name: Name of column containing PFAS compound names
    
    Returns:
    GeoDataFrame with new 'pfas_standardized' column
    """
    gdf = gpd.read_file(standardize_input_gpkg) #inserted here to read the input GPKG file
    # Mapping dictionary - combining acids and salts under same abbreviation
    pfas_mapping = {
        # Carboxylic acids
        'Perfluoro Hexanoic Acid': 'PFHxA',
        'Perfluorohexanoic acid (PFHXA)': 'PFHxA',
        'Perfluoro Pentanoic Acid': 'PFPeA',
        'Perfluoropentanoic acid (PFPeA)': 'PFPeA',
        'Perfluoro Heptanoic Acid': 'PFHpA',
        'Perfluoroheptanoic acid (PFHpA)': 'PFHpA',
        'Perfluoro Nonanoic Acid': 'PFNA',
        'Perfluorononanoic acid (PFNA)': 'PFNA',
        'Perfluoro Decanoic Acid': 'PFDA',
        'Perfluorodecanoic acid (PFDA)': 'PFDA',
        'Perfluoro Undecanoic Acid': 'PFUnDA',
        'Perfluoroundecanoic acid (PFUnDA)': 'PFUnDA',
        'Perfluoro Dodecanoic Acid': 'PFDoDA',
        'Perfluorododecanoic acid (PFDoDA)': 'PFDoDA',
        'Perfluorooctanoic acid (PFOA)': 'PFOA',
        'PFOA': 'PFOA',
        'Perfluorotetradecanoic acid': 'PFTeDA',
        'Perfluorotridecanoic acid': 'PFTrDA',
        'Heptafluorobutyric acid (HFBA)': 'HFBA',
        
        # Sulfonic acids and sulfonates (combined under same abbreviation)
        'Perfluorohexanesulfonic acid (PFHxS)': 'PFHxS',
        'Perfluorohexane sulfonate': 'PFHxS',
        'Perfluorooctane sulphonate (PFOS)': 'PFOS',
        'PFOS': 'PFOS',
        'Perfluorobutane sulphonate': 'PFBS',
        'Perfluorobutane sulfonate': 'PFBS',
        'Perfluoropentane sulfonate': 'PFPeS',
        '3,3,4,4,5,5,6,6,7,7,8,8,8-Tridecafluorooctanesulfonic acid': '6:2 FTSA',
        
        # Sulfonamides
        'Perfluorooctanesulfonamide (PFOSA)': 'PFOSA',
        
        # Complex compounds (no standard abbreviations - keep original)
        'Carboxymethyldimethyl-3-[[(3,3,4,4,5,5,6,6,7,7,8,8,8-tridecafluorooctyl)sulfonyl]amino]propylammonium hydroxide': 'Carboxymethyldimethyl-3-[[(3,3,4,4,5,5,6,6,7,7,8,8,8-tridecafluorooctyl)sulfonyl]amino]propylammonium hydroxide',
        '1H,1H,2H,2H-Perfluoro-1-decanol': '1H,1H,2H,2H-Perfluoro-1-decanol',
        '10:2 FTOH Sulfate Potassium Salt': '10:2 FTOH Sulfate Potassium Salt'
    }
    
    # Create new column with standardized names
    gdf = gdf.copy()
    gdf['pfas_standardized'] = gdf[column_name].map(pfas_mapping).fillna(gdf[column_name])
    
    
    gdf.to_file(standardize_output_gpkg, layer='CIP_standardized', driver='GPKG')
    print(f"Standardized GeoDataFrame saved to {standardize_output_gpkg}")  
    # save to csv
    output_csv = os.path.splitext(standardize_output_gpkg)[0] + '.csv'      
    print(f"Saving standardized data to CSV: {output_csv}")
    gdf.to_csv(output_csv, index=False)
    print('saved to csv', output_csv)
    return gdf

gdf_standardize = standardize_pfas_combine_acid_salt(standardize_input_gpkg, column_name='NameDeterminandName')
print("======== subset CIP pfas =========  ")

def subset_CIP_pfas(subset_gpkg_file_path, base_path_subset):
    cip_pfas_gdf = gpd.read_file(subset_gpkg_file_path)
    print('cip_pfas_gdf', cip_pfas_gdf.head(2))

    unique_sampleLocation = cip_pfas_gdf['SampleLocationName'].unique()
    print('unique_sampleLocation', unique_sampleLocation)

    CIP_upgradient = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'Up Gradient']
    CIP_upgradient = CIP_upgradient.reset_index(drop=True)
    CIP_downgradient = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'Down Gradient']
    CIP_downgradient = CIP_downgradient.reset_index(drop=True)
    CIP_treatmentEffluent = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'Treatment Effluent']
    CIP_treatmentEffluent = CIP_treatmentEffluent.reset_index(drop=True)
    CIP_spread_to_land = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'Spread to Land']
    CIP_spread_to_land = CIP_spread_to_land.reset_index(drop=True)
    CIP_treatmentInfluent = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'Treatment Influent']
    CIP_treatmentInfluent = CIP_treatmentInfluent.reset_index(drop=True)
    CIP_river_upstream = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'River Upstream']
    CIP_river_upstream = CIP_river_upstream.reset_index(drop=True)   
    CIP_river_downstream = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'River Downstream']
    CIP_river_downstream = CIP_river_downstream.reset_index(drop=True)  
    CIP_on_stw_site = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'On STW Site']
    CIP_on_stw_site = CIP_on_stw_site.reset_index(drop=True)    
    CIP_sewer_catchment = cip_pfas_gdf[cip_pfas_gdf['SampleLocationName'] == 'Sewer Catchment']
    CIP_sewer_catchment = CIP_sewer_catchment.reset_index(drop=True)

    #dict of df
    dataframes_dict = {
        'CIP_upgradient': CIP_upgradient,
        'CIP_downgradient': CIP_downgradient,
        'CIP_treatmentEffluent': CIP_treatmentEffluent,
        'CIP_spread_to_land': CIP_spread_to_land,
        'CIP_treatmentInfluent': CIP_treatmentInfluent,
        'CIP_river_upstream': CIP_river_upstream,
        'CIP_river_downstream': CIP_river_downstream,
        'CIP_on_stw_site': CIP_on_stw_site,
        'CIP_sewer_catchment': CIP_sewer_catchment
    }

    def save_dataframes(dataframes_dict, base_path, create_path=True, save_formats=None):
        #print(os.path.exists("D:/MyDocumentsD/ProjectsD/pfas-diss/data/EA_25/pfas_gdf1/subsets/"))

        # Default formats if none specified
        if save_formats is None:
            save_formats = ['gpkg', 'csv']
        
        # Create the directory if it doesn't exist and create_path is True
        if create_path:
            print(f"Checking if path exists: {base_path}")
            if not os.path.exists(base_path):
                print(f"Creating directory: {base_path}")
                os.makedirs(base_path, exist_ok=True)
        
        saved_files = {}
        
        # Loop through each dataframe and save in requested formats
        for df_name, df in dataframes_dict.items():
            saved_files[df_name] = []
            
            # Save as GeoPackage if requested
            if 'gpkg' in save_formats:
                gpkg_path = os.path.join(base_path, f"{df_name}.gpkg")
                try:
                    df.to_file(gpkg_path, driver='GPKG', layer=df_name)
                    print(f"{df_name} saved to: {gpkg_path}")
                    saved_files[df_name].append(gpkg_path)
                except Exception as e:
                    print(f"Error saving {df_name} as GPKG: {e}")
            
            # Save as CSV if requested
            if 'csv' in save_formats:
                csv_path = os.path.join(base_path, f"{df_name}.csv")
                try:
                    df.to_csv(csv_path)
                    print(f"{df_name} saved to: {csv_path}")
                    saved_files[df_name].append(csv_path)
                except Exception as e:
                    print(f"Error saving {df_name} as CSV: {e}")
        
        return saved_files

    saved_files = save_dataframes(dataframes_dict, base_path_subset)
    
subset_CIP_pfas(subset_gpkg_file_path, base_path_subset)

print("======== select PFAS =========")
pfas_compounds = [
    'PFOS', 'PFHxA', 'PFPeA', 'PFDoDA', 'PFHpA', 
    'PFTeDA', 'PFUnDA', 'PFDA', 'PFNA',   'PFOA',
]

def filter_pfas(input_gpkg, pfas_compounds,  output_gpkg):
    """
    Filters the GeoDataFrame for rows containing any of the specified PFAS compounds.
    
    Parameters:
    gdf (GeoDataFrame): The input GeoDataFrame.
    pfas_compounds (list): List of PFAS compounds to search for.
    
    Returns:
    GeoDataFrame: Filtered GeoDataFrame containing only rows with specified PFAS compounds.
    """
    gdf = gpd.read_file(input_gpkg)
    gdf['pfas_standardized'] = gdf['pfas_standardized']
    
    # Create a boolean mask to filter rows containing any of the PFAS compounds
    mask = gdf['pfas_standardized'].str.contains('|'.join(pfas_compounds), case=False, na=False)
    
    # Filter the GeoDataFrame
    filtered_gdf = gdf[mask]
    
    # Save the filtered data to a new GeoPackage and csv file
    filtered_gdf.to_file(output_gpkg, driver='GPKG')    
    filtered_gdf.to_csv(output_gpkg.replace('.gpkg', '.csv'), index=False)
    # Print the number of rows in the original and filtered GeoDataFrames       
    print(f"Filtering {len(gdf)} rows for PFAS compounds: {', '.join(pfas_compounds)}")
    print(f"Filtered {len(filtered_gdf)} rows containing specified PFAS compounds.")        


    print(f"Original dataset: {len(gdf)} rows")
    print(f"Filtered dataset: {len(filtered_gdf)} rows")
    print(f"Filtered data saved to: {output_gpkg}")
    print('original unique pfas', gdf['pfas_standardized'].unique())
    print('filtered unique pfas', filtered_gdf['pfas_standardized'].unique())
    
    return filtered_gdf
# Filter the groundwater and surface water GeoDataFrames        
filtered_gdf_cip = filter_pfas(selPFAS_input_gpkg, pfas_compounds, selPFAS_output_gpkg_cip)

#check filtered data
print('columns')
print(filtered_gdf_cip.columns)  # Display the columns of the filtered GeoDataFrame
print('unique')
print(filtered_gdf_cip['pfas_standardized'].unique())  # Display unique PFAS compounds in the filtered data
# Print the first few rows of the filtered GeoDataFrames         
print("Filtered groundwater data:")
print(filtered_gdf_cip.head())    

print('saved to', selPFAS_output_gpkg_cip)

print( "======== end of prep cip =========")