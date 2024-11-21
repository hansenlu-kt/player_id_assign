#import mlbstatsapi
import pandas as pd
import re 
import streamlit as st
import requests
import time
from unidecode import unidecode

def search_existing_players(df, player_id_db):
        matches = []
        for index, row in df.iterrows():
            # Find matches in player_id_db
            match = player_id_db[
                (player_id_db['FirstName'].str.lower() == row['FirstName'].lower()) &
                (player_id_db['LastName'].str.lower() == row['LastName'].lower())
            ]
            if not match.empty:
                matches.append({
                    'FirstName': row['FirstName'],
                    'LastName': row['LastName'],
                    'UniformNumber':  row['UniformNumber'],
                    'Weight': row['Weight'],
                    'Found_In_DB': 'Yes',
                    'DB_ID': match['ID'].values[0] if not match.empty else None
                })
            else:
                matches.append({
                    'FirstName': row['FirstName'],
                    'LastName': row['LastName'],
                    'UniformNumber':  row['UniformNumber'],
                    'Weight': row['Weight'],
                    'Found_In_DB': 'No',
                    'DB_ID': None
                })
        return pd.DataFrame(matches)
# Sequential Query, Slow, Do not use, for testing purposes
@st.cache_data
def player_check(df,player_id_db, endpoint,leagues_endpoint):
    start_time = time.time()
    df['full_name'] = df['FirstName'] + ' ' + df['LastName']
 
    # DF to stage and cross-reference 
    roster_df = pd.DataFrame(columns=['MLBAMID', 'DB_ID', 'FirstName', 'LastName', 'UniformNumber','FirstName_MLB','LastName_MLB','useFirstName_MLB','useLastName_MLB','MiddleName_MLB','DOB_MLB'])

    # Sequential Query, similar speed as Batch Query for low volume roster checks
    for index, row in df.iterrows():
        # Query existing player DB
        existing_players = player_id_db[
            (player_id_db['FirstName'].str.lower() == row['FirstName'].lower()) &
            (player_id_db['LastName'].str.lower() == row['LastName'].lower())
        ]
        # Query from mlbstatsapi
        url = f"{base_url}{endpoint}{row['full_name']}{leagues_endpoint}"
        #st.write(url)
        response = requests.get(url)
        data = response.json()
        #st.write(data)
        if data and 'people' in data and data['people']:
            roster_df.loc[index, 'FirstName'] =  row['FirstName']
            roster_df.loc[index, 'LastName'] =  row['LastName']
            roster_df.loc[index, 'UniformNumber'] =  row['UniformNumber']
            # Conditions if BAM_ID or Database ID exists
            roster_df.loc[index, 'MLBAMID'] = data['people'][0]['id']
            roster_df.loc[index, 'FirstName_MLB'] = data['people'][0]['firstName']
            roster_df.loc[index, 'useFirstName_MLB'] = data['people'][0]['useName']
            roster_df.loc[index, 'useLastName_MLB'] = data['people'][0]['useLastName']
            roster_df.loc[index, 'LastName_MLB'] = data['people'][0]['lastName']
            try:
                roster_df.loc[index, 'MiddleName'] = data['people'][0]['middleName']
            except:
                pass
            roster_df.loc[index, 'DOB_MLB'] = data['people'][0]['birthDate']
            try:
                roster_df.loc[index, 'Weight_MLB'] = data['people'][0]['weight']
            except:
                pass
            try:
                roster_df.loc[index, 'UniformNumber_MLB'] = data['people'][0]['primaryNumber']
            except:
                pass
        if not existing_players.empty and data['people']:
            #player_id_db.loc[existing_players.index, 'ID']
            
            roster_df.loc[index, 'MLBAMID'] = data['people'][0]['id']
            roster_df.loc[index, 'DB_ID'] = player_id_db.loc[existing_players.index, 'ID'].unique()
            if roster_df.loc[index, 'MLBAMID'] == int(roster_df.loc[index, 'kt_id']):
                st.success(f"MLBAMID: {data['people'][0]['id']} ID:{player_id_db.loc[existing_players.index, 'ID'].unique()} Player {row['FirstName']} {row['LastName']} already exists in the database.")

            elif roster_df.loc[index, 'MLBAMID'] != roster_df.loc[index, 'kt_id']:
                st.info(f"MLBAMID: {data['people'][0]['id']} ID:{player_id_db.loc[existing_players.index, 'ID'].values.unique()} Player {row['FirstName']} {row['LastName']} already exists in the database.")

        # elif data['people'] and existing_players.empty:
        #     roster_df.loc[index, 'MLBAMID'] = data['people'][0]['id']
        #     st.warning(f"MLBAMID exists: {data['people'][0]['id']} Player {row['FirstName']} {row['LastName']} does not exist in the database.")

        elif not data['people'] and not existing_players.empty:
            roster_df.loc[index, 'kt_id'] = player_id_db.loc[existing_players.index, 'ID'].values.unique()
            st.warning(f"ID:{player_id_db.loc[existing_players.index, 'ID'].values.unique()} Player: {row['FirstName']} {row['LastName']} found in the database.")

        elif not data['people'] and  existing_players.empty:
            st.error(f"Player: {row['FirstName']} {row['LastName']} does not exist in the database and MLBAMID not found.")
    #st.write(roster_df)
    end_time = time.time()
    execution_time = end_time - start_time
    return(roster_df,execution_time)

@st.cache_data
def batch_player_query_mlb_api(df, endpoint, league_string):
    start_time = time.time()
    base_url = "https://statsapi.mlb.com/api/v1/"
    df['full_name'] = df['FirstName'] + ' ' + df['LastName']
    #Batch Query MLBSTATSAPI
    names_string = '&names='.join(df['full_name'].str.replace(' ', '+'))
    names_string = 'names=' + names_string
    url = f"{base_url}{endpoint}{names_string}{league_string}"
    #st.write(url)
    response = requests.get(url)
    player_data = response.json()
    #st.write(data)
    end_time = time.time()
    execution_time = end_time - start_time

    query_df = pd.DataFrame(columns=['MLBAMID', 'FirstName', 'LastName','useFirstName','useLastName','MiddleName','DOB','UniformNumber','Weight'])
    for index, player in enumerate(player_data["people"]):
        query_df.loc[index, 'MLBAMID'] = player['id']
        query_df.loc[index, 'FirstName'] = player['firstName']
        query_df.loc[index, 'useFirstName'] = player['useName']
        query_df.loc[index, 'useLastName'] = player['useLastName']
        query_df.loc[index, 'LastName'] = player['lastName']
        try:
            query_df.loc[index, 'MiddleName'] = player['middleName']
        except:
            pass
        query_df.loc[index, 'DOB'] = player['birthDate']
        try:
            query_df.loc[index, 'Weight'] = player['weight']
        except:
            pass
        try:
            query_df.loc[index, 'UniformNumber'] = player['primaryNumber']
        except:
            pass
            
    return (query_df,execution_time)

# Highlight dataframe
def highlight_cell_red(value):
    return f"background-color: pink;" if pd.isna(value) else None
def highlight_cell_green(value1, value2):
    return f"background-color: green;" if value1 == value2 else None
def filtering_query(existing, query_df):
    # Add normalized name columns to both dataframes
    # Helper function to normalize names

    def normalize_name(name):
        # Remove accents, convert to lowercase, remove Jr./Sr./III etc
        name = unidecode(str(name)).lower()
        name = re.sub(r'\s+(jr\.?|sr\.?|[ivx]+)$', '', name)
        return name
    #st.write(query_df)
    existing['FirstName_norm'] = existing['FirstName'].apply(normalize_name)
    existing['LastName_norm'] = existing['LastName'].apply(normalize_name)
    query_df['FirstName_norm'] = query_df['FirstName'].apply(normalize_name)
    query_df['LastName_norm'] = query_df['LastName'].apply(normalize_name)
    query_df['useFirstName_norm'] = query_df['useFirstName'].apply(normalize_name)
    query_df['useLastName_norm'] = query_df['useLastName'].apply(normalize_name)
     # Create new MLBAMID column in existing df
    existing['MLBAMID'] = None
    existing['FirstName_MLB'] = None
    existing['LastName_MLB'] = None
    existing['useFirstName_MLB'] = None
    existing['useLastName_MLB'] = None
    existing['MiddleName_MLB'] = None

    # Convert DB_ID to numeric for comparison
    existing['DB_ID'] = pd.to_numeric(existing['DB_ID'], errors='coerce')
    
    # Find matches between DB_ID and MLBAMID
    for idx, row in existing.iterrows():
        if pd.notna(row['DB_ID']):
            match = query_df[query_df['MLBAMID'] == row['DB_ID']]
            if not match.empty:
                existing.at[idx, 'MLBAMID'] = row['DB_ID']
                existing.at[idx, 'FirstName_MLB'] = match.iloc[0]['FirstName']
                existing.at[idx, 'LastName_MLB'] = match.iloc[0]['LastName']
                existing.at[idx, 'useFirstName_MLB'] = match.iloc[0]['useFirstName']
                existing.at[idx, 'useLastName_MLB'] = match.iloc[0]['useLastName']
                existing.at[idx, 'MiddleName_MLB'] = match.iloc[0]['MiddleName']
                existing.at[idx, 'DOB_MLB'] = match.iloc[0]['DOB']
    
    # Match Query Name with Roster Name, no need for players who exists in database and have matching ids
    #compares firstname and lastname, also usefirstname and use lastname, also usefirstname and use uselastname

    # Drops duplicates
    query_df_no_dupes = query_df.drop_duplicates(subset=['FirstName', 'LastName'], keep=False)
    #st.write(query_df_no_dupes)
    
    # Cross Reference Weight 
    duplicates = query_df[query_df.duplicated(subset=['FirstName', 'LastName'], keep=False)]
    duplicates = duplicates.sort_values(['FirstName', 'LastName'])
    #st.write(duplicates)
    # After creating duplicates dataframe...
    filtered_duplicates = duplicates.copy()
    indices_to_drop = set()  # Keep track of indices to drop

    for idx, row in duplicates.iterrows():
        # Find matching name in existing df
        existing_match = existing[
            ((existing['FirstName_norm'] == row['FirstName_norm']) & 
             (existing['LastName_norm'] == row['LastName_norm'])) |
            ((existing['FirstName_norm'] == row['useFirstName_norm']) & 
             (existing['LastName_norm'] == row['LastName_norm'])) |
            ((existing['FirstName_norm'] == row['FirstName_norm']) & 
             (existing['LastName_norm'] == row['useLastName_norm']))
        ]
        
        if not existing_match.empty:
            # Get the weight from existing df
            if not pd.isna(existing_match.iloc[0]['Weight']):
                existing_weight = existing_match.iloc[0]['Weight']
                
                # Find all rows in duplicates with same name
                same_name_rows = duplicates[
                    ((duplicates['FirstName_norm'] == row['FirstName_norm']) & 
                     (duplicates['LastName_norm'] == row['LastName_norm'])) |
                    ((duplicates['useFirstName_norm'] == row['FirstName_norm']) & 
                     (duplicates['LastName_norm'] == row['LastName_norm'])) |
                    ((duplicates['FirstName_norm'] == row['FirstName_norm']) & 
                     (duplicates['useLastName_norm'] == row['LastName_norm']))
                ]
                
                # Keep only the row where weight matches
                if pd.notna(existing_weight):
                    # Add indices of non-matching weights to the set
                    indices_to_drop.update(
                        same_name_rows[same_name_rows['Weight'] != existing_weight].index
                    )

    # Drop all collected indices at once
    filtered_duplicates = filtered_duplicates.drop(index=list(indices_to_drop))

    # st.write('Filtered duplicates (after weight matching):')
    # st.write(filtered_duplicates)
    query_df_no_dupes = pd.concat([query_df_no_dupes, filtered_duplicates], ignore_index=True)


    for idx, row in existing.iterrows():
        if pd.isna(row['MLBAMID']):  # Only check rows where MLBAMID is empty
            matches = query_df_no_dupes[
                ((query_df_no_dupes['FirstName_norm'] == row['FirstName_norm']) & 
                (query_df_no_dupes['LastName_norm'] == row['LastName_norm'])) |
                ((query_df_no_dupes['useFirstName_norm'] == row['FirstName_norm']) & 
                (query_df_no_dupes['LastName_norm'] == row['LastName_norm'])) |
                ((query_df_no_dupes['FirstName_norm'] == row['FirstName_norm']) & 
                (query_df_no_dupes['useLastName_norm'] == row['LastName_norm']))
            ]
            
            if not matches.empty:
                # Take the first match if multiple exist
                match = matches.iloc[0]
                existing.at[idx, 'MLBAMID'] = match['MLBAMID']
                existing.at[idx, 'FirstName_MLB'] = match['FirstName']
                existing.at[idx, 'LastName_MLB'] = match['LastName']
                existing.at[idx, 'useFirstName_MLB'] = match['useFirstName']
                existing.at[idx, 'useLastName_MLB'] = match['useLastName']
                existing.at[idx, 'MiddleName_MLB'] = match['MiddleName']
                existing.at[idx, 'DOB_MLB'] = match['DOB']
                existing.at[idx, 'Weight_MLB'] = match['Weight']

    return (existing, query_df_no_dupes)

def get_available_ids(player_id_db, site, num_needed):
    """
    Find available IDs within a site's ranges.
    
    Args:
        player_id_db: DataFrame containing existing player IDs
        site: String key for site_ranges dictionary
        num_needed: Number of IDs needed
    
    Returns:
        List of available IDs within the site's ranges
    """
    if site not in site_ranges:
        return []
    
    try:
        # Convert existing IDs to set for O(1) lookup, excluding NaN values
        valid_ids = player_id_db['ID'].dropna()  # First drop NaN values
        # Convert to integers, ignoring any non-numeric values
        numeric_ids = pd.to_numeric(valid_ids, errors='coerce')
        # Drop any NaN values that resulted from the conversion
        clean_ids = numeric_ids.dropna()
        # Convert to integers and then to a set
        existing_ids = set(clean_ids.astype(int).values)
        
        available_ids = []
        
        # Check each range for the site
        for start, end in site_ranges[site]:
            if len(available_ids) >= num_needed:
                break
                
            range_ids = set(range(start, end + 1))
            available_in_range = list(range_ids - existing_ids)
            
            remaining_needed = num_needed - len(available_ids)
            available_ids.extend(sorted(available_in_range)[:remaining_needed])
        
        return sorted(available_ids)[:num_needed]
        
    except Exception as e:
        st.error(f"Error processing IDs: {str(e)}")
        st.write("First few rows of ID column:", player_id_db['ID'].head())
        return []

def get_detailed_site_availability(player_id_db):
    """
    Get detailed availability information for each site.
    """
    try:
        valid_ids = player_id_db['ID'].dropna()
        numeric_ids = pd.to_numeric(valid_ids, errors='coerce')
        clean_ids = numeric_ids.dropna()
        existing_ids = set(clean_ids.astype(int).values)
        
        site_details = []
        
        for site, ranges in site_ranges.items():
            total_slots = 0
            total_available = 0
            
            # Ensure ranges is a list of tuples
            if not isinstance(ranges, list):
                ranges = [ranges]  # Convert to list if it's a single tuple
            
            for start, end in ranges:
                range_ids = set(range(start, end + 1))
                available_in_range = len(range_ids - existing_ids)
                total_slots += len(range_ids)
                total_available += available_in_range
            
            site_details.append({
                'Site': site,
                'Total Slots': total_slots,
                'Used IDs': total_slots - total_available,
                'Available IDs': total_available,
                #'Utilization %': round((1 - total_available / total_slots) * 100, 2)
            })
            
        return pd.DataFrame(site_details)
        
    except Exception as e:
        st.error(f"Error analyzing site availability: {str(e)}")
        return pd.DataFrame()

def search_mlbam_id(df):
    api_url = "https://statsapi.mlb.com/api/v1/people/"
    id_string = '&personIds='.join(df['MLBAMID'].astype(str))
    id_string = '?personIds=' + id_string
    url = f"{api_url}{id_string}"
    #st.write(url)
    df = df.reset_index(drop=True)
    response = requests.get(url)
    player_data = response.json()
    query_df = pd.DataFrame(columns=['FirstName','LastName','MLBAMID', 'FirstName_MLB', 'LastName_MLB','useFirstName_MLB','useLastName_MLB','MiddleName_MLB','DOB_MLB','UniformNumber_MLB','Weight_MLB'])
    for index, player in enumerate(player_data["people"]):
        query_df.loc[index, 'FirstName'] = df.loc[index,'FirstName']
        query_df.loc[index, 'LastName'] = df.loc[index,'LastName']
        query_df.loc[index, 'MLBAMID'] = player['id']
        query_df.loc[index, 'FirstName_MLB'] = player['firstName']
        query_df.loc[index, 'useFirstName_MLB'] = player['useName']
        query_df.loc[index, 'useLastName_MLB'] = player['useLastName']
        query_df.loc[index, 'LastName_MLB'] = player['lastName']
        try:
            query_df.loc[index, 'MiddleName_MLB'] = player['middleName']
        except:
            pass
        query_df.loc[index, 'DOB_MLB'] = player['birthDate']
        try:
            query_df.loc[index, 'Weight_MLB'] = player['weight']
        except:
            pass
        try:
            query_df.loc[index, 'UniformNumber_MLB'] = player['primaryNumber']
        except:
            pass
    st.write(query_df)
    return(query_df)

#####################################################################################################
# MLB Stats API
#mlb = mlbstatsapi.Mlb() # python module for MLB stats API

# Base URL of the Stats API
base_url = "https://statsapi.mlb.com/api/v1/"
# End Point to search in All Leagues (Major League, Minors, AAA...)
# MLB: 1, AAA: 11, Double-A: 12, High A: 13, Single A: 14, Rookie: 16, Winter: 17
# Minor League: 21, Indie League: 23, https://statsapi.mlb.com/docs/endpointPage?random=90#get-/api/v1/sports for more leagues
# ALL LEAGUES: 
#leagues_endpoint = "&sportIds=1&sportIds=11&sportIds=12&sportIds=13&sportIds=14&sportIds=16&sportIds=17&sportIds=21&sportIds=23&sportIds=61&sportIds=32&sportIds=31&sportIds=51&sportIds=509&sportIds=510&sportIds=6005&sportIds=22&sportIds=586"
leagues_endpoint = "&sportIds=1&sportIds=11&sportIds=12&sportIds=13&sportIds=14&sportIds=16&sportIds=17&sportIds=21&sportIds=23&sportIds=61&sportIds=32&sportIds=31&sportIds=51&sportIds=509&sportIds=510&sportIds=6005&sportIds=22&sportIds=586"
endpoint_series = "people/search?names="
endpoint_batch = "people/search?"
league_list_url = "https://statsapi.mlb.com/api/v1/sports"
league_response = requests.get(league_list_url)
league_list = league_response.json()
#######################################################################################################\
# Pre-defined ranges for each site
site_ranges = {
    'ABW' : [(320000,320999),(904000,904999)],
    'AUB' : [(321000,321999),(500870,500898),(700299,700499)],
    'MEM' : [(900000,900499)],
    'TEX' : [(400011,403679)],
    'LAD/CHW/OAK': [(200000,210999)],
    'CIN' : [(700000,700298)],
    'MIN' : [(700500,704999)],
    'UMB' : [(705000,705300)],
    'ARI' : [(900500,901496)],
    'STF' : [(800001,800148),(900149,901414),(901497,901598)],
    'TBR' : [(100000,106310)],
    'SEA' : [(902000,902599)],
    'WFU' : [(902600,902999),(905000,905499)],
    'UOA' : [(905500,905899)],
    'DLL' : [(905900,905999)],
    'MIL' : [(903000,903999)],
    'BNR' : [(904000,904999)],
    'BOS/PHI' : [(300000,310205)],
    'CHC' : [(1,99999)]

}

site_range_indices = [index for index, site_range in enumerate(site_ranges)]

# Session State Initialization
if 'start_interval' not in st.session_state:
    st.session_state.start_interval = None
if 'end_interval' not in st.session_state:
    st.session_state.end_interval = None
if 'site' not in st.session_state:
    st.session_state.site = None


########################################################################################################
# Load Player ID Database
player_id_db = pd.read_csv(r"C:\Users\HansonLu\Downloads\player_id_db_kinatrax_rev1.csv")

# Upload Roster CSV
st.title("Player ID Assigner")
st.header('1. Import Roster CSV')
uploaded_file = st.file_uploader("Upload Roster CSV", type="csv")
st.header('2. Enter Additional Information')
col1, col2, col3 = st.columns(3)
st.session_state.site = col1.selectbox('Site',site_ranges) # Should have pre-assigned intervals for each site, input the interval and find available ids from DB
date_of_assignment = col2.date_input('Todays Date')
league = col3.selectbox('League', [sport['name'] for sport in league_list['sports']])
league_id = next((sport['id'] for sport in league_list['sports'] if sport['name'] == league), None)

# Insert into Dataframe about site, year, and league
if st.checkbox("Custom Interval/Site"):
    st.session_state.start_interval = col1.number_input('Start Interval', step= 1)
    st.session_state.end_interval = col2.number_input('End Interval', max_value = 999999)
    col3.number_input('Number of Players', step= 1, max_value = st.session_state.end_interval - st.session_state.start_interval)
    site2 = col1.text_input('Custom Site', value = 'NaN')
    if site2 is not 'NaN':
        st.session_state.site = site2

    
    # With custom intervals, search available IDs between those numbers, and return sequentially a list of available ids
    # Or auto detect on the roster, fill in needed positions
detailed_availability = get_detailed_site_availability(player_id_db)
with st.expander("Detailed Site Availability:"):
    st.dataframe(detailed_availability.set_index('Site'))


if uploaded_file is not None:
    # Intake Roster CSV
    df = pd.read_csv(uploaded_file)

    # Search Existing Database to see if it matches any player names
    existing = search_existing_players(df, player_id_db)

    # Session State Caching - Streamlit Specific Data Retention Method
    if 'existing' not in st.session_state:
        st.session_state.existing = existing
    if 'empty_db_id_df' not in st.session_state:
        st.session_state.empty_db_id_df = pd.DataFrame()
    if 'sequential_query_df' not in st.session_state:
        st.session_state.sequential_query_df = None

    # Batch Querying MLB API
    query_df, batch_time = batch_player_query_mlb_api(df, endpoint_batch, leagues_endpoint)
    #st.write(query_df)
    # Filtering Query to remove duplicate player names
    existing, filtered_query_df = filtering_query(existing, query_df)
    
    # Query players the batch query missed
    st.write("Players needing ID assignment:")
    empty_bam_id_df = existing [existing ['MLBAMID'].isna()]
    st.session_state.sequential_query_df, time = player_check(empty_bam_id_df, player_id_db, endpoint_series,leagues_endpoint)
    existing.update(st.session_state.sequential_query_df)

    # Adding additional Information
    existing['site'] = st.session_state.site
    existing['league_id'] = league_id
    existing['date_of_assignment'] = date_of_assignment

    st.header("3. Auto-Fill BAMID as Player ID")
    if st.button('Assign MLBAM ID as Player_ID'):

        # Find rows where DB_ID is empty and MLBAMID exists
        #existing = st.session_state.existing
        mask = (existing['DB_ID'].isna()) & (existing['MLBAMID'].notna())
        # Update DB_ID with MLBAMID where mask is True
        existing.loc[mask, 'DB_ID'] = existing.loc[mask, 'MLBAMID']
        existing['DB_ID'] = existing['DB_ID'].astype('Int64') 
        existing['MLBAMID'] = existing['MLBAMID'].astype('Int64') 
        st.session_state.existing = existing
        #st.write(st.session_state.sequential_query_df)

        # Optional: Show success message
        num_updated = mask.sum()
        if num_updated > 0:
            st.success(f"Updated {num_updated} player(s) with new DB_ID from MLBAMID")
        else:
            st.info("No new DB_IDs to assign")
    if 'existing' in st.session_state and (st.session_state.existing is not None and not st.session_state.existing.empty):
        st.session_state.empty_db_id_df = st.session_state.existing [st.session_state.existing ['DB_ID'].isna()]
        

    st.session_state.empty_db_id_df = st.data_editor(st.session_state.empty_db_id_df )
    st.header('4. Manually Input BAMID, if Query Tool Cannot Find')
    if st.button('Manual MLBAMID Input'):
        if not st.session_state.empty_db_id_df[st.session_state.empty_db_id_df['MLBAMID'].notna()].empty:
            #st.write(st.session_state.empty_db_id_df[st.session_state.empty_db_id_df['MLBAMID'].notna()])
            new_df = search_mlbam_id(st.session_state.empty_db_id_df[st.session_state.empty_db_id_df['MLBAMID'].notna()])
            # If BAM ID search returns nothing, prompt BAM ID INVALID
            # Assuming new_df contains the updated information and existing is your original DataFrame
            new_df['DB_ID'] = new_df['MLBAMID']
            # Merge existing with new_df on FirstName and LastName
            merged_df = st.session_state.existing.merge(new_df, on=['FirstName', 'LastName'], how='left', suffixes=('', '_new'))

            # Update the existing DataFrame with values from new_df where they exist
            for column in new_df.columns:
                try:
                    if column not in ['FirstName', 'LastName']:  # Skip the matching columns
                        merged_df[column] = merged_df[column + '_new'].combine_first(merged_df[column])
                except:
                    pass

            # Drop the temporary columns created during the merge
            merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_new')])

            # Update the existing DataFrame in session state
            st.session_state.existing = merged_df
            if new_df.empty:
                st.write('No BAM ID Found')
            # Optionally, display the updated DataFrame
            #t.write(st.session_state.existing)
        else:
            st.write('No BAM ID Entered')
        
    
        # we want to check the dataframe to be robust before pushing it into the database
        # We also want to create a column for cubs ids and site code and league
        # We want to also have a uploaded roster with assigned ids for manual input

    st.session_state.empty_db_id_df = st.session_state.empty_db_id_df[st.session_state.empty_db_id_df['MLBAMID'].isna()]
    num_ids_needed = len(st.session_state.empty_db_id_df)

    selected_site = st.session_state.site  # or however you're getting the site
    
    st.header("5. if No-BAMID, Assign Player ID")
    if selected_site:
        available_ids = get_available_ids(player_id_db, selected_site, num_ids_needed)
        if len(available_ids) >= num_ids_needed:
            st.write(f"Available IDs for {selected_site}: {available_ids}")
            if st.button('Assign Available ID to New Players'): # We can move this button to process below
                st.session_state.empty_db_id_df.loc[st.session_state.empty_db_id_df['DB_ID'].isna(), 'DB_ID'] = available_ids[:num_ids_needed]
                assigned_player_id_df = st.session_state.existing.merge(st.session_state.empty_db_id_df, on=['FirstName', 'LastName'], how='left', suffixes=('', '_new'))

                # Update the existing DataFrame with values from new_df where they exist
                for column in st.session_state.empty_db_id_df.columns:
                    try:
                        if column not in ['FirstName', 'LastName']:  # Skip the matching columns
                            assigned_player_id_df[column] = assigned_player_id_df[column + '_new'].combine_first(assigned_player_id_df[column])
                    except:
                        pass

                # Drop the temporary columns created during the merge
                assigned_player_id_df = assigned_player_id_df.drop(columns=[col for col in assigned_player_id_df.columns if col.endswith('_new')])

                # Update the existing DataFrame in session state
                st.session_state.existing = assigned_player_id_df
                

        else:
            st.error(f"Not enough available IDs in the ranges for {selected_site}")
            #If there's not enough available IDs start at 100000
            if len(available_ids) < num_ids_needed:
                st.write(len(available_ids))

    if 'existing' in st.session_state and (st.session_state.existing is not None and not st.session_state.existing.empty):
        st.dataframe(st.session_state.existing.style.apply(lambda x: ['background-color: pink' if pd.isna(x['DB_ID']) else '' for i in x], axis=1))
    #t.dataframe(st.session_state.existing.style.apply(lambda x: ['background-color: pink' if pd.isna(x['DB_ID']) else '' for i in x], axis=1))

    #I'm just going to keep this as a google sheet, and using api to append rows of new assigned IDs to the page