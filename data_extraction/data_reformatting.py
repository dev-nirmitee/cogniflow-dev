import copy

import pandas as pd

def guarvis_input_data(activity_data, sleep_data):
    if not activity_data:
        raise ValueError("Activity data is required for Guarvis model input.")
    if not sleep_data:
        raise ValueError("Sleep data is required for Guarvis model input.")
    required_activity_columns = ['act_in_vehicle_ep_0','act_in_vehicle_ep_1','act_in_vehicle_ep_2','act_in_vehicle_ep_3','act_in_vehicle_ep_4',
                         'act_on_bike_ep_0','act_on_bike_ep_1','act_on_bike_ep_2','act_on_bike_ep_3','act_on_bike_ep_4',
                         'act_on_foot_ep_0','act_on_foot_ep_1','act_on_foot_ep_2','act_on_foot_ep_3','act_on_foot_ep_4',
                         'act_running_ep_0','act_running_ep_1','act_running_ep_2','act_running_ep_3','act_running_ep_4',
                         'act_still_ep_0','act_still_ep_1','act_still_ep_2','act_still_ep_3','act_still_ep_4',
                         'act_tilting_ep_0','act_tilting_ep_1','act_tilting_ep_2','act_tilting_ep_3','act_tilting_ep_4',
                         'act_unknown_ep_0','act_unknown_ep_1','act_unknown_ep_2','act_unknown_ep_3','act_unknown_ep_4',
                         'act_walking_ep_0','act_walking_ep_1','act_walking_ep_2','act_walking_ep_3','act_walking_ep_4',
                         'quality_activity']
    required_sleep_columns = ['sleep_duration','sleep_end', 'sleep_start']
    useful_columns = ['date', 'project', 'user']
    filtered_activity_data = list()
    for record in activity_data:
        filtered_record = dict()
        for col in useful_columns:
            if col in record:
                filtered_record[col] = record[col]
            else:
                raise ValueError(f"Missing essential column '{col}' in activity data.")
        for col in required_activity_columns:
            if col in record:
                filtered_record[col] = record[col]
            else:
                filtered_record[col] = 0
        filtered_activity_data.append(filtered_record)
    filtered_activity_data = pd.DataFrame(filtered_activity_data)
    filtered_activity_data = filtered_activity_data.drop_duplicates(subset=['date', 'project', 'user'], keep='last')
    filtered_sleep_data = list()
    for record in sleep_data:
        filtered_record = dict()
        for col in useful_columns:
            if col in record:
                filtered_record[col] = record[col]
            else:
                raise ValueError(f"Missing essential column '{col}' in sleep data.")
        for col in required_sleep_columns:
            if col in record:
                filtered_record[col] = record[col]
            else:
                filtered_record[col] = 0
        filtered_sleep_data.append(filtered_record)
    filtered_sleep_data = pd.DataFrame(filtered_sleep_data)
    filtered_sleep_data = filtered_sleep_data.drop_duplicates(subset=['date', 'project', 'user'], keep='last')
    combined_data = pd.merge(filtered_activity_data, filtered_sleep_data, on=['date', 'project', 'user'], how='outer')
    combined_data = combined_data.fillna(0)
    return combined_data

def imh_input_data(activity_data, activity_type_data, location_data):
    if not activity_data:
        raise ValueError("Activity data is required for IMH model input.")
    if not activity_type_data:
        raise ValueError("Activity type data is required for IMH model input.")
    if not location_data:
        raise ValueError("Location data is required for IMH model input.")
    required_activity_columns = ['stepcount', 'user', 'project', 'date']
    activity_data_list = list()
    for record in activity_data:
        filtered_record = dict()
        for col in required_activity_columns:
            if col in record:
                filtered_record[col] = copy.copy(record[col])
            else:
                raise ValueError(f"Missing essential column '{col}' in activity data.")
        activity_data_list.append(copy.deepcopy(filtered_record))
    activity_data_df = pd.DataFrame(activity_data_list)
    activity_data_df = activity_data_df.drop_duplicates(subset=['date', 'project', 'user'], keep='last')
    
    activity_type_data_list = list()
    for record in activity_type_data:
        for col in ['user', 'project', 'date']:
            if col not in record:
                raise ValueError(f"Missing essential column '{col}' in activity type data.")
            filtered_record[col] = copy.copy(record[col])
        if 'history' in record:
            filtered_record['count'] = len(record['history'])
        activity_type_data_list.append(copy.deepcopy(filtered_record))
    activity_type_data_df = pd.DataFrame(activity_type_data_list)
    activity_type_data_df = activity_type_data_df.drop_duplicates(subset=['date', 'project', 'user'], keep='last')

    location_data_list = list()
    for record in location_data:
        filtered_record = dict()
        for col in ['user', 'project', 'date']:
            if col not in record:
                raise ValueError(f"Missing essential column '{col}' in location data.")
            filtered_record[col] = copy.copy(record[col])
        if 'history' in record:
            filtered_record['count'] = len(record['history'])
        location_data_list.append(copy.deepcopy(filtered_record))
    location_data_df = pd.DataFrame(location_data_list)
    location_data_df = location_data_df.drop_duplicates(subset=['date', 'project', 'user'], keep='last')

    combined_data = pd.merge(activity_data_df, activity_type_data_df, on=['date', 'project', 'user'], how='outer')
    combined_data = pd.merge(combined_data, location_data_df, on=['date', 'project', 'user'], how='outer')
    combined_data = combined_data.fillna(0)
    return combined_data