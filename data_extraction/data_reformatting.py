def guarvis_input_data(activity_data, sleep_data):
    if not activity_data:
        raise ValueError("Activity data is required for Guarvis model input.")
    if not sleep_data:
        raise ValueError("Sleep data is required for Guarvis model input.")
    return 1

def imh_input_data(activity_data, activity_type_data, location_data):
    if not activity_data:
        raise ValueError("Activity data is required for IMH model input.")
    if not activity_type_data:
        raise ValueError("Activity type data is required for IMH model input.")
    if not location_data:
        raise ValueError("Location data is required for IMH model input.")
    return 1