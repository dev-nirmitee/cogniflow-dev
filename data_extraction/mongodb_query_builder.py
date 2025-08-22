import datetime

def data_query(user_id, project_id, start_date=None, end_date=None):
    query = {
        "user": user_id,
        "project": project_id
    }
    
    if not end_date:
        end_date = datetime.datetime.today() - datetime.timedelta(days=1)
    if not start_date:
        start_date = end_date - datetime.timedelta(days=60)
    
    query["date"] = {"$gte": start_date}
    query["date"]["$lte"] = end_date
    
    return query