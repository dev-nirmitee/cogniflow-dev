import pymongo

class MongoDBConnector:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.collections = self.db.list_collection_names()
        
    def fetch_data(self, collection_name, query=None):
        collection = self.get_collection(collection_name)
        if query is None:
            return list(collection.find())
        return list(collection.find(query))
    
    def insert_result(self, result, collection_name):
        collection = self.get_collection(collection_name)
        insert_result = collection.insert_one(result)
        return insert_result
    
    def get_collection(self, collection_name):
        return self.db[collection_name]

    def close(self):
        self.client.close()

