import os

from collected_item import CollectedItem
from pymongo import MongoClient, errors
from dotenv import load_dotenv

class MongoDBConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.client = None
        self.db = None
        self.collected_items_coll = None
        self.__connect()
    
    def __create_or_get_collection(self, name):
        if name not in self.db.list_collection_names():
            self.collected_items_coll = self.db.create_collection(name)
        else:
            self.collected_items_coll = self.db.get_collection(name)

    def __connect(self):
        '''
        Creates a connection to MongoDB and sets up the database and collection.
        '''

        try:
            print('Connecting to the MongoDB...')
            self.client = MongoClient(self.connection_string)
            
            # Check or create the database "prismai"
            self.db = self.client.get_database('prismai')
            
            # Check or create the collection "collected_items"
            self.__create_or_get_collection('collected_items')
                
            print('Connected to MongoDB successfully.')
        except errors.ConnectionFailure as e:
            print(f'Failed to connect to MongoDB: {e}')
        except errors.PyMongoError as e:
            print(f'MongoDB error: {e}')

    def insert_collected_item(self, item: CollectedItem):
        '''
        Insert a CollectedItem only if it doesnt exist already.
        '''
        self.collected_items_coll.update_one(
            {"id": item.id},
            {"$setOnInsert": item.__dict__},
            upsert=True
        )

    def insert_many_collected_items(self, items: list[CollectedItem]):
        '''
        Insert many CollectedItems only if it doesnt exist already.
        '''
        self.collected_items_coll.insert_many([item.__dict__ for item in items])

    def update_collected_item(self, collected_item: CollectedItem):
        try:
            item_dict = collected_item.__dict__
            item_id = item_dict.pop("id", None)
            
            if item_id is None:
                raise ValueError('CollectedItem object must have an id field.')

            result = self.collected_items_coll.update_one(
                {"id": item_id}, 
                {"$set": item_dict} 
            )
            return result
        except Exception as ex:
            print(f'Error while updating item with id {item_id}: {ex}')
            return None

    def get_collected_items_by_domain(self, domain, batch_size=100000, skip=0):
        try:
            cursor = self.collected_items_coll.find({"domain": domain}, no_cursor_timeout=True).sort([('_id', -1)]).skip(skip)
            while True:
                batch = []
                for _ in range(batch_size):
                    try:
                        batch.append(next(cursor))
                    except StopIteration:
                        break  
                if not batch:
                    break
                for item in batch:
                    yield item
        except Exception as ex:
            print('Error while fetching items by domain: ', ex)
        finally:
            if cursor:
                cursor.close()

    def get_collected_items_by_text(self, text, batch_size=100000, skip=0):
        try:
            cursor = self.collected_items_coll.find({"text": text}, no_cursor_timeout=True).sort([('_id', -1)]).skip(skip)
            while True:
                batch = []
                for _ in range(batch_size):
                    try:
                        batch.append(next(cursor))
                    except StopIteration:
                        break 
                if not batch:
                    break
                for item in batch:
                    yield item
        except Exception as ex:
            print('Error while fetching items by text: ', ex)
        finally:
            if cursor:
                cursor.close()

    def close(self):
        self.client.close()

    def count_collected_items(self) -> int:
        if self.collected_items_coll is not None:
            try:
                count = self.collected_items_coll.count_documents({})
                print(f"Document count in 'collected_items': {count}")
                return count
            except errors.PyMongoError as e:
                print(f'Error counting documents: {e}')
                return 0
        else:
            print('Collection not initialized.')
            return 0

# Test the Client
if __name__ == '__main__':
    load_dotenv()
    connection_string = os.getenv('MONGO_DB_CONNECTION')
    mongo_conn = MongoDBConnection(connection_string)
    document_count = mongo_conn.count_collected_items()
    print('collected_items: ' + str(document_count))
    mongo_conn.close()
