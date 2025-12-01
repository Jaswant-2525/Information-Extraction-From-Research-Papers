import json
import mysql.connector
from pymongo import MongoClient

# ---------- Load Extracted JSON ----------
def load_extracted_data(json_path="output.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- MySQL Storage ----------
def store_in_mysql(data):
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="jash@2532006",
        database="extract_paper"
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS named_entities (
            id INT AUTO_INCREMENT PRIMARY KEY,
            entity TEXT,
            label VARCHAR(255)
        )
    """)

    cursor.execute("DELETE FROM named_entities")  # optional: clear existing data

    for entity, label in data.get("named_entities", []):
        cursor.execute("INSERT INTO named_entities (entity, label) VALUES (%s, %s)", (entity, label))

    conn.commit()
    cursor.close()
    conn.close()
    print("Data saved to MySQL successfully.")

# ---------- MongoDB Storage ----------
def store_in_mongodb(data):
    client = MongoClient("mongodb://localhost:27017")
    db = client["pdf_extraction"]
    collection = db["extracted_data"]

    collection.delete_many({})  # optional: clear existing data
    collection.insert_one(data)
    print("Data saved to MongoDB successfully.")

# ---------- Main Execution ----------
if __name__ == "__main__":
    extracted_data = load_extracted_data()


    
    # MySQL
    store_in_mysql(extracted_data)

    # MongoDB
    store_in_mongodb(extracted_data)
