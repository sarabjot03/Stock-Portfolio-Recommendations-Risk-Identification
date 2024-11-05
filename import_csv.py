import os
import pymysql

# Define your variables
MYSQL_USER = "anmol"  # Replace with your MySQL username
MYSQL_PASSWORD = "root"  # Replace with your MySQL password
DATABASE_NAME = "Capstone_Project"  # Your database name
TABLE_NAME = "stocks"  # Your target table name
CSV_FOLDER = "/Users/anmol/Documents/Lambton Stuff/Semester 3/Big Data Capstone Project"  # Folder containing CSV files
MYSQL_HOST = "34.130.190.232"  # IP address of the MySQL server

# Establish database connection
connection = pymysql.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=DATABASE_NAME,
    local_infile=1
)

try:
    with connection.cursor() as cursor:
        # Loop through all CSV files in the specified directory
        for filename in os.listdir(CSV_FOLDER):
            if filename.endswith(".csv"):
                file_path = os.path.join(CSV_FOLDER, filename)
                print(f"Loading {file_path} into {TABLE_NAME}...")

                load_data_command = f"""
                LOAD DATA LOCAL INFILE '{file_path}'
                INTO TABLE {TABLE_NAME}
                FIELDS TERMINATED BY ','
                ENCLOSED BY '"'
                LINES TERMINATED BY '\\n'
                IGNORE 1 ROWS;
                """

                # Execute the LOAD DATA command
                try:
                    cursor.execute(load_data_command)
                    connection.commit()
                    print(f"Successfully loaded {file_path}")
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

finally:
    connection.close()

print("All files have been attempted for loading.")
