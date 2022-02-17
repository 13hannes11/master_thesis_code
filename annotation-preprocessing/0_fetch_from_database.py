import mysql.connector
import pandas as pd
import os
from dotenv import load_dotenv

BASE_OBJECT_SQL = """
FROM UniqueGroundTruth 
        JOIN DetectedObject on DetectedObject.id = UniqueGroundTruth.object_id
        JOIN Image on Image.id = DetectedObject.image_id  
        JOIN FocusStack on FocusStack.id = Image.focus_stack_id
        JOIN Scan on Scan.id = FocusStack.scan_id
        JOIN Slide on Slide.id = Scan.slide_id 
        JOIN ObjectType on ObjectType.id = UniqueGroundTruth.object_type_id 
        WHERE metaclass_id = 1 -- only select eggs;
            AND study_id = 31
        ORDER BY UniqueGroundTruth.focus_stack_id
"""

def fetch_objects_from_datase(db):
    cursor = db.cursor()

    cursor.execute("""SELECT
            UniqueGroundTruth.focus_stack_id,
            UniqueGroundTruth.x_min, 
            UniqueGroundTruth.y_min,
            UniqueGroundTruth.x_max, 
            UniqueGroundTruth.y_max,
            UniqueGroundTruth.object_type_id,
            ObjectType.name,
            Image.add_date""" + BASE_OBJECT_SQL)

    result = cursor.fetchall()
    return result

def fetch_focus_stacks_from_database(db):
    cursor = db.cursor()

    cursor.execute("""SELECT 
            FocusStack.id as foucs_stack_id, 
            CONCAT (study_id, "/", uuid, "/", file_name) as file_path, 
            file_name,
            uuid,
            study_id,
            Image.pos_z,
            Image.focus_value,
            Image.add_date
        FROM FocusStack
        JOIN Scan on Scan.id = FocusStack.scan_id
        JOIN Slide on Slide.id = Scan.slide_id 
        JOIN Study on Study .id = Slide.study_id 
        JOIN Image on Image.focus_stack_id  = FocusStack.id
        WHERE 
            FocusStack.id IN( -- get all focus stacks that have objects in them;
                SELECT DISTINCT
                    UniqueGroundTruth.focus_stack_id
                """ + BASE_OBJECT_SQL
        + """
            )
        ORDER BY FocusStack.id DESC, focus_value, focus_level
        """)
    result = cursor.fetchall()
    return result



if __name__ == "__main__":
    load_dotenv()

    db = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

    print("Querring objects...")
    df_objects = pd.DataFrame(fetch_objects_from_datase(db))
    print("Querring stacks...")
    df_stacks = pd.DataFrame(fetch_focus_stacks_from_database(db))

    df_objects.columns = ['stack_id', 'x_min', 'y_min', 'x_max', 'y_max', 'object_type_id', 'name', 'add_date']
    df_stacks.columns = ['stack_id', 'file_path', 'file_name',
            'uuid', 'study_id', 'pos_z', 'focus_value', 'add_date']

    print("Writing objects to file...")
    df_objects.to_csv("out/objects.csv")
    print("Writing stacks to file...")
    df_stacks.to_csv("out/stacks.csv")
