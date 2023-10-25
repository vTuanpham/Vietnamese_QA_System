import os
import sys
sys.path.insert(0, r'./')
import sqlite3
from sqlite3 import Cursor, Connection, OperationalError
import warnings
from typing import List, Optional, Union, Any

from src.utils import timeit


def setup_database(database_name: str,
                   table_names: List[str] = ["documents"],
                   fields: List[str] = ['''(id INTEGER PRIMARY KEY AUTOINCREMENT, doc TEXT, source TEXT)'''],
                   database_dir: str = "./inference_pipeline/dbs",
                   verbose: bool = True) -> str:
    assert os.path.isdir(database_dir), f"Invalid database_dir path: {database_dir}"
    assert len(table_names) == len(fields), f"The table_names and the fields args must have the same length"
    database_path = os.path.join(database_dir, f"{database_name}.db")
    try:
        connection = sqlite3.connect(database_path)
    except OperationalError as e:
        raise f"Connection to database {database_name} failed with the following error\n" \
              f"Error message: {e}"
    if verbose: print(f"Successfully create database {database_path}")

    cursor = connection.cursor()
    for table_name, field in zip(table_names, fields):
        try:
            cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} {field}')
        except OperationalError as e:
            raise f"Create table {table_name} fail with the following error: {e}"
        if verbose: print(f"Successfully create table {table_name} with field {field}")

    connection.commit()
    connection.close()
    return database_path


def drop_tables(database_path: str,
                tables_to_drop: List[str],
                verbose: bool = True):
    connection = connect_database(database_path, verbose=verbose)
    cursor = connection.cursor()

    for table_name in tables_to_drop:
        try:
            cursor.execute(f"DROP TABLE {table_name}")
        except OperationalError as e:
            raise f"Cannot drop table {table_name} with the following error: {e}"
        if verbose: print(f"Successfully drop table {table_name}")

    connection.commit()
    connection.close()

    if verbose: print(f"Drop tables: {tables_to_drop} successfully")


def query(database_path: str,
          query_string: str,
          fetch_size: Union[int, str] = "all",
          verbose: bool = False) -> Union[list, Any]:
    connection = connect_database(database_path, verbose=verbose)
    cursor = connection.cursor()
    try:
        cursor.execute(query_string)
    except OperationalError as e:
        raise f"Query {query_string} failed with the following error: {e}"

    if fetch_size == 'all':
        if verbose: print("Fetch all rows")
        data = cursor.fetchall()
    elif fetch_size > 1:
        if verbose: print(f"Fetch {fetch_size} rows")
        data = cursor.fetchmany(size=fetch_size)
    elif fetch_size == 1:
        if verbose: print("Fetch 1 row")
        data = cursor.fetchone()
    else:
        connection.close()
        raise "Invalid fetch mode"
    connection.close()
    return data


@timeit
def insert_data(database_path: str,
                table_name: str,
                data: List[dict],
                verbose: bool = True):
    connection = connect_database(database_path, verbose=verbose)
    cursor = connection.cursor()

    try:
        # Start a transaction
        cursor.execute('BEGIN TRANSACTION')

        columns = ', '.join(data[0].keys()) if data else ''
        placeholders = ', '.join(['?'] * len(data[0])) if data else ''

        insert_query = f'INSERT INTO {table_name} ({columns}) VALUES ({placeholders})'
        if verbose: print(f"The query for insert: {insert_query}")

        # Create a list of values for insertion
        values = [tuple(row.values()) for row in data]

        cursor.executemany(insert_query, values)

        # Commit the transaction
        connection.commit()
        if verbose: print(f"Successfully inserted {len(data)} rows into table {table_name} in {database_path}")
    except OperationalError as e:
        connection.rollback()
        raise f"Insertion failed with the following error: {e}"
    finally:
        connection.close()


def connect_database(database_path: str,
                     verbose: bool=False) -> Connection:
    assert os.path.isfile(database_path), f"Invalid database path for {database_path}"
    assert database_path[-2:] == "db" or database_path[-6:] == "sqlite",\
        f"Invalid file, the file must have an extension .db or .sqlite"
    try:
        connection = sqlite3.connect(database_path)
    except OperationalError as e:
        raise f"Connection to database {database_path} failed with the following error\n" \
              f"Error message: {e}"

    if verbose: print(f"Connect to database {database_path} successfully")

    return connection


if __name__ == "__main__":
    setup_database("documents",
                   table_names=["documents", "wiki", "usr_info"],
                   fields=['''(id INTEGER PRIMARY KEY AUTOINCREMENT, doc TEXT, source TEXT)''',
                           '''(id INTEGER PRIMARY KEY AUTOINCREMENT, wikidoc TEXT, header TEXT, source TEXT)''',
                           '''(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT)''']
                   )
    fake_data = [
        {'name': 'John Smith', 'email': 'john.smith@example.com'},
        {'name': 'Alice Johnson', 'email': 'alice.johnson@example.com'},
        {'name': 'David Williams', 'email': 'david.williams@example.com'},
        {'name': 'Emily Brown', 'email': 'emily.brown@example.com'},
        {'name': 'Michael Davis', 'email': 'michael.davis@example.com'},
        {'name': 'Sophia Wilson', 'email': 'sophia.wilson@example.com'},
        {'name': 'Daniel Jones', 'email': 'daniel.jones@example.com'},
        {'name': 'Olivia Miller', 'email': 'olivia.miller@example.com'},
        {'name': 'William Taylor', 'email': 'william.taylor@example.com'},
        {'name': 'Ava Anderson', 'email': 'ava.anderson@example.com'}
    ]
    insert_data("inference_pipeline/dbs/documents.db",
                table_name='usr_info',
                data=fake_data)
    print(query("inference_pipeline/dbs/documents.db",
                query_string='''SELECT * FROM usr_info''',
                fetch_size="all"))
    drop_tables("inference_pipeline/dbs/documents.db",
                tables_to_drop=["documents", "wiki", "usr_info"])