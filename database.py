import traceback
import sys

import duckdb
import pandas as pd
from sqlalchemy import create_engine

import config
from logger import create_logger


class Database:
    def __init__(self, database_name: str = config.DB_NAME) -> None:
        try:
            self.logger = create_logger("database.log")
        except Exception:
            print("Failed to create logger! \n%s", traceback.format_exc())
            
        try:
            self.connection = duckdb.connect(database_name)
        except Exception:
            self.logger.critical(
                "Failed to connect to database! \n%s", traceback.format_exc()
            )
            sys.exit()
        
    def create_table_from_csv(self, table_name: str, csv_path: str) -> None:
        """
        Creates a table in the database from a given CSV file.

        :param table_name: The name of the table to be created.
        :param csv_path: The path to the CSV file to be used to populate the table.
        :return: None
        :raises Exception: If there is an error creating the table.
        """
        try:
            self.connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS 
                SELECT * FROM read_csv_auto('{csv_path}')
                """
            )
        except Exception:
            self.logger.error(
                "Failed to create table %s from CSV %s! \n%s",
                table_name,
                csv_path,
                traceback.format_exc(),
            )
            
    def execute_read_query(
        self, query: str, args: tuple = (), return_as_dataframe: bool = True
    ) -> None | pd.DataFrame | list:
        """
        Executes a read query on the database and returns the result
        as a Pandas DataFrame (default) or a list of tuples.

        :param query: The SQL query to be executed.
        :param return_as_dataframe: If True, the result will be returned as a Pandas DataFrame.
        :return: The result of the query. If return_as_dataframe is False, the result will be a list of tuples.
        :raises Exception: If there is an error executing the query.
        """
        try:
            result = self.connection.execute(query, args)
            if return_as_dataframe:
                return result.df()
            return result.fetchall()
        except Exception:
            self.logger.error(
                "Failed to execute read query! \n%s", traceback.format_exc()
            )
            return None
    
    def execute_write_query(self, query: str) -> bool:
        """
        Executes a write query on the database and returns True if
        the query is executed successfully, False otherwise.

        :param query: The SQL query to be executed.
        :return: True if the query is executed successfully, False otherwise.
        :raises Exception: If there is an error executing the query.
        """
        try:
            self.connection.execute(query)
            return True
        except Exception:
            self.logger.error(
                "Failed to execute write query: %s \n%s",
                query,
                traceback.format_exc(),
            )
            return False

    def check_if_table_exists(self, table_name: str) -> bool:
        """
        Checks if a table exists in the database.

        :param table_name: The name of the table to check.
        :return: True if the table exists, False otherwise.
        :raises Exception: If there is an error while checking if the table exists.
        """
        try:
            result = self.connection.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
                """,
                [table_name],
            ).fetchone()
            
            if isinstance(result, tuple):
                result = result[0]
                return result > 0
            return False

        except Exception as e:
            self.logger.error(
                "Error while checking if table %s exists: %s",
                table_name,
                e,
            )
            return False