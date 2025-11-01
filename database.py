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
            self.logger.info("Connected to database: %s", database_name)
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
            self.logger.info(
                "Created table %s from CSV %s", table_name, csv_path
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
            self.logger.info("Executed read query: %s", query)
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
            self.logger.info("Executed write query: %s", query)
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
            
            self.logger.info("Checked if table %s exists.", table_name)
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
        
    def create_table_from_df(
        self, table_name: str, df: pd.DataFrame, if_exists: str = "fail"
    ) -> None:
        """
        Creates or modifies a table in the database from a pandas DataFrame.

        This method leverages DuckDB's direct support for pandas DataFrames,
        making it highly efficient.

        Parameters
        ----------
        table_name : str
            The name of the table to be created or modified.
        df : pd.DataFrame
            The DataFrame containing the data to be written.
        if_exists : str, optional
            How to behave if the table already exists. Can be one of:
            - 'fail': Raise a ValueError. (Default)
            - 'replace': Drop the table before inserting new values.
            - 'append': Insert new values to the existing table.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the table exists and `if_exists` is 'fail', or if an invalid
            value for `if_exists` is provided.
        Exception
            If any other database error occurs during the operation.
        """
        if df.empty:
            self.logger.warning(
                "DataFrame is empty. Skipping table creation for '%s'.", table_name
            )
            return

        try:
            if if_exists == "replace":
                # Atomically creates a new table or replaces an existing one.
                self.connection.sql(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
                self.logger.info(
                    "Successfully created or replaced table '%s' with %d rows.",
                    table_name,
                    len(df),
                )

            elif if_exists == "append":
                # Appends data from the DataFrame to an existing table.
                self.connection.sql(f"INSERT INTO {table_name} SELECT * FROM df")
                self.logger.info(
                    "Successfully appended %d rows to table '%s'.", len(df), table_name
                )

            elif if_exists == "fail":
                # Default behavior: check for existence before creating.
                if self.check_if_table_exists(table_name):
                    raise ValueError(f"Table '{table_name}' already exists.")
                
                self.connection.sql(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                self.logger.info(
                    "Successfully created new table '%s' with %d rows.", table_name, len(df)
                )

            else:
                # --- Handle invalid 'if_exists' arguments ---
                raise ValueError(
                    f"Invalid value for if_exists: '{if_exists}'. "
                    "Expected 'fail', 'replace', or 'append'."
                )

        except Exception:
            self.logger.error(
                "Failed to create/modify table '%s' from DataFrame! \n%s",
                table_name,
                traceback.format_exc(),
            )