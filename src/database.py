from pathlib import Path
import sqlite3
from typing import Optional

import pandas as pd

from src.config import DATABASE_FILE, CSV_FILE



def get_connection() -> sqlite3.Connection:
    """Create a connection to the local SQLite database."""
    connection = sqlite3.connect(DATABASE_FILE, check_same_thread = False)
    connection.row_factory = sqlite3.Row
    return connection



def create_database() -> None:
    """Create the SQLite database from the original CSV dataset."""

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Dataset not found:\n{CSV_FILE}")

    print(f"Loading dataset from: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)


    column_mapping = {
        "timestamp": "timestamp",
        "vehicle_gps_latitude": "latitude",
        "vehicle_gps_longitude": "longitude",
        "iot_temperature": "current_temperature_c",
        "cargo_condition_status": "cargo_condition_code",
        "risk_classification": "risk_classification",
        "delay_probability": "delay_probability",
        "port_congestion_level": "port_congestion_level",
        "route_risk_level": "route_risk_index",

        "fuel_consumption_rate": "fuel_consumption_rate",
        "eta_variation_hours": "eta_variation_hours",
        "traffic_congestion_level": "traffic_congestion_level",
        "warehouse_inventory_level": "warehouse_inventory_level",
        "weather_condition_severity": "weather_condition_severity",
        "shipping_costs": "shipping_costs",
        "supplier_reliability_score": "supplier_reliability_score",
        "lead_time_days": "lead_time_days",
        "historical_demand": "historical_demand",
        "customs_clearance_time": "customs_clearance_time",
        "driver_behavior_score": "driver_behavior_score",
        "fatigue_monitoring_score": "fatigue_monitoring_score",
        "disruption_likelihood_score": "disruption_likelihood_score",
        "delivery_time_deviation": "delivery_time_deviation",
    }

    available_columns = [column for column in column_mapping if column in df.columns]

    df = df[available_columns].rename(
        columns = {
            column: column_mapping[column]
            for column in available_columns
        }
    )

    DATABASE_FILE.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    connection = sqlite3.connect(DATABASE_FILE)

    try:
        df.to_sql("fleet", connection, if_exists = "replace", index = False)
        connection.commit()

    finally:
        connection.close()

    print(f"Database created successfully: {DATABASE_FILE}")
    print(f"Rows inserted: {len(df)}")


def execute_select_query(
    sql_query: str,
    max_rows: int = 20
) -> list[dict]:
    """
    Execute a read-only SELECT query.
    The agent is only allowed to perform SELECT queries.
    """

    cleaned_query = sql_query.strip()

    if not cleaned_query:
        raise ValueError("SQL query cannot be empty.")

    if not cleaned_query.upper().startswith("SELECT"):
        raise PermissionError(
            "Only SELECT queries are allowed."
        )

    # Prevent multiple statements
    if ";" in cleaned_query.rstrip(";"):
        raise PermissionError(
            "Multiple SQL statements are not allowed."
        )

    connection = get_connection()

    try:
        cursor = connection.execute(cleaned_query)

        rows = cursor.fetchmany(max_rows)

        return [
            dict(row)
            for row in rows
        ]

    finally:
        connection.close()



def get_table_schema() -> list[dict]:
    """Return information about the fleet table."""
    connection = get_connection()

    try:
        cursor = connection.execute("PRAGMA table_info(fleet)")
        return [dict(row) for row in cursor.fetchall()]

    finally: 
        connection.close()



if __name__ == "__main__":

    if not DATABASE_FILE.exists():
        create_database()

    result = execute_select_query("""
        SELECT
            latitude,
            longitude,
            current_temperature_c,
            risk_classification,
            delay_probability
        FROM fleet
        LIMIT 5
        """)

    for row in result:
        print(row)