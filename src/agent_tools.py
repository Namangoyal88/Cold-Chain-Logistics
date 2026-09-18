from langchain_core.tools import tool

from src.database import execute_select_query
from src.rag import search_sop, format_sop_results
from src.weather import get_weather_text
from src.config import MAX_DATABASE_ROWS


@tool
def query_fleet_database(sql_query: str) -> str:
    """Query the local cold-chain fleet database.

    Use this tool whenever information about shipments,
    vehicles, temperatures, locations, risks, delays,
    congestion or route conditions is required.

    The database contains the following important columns:

    timestamp
    latitude
    longitude
    current_temperature_c
    cargo_condition_code
    risk_classification
    delay_probability
    port_congestion_level
    route_risk_index
    fuel_consumption_rate
    eta_variation_hours
    traffic_congestion_level
    weather_condition_severity
    shipping_costs
    supplier_reliability_score
    lead_time_days
    historical_demand
    customs_clearance_time
    driver_behavior_score
    fatigue_monitoring_score
    disruption_likelihood_score
    delivery_time_deviation

    Only SELECT queries are allowed.
    """

    try:
        rows = execute_select_query(sql_query, max_rows = MAX_DATABASE_ROWS)
        if not rows:
            return (
                "No fleet records matched "
                "the requested criteria."
            )

        result = []

        for row in rows:
            result.append(str(row))
        return "\n".join(result)

    except Exception as error:
        return (f"Fleet database error: {error}")


@tool
def search_compliance_sop(query: str) -> str:
    """Search the cold-chain compliance SOP.
    Use this tool whenever you need company-specific
    rules about temperature limits, cold-chain breaches,
    port congestion, diversions, escalation or other
    operational procedures."""

    try:

        documents = search_sop(
            query
        )

        return format_sop_results(
            documents
        )

    except Exception as error:

        return (
            f"SOP retrieval error: {error}"
        )


@tool
def fetch_corridor_conditions(latitude: float, longitude: float) -> str:
    """
    Retrieve current weather conditions for a shipment
    location using the Open-Meteo API.

    Use this when environmental conditions could be
    relevant to a logistics incident.
    """

    return get_weather_text(latitude, longitude)


FDE_TOOLS = [query_fleet_database, fetch_corridor_conditions, search_compliance_sop]


if __name__ == "__main__":
    print("\n--- DATABASE TOOL ---")

    print(query_fleet_database.invoke(
            """
            SELECT
                latitude,
                longitude,
                current_temperature_c,
                risk_classification,
                delay_probability
            FROM fleet
            LIMIT 3
            """))
    print("\n--- WEATHER TOOL ---")

    print(fetch_corridor_conditions.invoke({"latitude": 33.77, "longitude": -118.19}))
    print("\n--- SOP TOOL ---")
    print(search_compliance_sop.invoke("What happens when fresh perishables exceed 4°C?"))