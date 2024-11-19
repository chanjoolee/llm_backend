import requests
from langchain_core.tools import tool

@tool
def get_failed_flink_logs(cluster: str, name: str):
    """
    Get the failed Flink logs for the given Flink application.

    Args:
        cluster (str): The name of the Flink cluster.
        name (str): The name of the Flink application.
    """

    endpoint = "http://swm-02-01:8081/flink/logs"
    result = requests.get(endpoint, params={"cluster": cluster, "name": name})

    return result.text

