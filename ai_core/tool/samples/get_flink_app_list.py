from langchain_core.tools import tool

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

@tool
def get_flink_app_list(cluster: str):
    """
    Get a list of Flink applications running on the given cluster.

    Args:
        cluster (str): The name of the Flink cluster.
    """
    gql_query = gql("""
    query GetFlinkMetadata($cluster: String) {
        flinkMetadata(cluster: $cluster) {
            flinkJobMetadataList {
                name, applicationId, startedTime, trackingUrl, numRestarts, backpressured, inLineage, maxCmsTime, nrOfTaskManagers
            }
        }
    }
    """)

    endpoint = "http://swm-02-01:8081/graphql"
    transport = RequestsHTTPTransport(url=endpoint)
    client = Client(transport=transport, fetch_schema_from_transport=True)

    result = client.execute(gql_query, variable_values={"cluster": cluster})

    return str(result)

