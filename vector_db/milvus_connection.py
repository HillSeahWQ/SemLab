"""
Milvus connection management utilities.
"""
from pymilvus import connections
from utils.logger import setup_logger
from config import LOGGING

logger = setup_logger(__name__, **LOGGING)


def get_connection(
    alias: str = "default",
    host: str = "localhost",
    port: str = "19530",
    reset: bool = False
) -> str:
    """
    Get or create a Milvus connection.
    
    Parameters
    ----------
    alias : str
        Connection alias
    host : str
        Milvus host address
    port : str
        Milvus port
    reset : bool
        If True, disconnect existing connection and reconnect
        
    Returns
    -------
    str
        The alias name of the connection
    """
    if reset:
        disconnect(alias)
        logger.info(f"Reset connection: {alias}")
    
    # Check if connection exists and is active
    active_connections = {c[0]: c[1] for c in connections.list_connections()}
    
    if alias not in active_connections or not active_connections[alias]:
        connections.connect(alias=alias, host=host, port=port)
        logger.info(f"Connected to Milvus at {host}:{port} (alias: {alias})")
    else:
        logger.debug(f"Using existing connection: {alias}")
    
    return alias


def disconnect(alias: str = "default") -> None:
    """
    Disconnect from Milvus.
    
    Parameters
    ----------
    alias : str
        Connection alias to disconnect
    """
    active_connections = {c[0]: c[1] for c in connections.list_connections()}
    
    if alias in active_connections and active_connections[alias]:
        connections.disconnect(alias)
        logger.info(f"Disconnected from Milvus (alias: {alias})")