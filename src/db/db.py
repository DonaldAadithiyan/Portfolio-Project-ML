import os
import logging
from supabase import create_client, Client

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
        _client = create_client(url, key)
        logger.info("[DB] Supabase client initialised")
    return _client


def check_schema() -> bool:
    """Return True if tc_depots table exists and is accessible."""
    try:
        sb = get_client()
        sb.table("tc_depots").select("depot_id").limit(1).execute()
        return True
    except Exception:
        return False
