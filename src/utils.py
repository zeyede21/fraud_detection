import socket
import struct
import pandas as pd

def ip_to_int(ip):
    """Convert dotted IP string to integer, handling NaNs and invalid IPs."""
    if pd.isna(ip) or not isinstance(ip, str):
        return None
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except (socket.error, OSError):
        return None
