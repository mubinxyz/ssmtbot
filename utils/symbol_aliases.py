def resolve_symbol(alias: str) -> str:
    """
    Maps user-friendly aliases to actual trading symbols.
    """
    alias_map = {
        "DXY": "USDX",
        "EU": "EURUSD",
        "GU": "GBPUSD",
        # Add more aliases as needed
    }
    return alias_map.get(alias.upper(), alias.upper())
