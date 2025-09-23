def standardize_columns(df, rename_map: dict):
    """
    Rename columns using a provided mapping. Returns:
      df2: DataFrame with renamed columns
      applied: {old_name: new_name} actually applied (present in df and different)
    """
    applied = {
        old: new for old, new in rename_map.items()
        if old in df.columns and old != new
    }
    df2 = df.rename(columns=applied)
    return df2, applied
