from __future__ import annotations
import pandas as pd
from csp.utils.diag import log_diag


def safe_reset_index(df: pd.DataFrame, name: str = "timestamp", overwrite: bool = True) -> pd.DataFrame:
    """
    Reset index into a column with a desired name without triggering
    'ValueError: cannot insert <name>, already exists'.

    - If the index already has a name and you pass a different 'name',
      we temporarily set index.name to that 'name'.
    - If a column with the same 'name' exists:
        * overwrite=True  -> drop that column first
        * overwrite=False -> keep it and name the index column '<name>_idx'
    """
    d = df.copy()
    # Decide the target name for the new column coming from the index
    target = name or (d.index.name if d.index.name is not None else "index")

    # If there is a collision with an existing column name
    if target in d.columns:
        if overwrite:
            log_diag(f"safe_reset_index: dropping existing column '{target}' to avoid collision")
            d = d.drop(columns=[target])
        else:
            target = f"{target}_idx"
            log_diag(f"safe_reset_index: renaming index column to '{target}' to avoid collision")

    # Set index name and reset
    d.index = pd.Index(d.index)  # ensure Index object
    d.index.name = target
    try:
        d = d.reset_index()
    except Exception as e:
        # Extra diagnostics to help debugging
        log_diag(f"safe_reset_index FAILED: idx.name={d.index.name}, cols={list(d.columns)}")
        raise
    return d
