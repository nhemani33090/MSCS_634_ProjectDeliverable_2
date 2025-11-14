from typing import Tuple, List
import numpy as np
import pandas as pd

def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    pct = (counts / len(df)) * 100
    return pd.DataFrame({'missing_count': counts, 'missing_pct': pct}).sort_values('missing_count', ascending=False)

def iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (q1 - k*iqr, q3 + k*iqr)

def clip_outliers_iqr(df: pd.DataFrame, cols: List[str], k: float = 1.5) -> pd.DataFrame:
    for c in cols:
        lower, upper = iqr_bounds(df[c].dropna(), k=k)
        df[c] = df[c].clip(lower=lower, upper=upper)
    return df