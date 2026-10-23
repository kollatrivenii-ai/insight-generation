"""
Insight Generation - simple base version
Usage:
    python insight_generator.py data.csv          # analyse your own CSV
    python insight_generator.py                   # run on built-in demo data
Requires: pandas, numpy   (pip install pandas numpy)
"""
import sys
import numpy as np
import pandas as pd
from pandas.api import types as pt


def cat_cols(df):
    return [c for c in df.columns if pt.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]


# ---------- Insight helpers: each returns a list of (score, text) ----------

def overview(df):
    out = [(100, f"Dataset has {len(df):,} rows and {df.shape[1]} columns.")]
    miss = df.isna().mean()
    for col, pct in miss[miss > 0].sort_values(ascending=False).items():
        out.append((60 + pct * 40, f"'{col}' has {pct:.1%} missing values."))
    dup = df.duplicated().sum()
    if dup:
        out.append((70, f"{dup} duplicate rows found."))
    return out


def numeric_insights(df):
    out = []
    for col in df.select_dtypes("number"):
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        if n_out:
            out.append((50 + min(n_out / len(s) * 100, 30),
                        f"'{col}' has {n_out} outliers (IQR rule); max={s.max():,.2f}, median={s.median():,.2f}."))
        skew = s.skew()
        if abs(skew) > 1:
            side = "right" if skew > 0 else "left"
            out.append((45, f"'{col}' is strongly {side}-skewed (skew={skew:.2f}); mean {s.mean():,.2f} vs median {s.median():,.2f}."))
    return out


def correlation_insights(df, threshold=0.6):
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return []
    corr = num.corr()
    out, cols = [], corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                kind = "positively" if r > 0 else "negatively"
                out.append((70 + abs(r) * 20, f"'{cols[i]}' and '{cols[j]}' are strongly {kind} correlated (r={r:.2f})."))
    return out


def categorical_insights(df):
    out = []
    for col in cat_cols(df):
        vc = df[col].value_counts(normalize=True)
        if 1 < len(vc) <= 50 and vc.iloc[0] > 0.4:
            out.append((55 + vc.iloc[0] * 20, f"'{col}': '{vc.index[0]}' dominates with {vc.iloc[0]:.1%} of records."))
    return out


def group_insights(df):
    """Which category splits a numeric column most strongly?"""
    out = []
    cats = [c for c in cat_cols(df) if 1 < df[c].nunique() <= 15]
    for num in df.select_dtypes("number"):
        for cat in cats:
            means = df.groupby(cat)[num].mean().dropna()
            if len(means) < 2 or means.mean() == 0:
                continue
            gap = (means.max() - means.min()) / abs(means.mean())
            if gap > 0.3:
                out.append((50 + min(gap * 20, 35),
                            f"Average '{num}' is highest for {cat}='{means.idxmax()}' ({means.max():,.2f}) "
                            f"and lowest for '{means.idxmin()}' ({means.min():,.2f})."))
    return out


def trend_insights(df):
    out = []
    date_cols = [c for c in df.columns if pt.is_datetime64_any_dtype(df[c])]
    for c in cat_cols(df):  # try parsing date-like strings
        if "date" in c.lower() or "time" in c.lower():
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[c] = parsed
                date_cols.append(c)
    for d in date_cols:
        for num in df.select_dtypes("number"):
            t = df[[d, num]].dropna().sort_values(d)
            if len(t) < 10:
                continue
            x = (t[d] - t[d].min()).dt.days.values
            slope = np.polyfit(x, t[num].values, 1)[0]
            change = slope * (x.max() - x.min()) / (abs(t[num].mean()) or 1)
            if abs(change) > 0.1:
                word = "increased" if slope > 0 else "decreased"
                out.append((65 + min(abs(change) * 20, 25), f"'{num}' has {word} by ~{abs(change):.0%} over the period in '{d}'."))
    return out


# ---------- Main pipeline ----------

def generate_insights(df, top_n=15):
    insights = []
    for fn in (overview, numeric_insights, correlation_insights,
               categorical_insights, group_insights, trend_insights):
        insights += fn(df)
    insights.sort(key=lambda x: -x[0])
    return [text for _, text in insights[:top_n]]


def demo_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.5, 0.2, 0.2, 0.1])
    ad = rng.gamma(2, 500, n)
    sales = ad * 3 + (region == "North") * 800 + np.arange(n) * 2 + rng.normal(0, 400, n)
    return pd.DataFrame({"date": dates, "region": region, "ad_spend": ad, "sales": sales})


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else demo_data()
    print("\n=== KEY INSIGHTS ===")
    for i, text in enumerate(generate_insights(data), 1):
        print(f"{i:>2}. {text}")
