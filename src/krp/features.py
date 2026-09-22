"""هندسة الخصائص: مؤشرات تآكل مبنية على الفيزياء + خصائص زمنية.

المفتاح هنا أن النموذج لا يُغذّى بالقراءات الخام وحدها، بل بمؤشرات
تعكس آلية التآكل الفعلية — أهمها هامش نقطة ندى الماء في قمة البرج:
تكوّن أول قطرة ماء سائل بوجود HCl يعني حمضاً مركّزاً جداً (pH قد يصل 1)،
وهو ما يذيب الحديد من جدران الأنابيب فيظهر لاحقاً في تحليل ماء القمة.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MW_WATER = 18.015          # g/mol
MW_OVHD_HC = 100.0         # g/mol — متوسط تقريبي لبخار القمة (نافثا خفيفة)
BAR_TO_MMHG = 750.062
ATM_BAR = 1.01325


def antoine_water_dewpoint_c(p_mmhg: pd.Series | np.ndarray) -> np.ndarray:
    """درجة حرارة ندى الماء (°م) من ضغطه الجزئي بالمليمتر زئبق.

    معادلة أنطوان معكوسة، بمعاملين حسب المدى:
    تحت 760 mmHg (أقل من 100°م) وفوقها.
    """
    p = np.asarray(p_mmhg, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p) & (p > 0)
    if not valid.any():
        return out

    low = valid & (p <= 760.0)
    high = valid & (p > 760.0)
    # المدى 1–100°م
    out[low] = 1730.63 / (8.07131 - np.log10(p[low])) - 233.426
    # المدى 100–374°م
    out[high] = 1810.94 / (8.14019 - np.log10(p[high])) - 244.485
    return out


def water_dewpoint(df: pd.DataFrame) -> pd.Series:
    """حرارة ندى الماء في قمة البرج الجوي (°م).

    الضغط الجزئي للماء = الضغط الكلي × الكسر المولي للبخار المائي،
    والكسر المولي يُحسب من بخار التعرية مقابل بخار الهيدروكربون.
    """
    needed = {"stripping_steam_rate", "ovhd_vapor_rate", "ovhd_pressure"}
    if not needed.issubset(df.columns):
        return pd.Series(np.nan, index=df.index, name="water_dewpoint_c")

    n_water = df["stripping_steam_rate"] / MW_WATER
    n_hc = df["ovhd_vapor_rate"] / MW_OVHD_HC
    total = n_water + n_hc
    y_water = (n_water / total).where(total > 0)

    p_total_mmhg = (df["ovhd_pressure"] + ATM_BAR) * BAR_TO_MMHG
    p_water = p_total_mmhg * y_water

    return pd.Series(
        antoine_water_dewpoint_c(p_water), index=df.index, name="water_dewpoint_c"
    )


def corrosion_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """مؤشرات التآكل المشتقة — هذه هي الخصائص ذات المعنى الهندسي."""
    fcfg = cfg["features"]
    ph_lo, ph_hi = fcfg["ph_target_band"]
    out = pd.DataFrame(index=df.index)

    # --- 1. هامش نقطة الندى: أهم منبئ بالتكثّف الحمضي المبكر ---
    dew = water_dewpoint(df)
    out["water_dewpoint_c"] = dew
    if "ovhd_temp" in df.columns:
        out["dewpoint_margin_c"] = df["ovhd_temp"] - dew
        out["dewpoint_margin_breach"] = (
            out["dewpoint_margin_c"] < fcfg["dewpoint_margin_alarm"]
        ).astype(float)

    # --- 2. حِمل الكلورايد الداخل: الملح المتبقي مضروباً بالتغذية ---
    if {"salt_in_crude_ptb", "crude_feed_rate"}.issubset(df.columns):
        out["chloride_load"] = df["salt_in_crude_ptb"] * df["crude_feed_rate"] / 1000.0

    # --- 3. أداء الديسولتر ---
    if {"wash_water_rate", "crude_feed_rate"}.issubset(df.columns):
        out["wash_water_ratio_pct"] = (
            100.0 * df["wash_water_rate"] / df["crude_feed_rate"].replace(0, np.nan)
        )

    # --- 4. انحراف الحموضة عن النطاق المستهدف (الاتجاه الحمضي فقط يضر) ---
    if "ovhd_water_ph" in df.columns:
        ph = df["ovhd_water_ph"]
        out["ph_below_band"] = (ph_lo - ph).clip(lower=0)
        out["ph_above_band"] = (ph - ph_hi).clip(lower=0)

    # --- 5. كفاية جرعة المحايد مقابل الكلورايد الفعلي ---
    if {"neutralizer_rate", "ovhd_water_chloride"}.issubset(df.columns):
        out["neutralizer_per_chloride"] = df["neutralizer_rate"] / (
            df["ovhd_water_chloride"].replace(0, np.nan)
        )

    # --- 6. شدة تآكل الأحماض النفثينية في الـ VDU ---
    # تنشط فوق ~220°م وتشتد مع الرقم الحمضي
    if {"crude_tan", "vac_transfer_line_temp"}.issubset(df.columns):
        excess_t = (df["vac_transfer_line_temp"] - 220.0).clip(lower=0)
        out["nac_severity"] = df["crude_tan"] * excess_t / 100.0

    # --- 7. شدة التآكل الكبريتي في منطقة الفرن ---
    if {"crude_sulfur", "furnace_coil_outlet_temp"}.issubset(df.columns):
        excess_t = (df["furnace_coil_outlet_temp"] - 260.0).clip(lower=0)
        out["sulfidic_severity"] = df["crude_sulfur"] * excess_t / 100.0

    return out


def add_time_features(df: pd.DataFrame, columns: list[str], lags_h: list[int],
                      windows_h: list[int], freq: str = "1h") -> pd.DataFrame:
    """يضيف قيماً مُزاحة زمنياً ومتوسطات/انحرافات متدحرجة.

    التآكل عملية تراكمية: المتوسط المتدحرج لـ 72 ساعة غالباً أقوى تنبؤياً
    من القراءة اللحظية، لأن الحديد يعكس تعرّضاً ممتداً لا لحظة واحدة.
    """
    steps_per_hour = pd.Timedelta(hours=1) / pd.Timedelta(freq)
    out = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        for lag in lags_h:
            out[f"{col}__lag{lag}h"] = s.shift(int(lag * steps_per_hour))
        for win in windows_h:
            w = max(2, int(win * steps_per_hour))
            out[f"{col}__mean{win}h"] = s.rolling(w, min_periods=max(2, w // 3)).mean()
            out[f"{col}__std{win}h"] = s.rolling(w, min_periods=max(2, w // 3)).std()
            out[f"{col}__slope{win}h"] = s.diff(w) / win
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1, sort=False)


def build_feature_frame(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """يبني مصفوفة الخصائص والمتغير الهدف المُزاح إلى المستقبل.

    الهدف هو قيمة الحديد بعد `horizon_h` ساعة — لذلك أي خاصية تُحسب
    من بيانات اللحظة الحالية فقط، ولا يُستخدم الحديد الحالي كخاصية
    إلا بصيغة مُزاحة للماضي (وهو مسموح ومفيد: الاتجاه السابق منبئ).
    """
    fcfg = cfg["features"]
    freq = cfg["data"]["resample_freq"]
    target_col = cfg["model"]["target"]
    horizon = cfg["model"]["horizon_h"]

    base = pd.concat([df, corrosion_indicators(df, cfg)], axis=1, sort=False)

    time_cols = [c for c in base.columns if c != target_col]
    # الهدف نفسه يدخل كخاصية ماضية فقط (إزاحات موجبة = قيم سابقة)
    feats = add_time_features(base, time_cols, fcfg["lags_h"], fcfg["rolling_windows_h"], freq)
    if target_col in base.columns:
        feats = add_time_features(
            feats, [target_col], fcfg["lags_h"], fcfg["rolling_windows_h"], freq
        )
        feats = feats.drop(columns=[target_col])

    steps = int(horizon * (pd.Timedelta(hours=1) / pd.Timedelta(freq)))
    y = df[target_col].shift(-steps).rename(f"{target_col}__plus{horizon}h")

    idx = y.dropna().index.intersection(feats.dropna(how="all").index)
    return feats.loc[idx], y.loc[idx]
