"""هندسة الخصائص: مؤشرات تآكل مبنية على تصميم الوحدة الفعلي + خصائص زمنية.

تصميم قمة الوحدة بحوضين (راجع IONE-AA01-PFD-0010-001 ورقة 2 و3):

    01-T-001 ──172°م──► 01-E-003 ──► 01-V-003 (الحوض الساخن، 138°م)
                                        │  ← حقن الأمين الفيلمي
                                        ▼
                         01-EA-004 ──► 01-E-020A/B ──► 01-V-005 (38°م)
                                                          │
                                                   الماء الحامض ──► SWS

الحوض الساخن مصمَّم ليبقى **فوق** نقطة ندى الماء فلا يتكوّن طور مائي فيه.
لذلك المؤشر الحرج ليس حرارة قمة البرج بل **هامش الحوض الساخن**: الفرق بين
حرارته ونقطة ندى الماء عند ضغطه. نزول هذا الهامش يعني أول قطرة ماء سائل
بوجود HCl — أي حمضاً شديد التركيز على جدار الحوض وأنابيبه، وهو ما يذيب
الحديد فيظهر لاحقاً في تحليل الماء الحامض المسحوب من 01-V-005.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MW_WATER = 18.015          # g/mol
MW_OVHD_HC = 100.0         # g/mol — متوسط تقريبي لبخار القمة (نافثا خفيفة)
BAR_TO_MMHG = 750.062
ATM_BAR = 1.01325

# عتبات تنشيط آليات التآكل عالي الحرارة (°م)
NAC_ONSET_C = 220.0        # الأحماض النفثينية
SULFIDIC_ONSET_C = 260.0   # التآكل الكبريتي


def antoine_water_dewpoint_c(p_mmhg: pd.Series | np.ndarray) -> np.ndarray:
    """درجة حرارة ندى الماء (°م) من ضغطه الجزئي بالمليمتر زئبق.

    معادلة أنطوان معكوسة، بمعاملين حسب المدى: تحت 760 mmHg (أقل من 100°م)
    وفوقها.
    """
    p = np.asarray(p_mmhg, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.isfinite(p) & (p > 0)
    if not valid.any():
        return out

    low = valid & (p <= 760.0)
    high = valid & (p > 760.0)
    out[low] = 1730.63 / (8.07131 - np.log10(p[low])) - 233.426     # 1–100°م
    out[high] = 1810.94 / (8.14019 - np.log10(p[high])) - 244.485   # 100–374°م
    return out


def water_mole_fraction(df: pd.DataFrame) -> pd.Series:
    """الكسر المولي للبخار المائي في بخار القمة.

    يُحسب من بخار التعرية الداخل إلى قاع البرج مقابل بخار الهيدروكربون
    الخارج من القمة. البخار لا يتكثّف قبل الحوض الساخن فيبقى الكسر ثابتاً
    عبر 01-E-003.
    """
    if not {"stripping_steam_rate", "atm_ovhd_vapor_rate"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index, name="water_mole_fraction")
    n_water = df["stripping_steam_rate"] / MW_WATER
    n_hc = df["atm_ovhd_vapor_rate"] / MW_OVHD_HC
    total = n_water + n_hc
    return (n_water / total).where(total > 0).rename("water_mole_fraction")


def dewpoint_at(df: pd.DataFrame, pressure_col: str) -> pd.Series:
    """نقطة ندى الماء (°م) عند ضغط النقطة المطلوبة من الوحدة."""
    if pressure_col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    y_water = water_mole_fraction(df)
    p_total_mmhg = (df[pressure_col] + ATM_BAR) * BAR_TO_MMHG
    return pd.Series(
        antoine_water_dewpoint_c(p_total_mmhg * y_water), index=df.index
    )


def corrosion_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """مؤشرات التآكل المشتقة — الخصائص ذات المعنى الهندسي في هذه الوحدة."""
    fcfg = cfg["features"]
    ph_lo, ph_hi = fcfg["ph_target_band"]
    out = pd.DataFrame(index=df.index)

    # --- 1. هامش الحوض الساخن 01-V-003: المؤشر الحرج الأول ---
    dew_hot = dewpoint_at(df, "hot_drum_pressure")
    out["hot_drum_dewpoint_c"] = dew_hot
    if "hot_drum_temp" in df.columns:
        out["hot_drum_margin_c"] = df["hot_drum_temp"] - dew_hot
        out["hot_drum_margin_breach"] = (
            out["hot_drum_margin_c"] < fcfg["dewpoint_margin_alarm"]
        ).astype(float)

    # --- 2. هامش قمة البرج: نذير مبكر لانهيار هامش الحوض ---
    dew_top = dewpoint_at(df, "atm_ovhd_pressure")
    if "atm_ovhd_temp" in df.columns:
        out["ovhd_margin_c"] = df["atm_ovhd_temp"] - dew_top

    # --- 3. حِمل الكلورايد الداخل من الملح المتبقي ---
    if {"salt_in_crude_ptb", "crude_feed_rate"}.issubset(df.columns):
        out["chloride_load"] = df["salt_in_crude_ptb"] * df["crude_feed_rate"] / 1000.0

    # --- 4. أداء مرحلتَي إزالة الملح ---
    # الكفاءتان تتضاعفان، فتدهور إحداهما قد يختفي خلف الأخرى في الناتج
    # النهائي. مؤشرا الجهد وفرق الضغط لكل مرحلة يكشفان أيّهما المتدهورة.
    if {"wash_water_rate", "crude_feed_rate"}.issubset(df.columns):
        feed = df["crude_feed_rate"].replace(0, np.nan)
        wash = df["wash_water_rate"]
        if "wash_water_recycle_rate" in df.columns:
            wash = wash + df["wash_water_recycle_rate"].fillna(0)
        out["wash_water_ratio_pct"] = 100.0 * wash / feed

    for stage in (1, 2):
        v, dp = f"desalter{stage}_voltage", f"desalter{stage}_dp"
        if {v, dp}.issubset(df.columns):
            # الجهد العالي مع فرق ضغط منخفض = فصل سليم؛ انقلاب النسبة
            # يشير إلى استحلاب أو ترسّب داخل الوعاء
            out[f"desalter{stage}_health"] = df[v] / (df[dp].replace(0, np.nan) + 0.1)

    if {"demulsifier1_rate", "demulsifier2_rate", "crude_feed_rate"}.issubset(df.columns):
        out["demulsifier_dose_ppm"] = (
            1000.0 * (df["demulsifier1_rate"] + df["demulsifier2_rate"])
            / df["crude_feed_rate"].replace(0, np.nan)
        )

    # --- 5. انحراف الحموضة عن النطاق المستهدف (الاتجاه الحمضي وحده يضر) ---
    if "cdu_sour_water_ph" in df.columns:
        ph = df["cdu_sour_water_ph"]
        out["ph_below_band"] = (ph_lo - ph).clip(lower=0)
        out["ph_above_band"] = (ph - ph_hi).clip(lower=0)

    # --- 6. كفاية المعالجة الكيميائية مقابل الكلورايد الفعلي ---
    # ثلاث نقاط حقن في هذه الوحدة: محايد عند البرج، محايد عند الحوض
    # الساخن، وفيلمي على بخار الخروج منه. المحايدان يعملان على الحموضة
    # نفسها فيُجمعان؛ والفيلمي يعمل بآلية أخرى (طبقة عازلة) فيبقى مستقلاً.
    if "cdu_sour_water_chloride" in df.columns:
        cl = df["cdu_sour_water_chloride"].replace(0, np.nan)
        neutralizer = pd.Series(0.0, index=df.index)
        for col in ("neutralizer_tower_rate", "neutralizer_drum_rate"):
            if col in df.columns:
                neutralizer = neutralizer + df[col].fillna(0)
        out["neutralizer_total_rate"] = neutralizer
        out["neutralizer_per_chloride"] = neutralizer / cl
        if "filmer_rate" in df.columns:
            out["filmer_per_chloride"] = df["filmer_rate"] / cl
        if "corrosion_inhibitor_rate" in df.columns:
            out["inhibitor_per_chloride"] = df["corrosion_inhibitor_rate"] / cl

    # --- 7. شدة تآكل الأحماض النفثينية في الـ VDU ---
    if {"crude_tan", "vac_transfer_line_temp"}.issubset(df.columns):
        excess = (df["vac_transfer_line_temp"] - NAC_ONSET_C).clip(lower=0)
        out["nac_severity"] = df["crude_tan"] * excess / 100.0

    # --- 8. شدة التآكل الكبريتي في منطقتَي الفرن ---
    for name, col in (("atm", "furnace_coil_outlet_temp"), ("vac", "vac_heater_cot")):
        if {"crude_sulfur", col}.issubset(df.columns):
            excess = (df[col] - SULFIDIC_ONSET_C).clip(lower=0)
            out[f"sulfidic_severity_{name}"] = df["crude_sulfur"] * excess / 100.0

    return out


def add_time_features(df: pd.DataFrame, columns: list[str], lags_h: list[int],
                      windows_h: list[int], freq: str = "1h") -> pd.DataFrame:
    """يضيف قيماً مُزاحة زمنياً ومتوسطات/انحرافات/ميولاً متدحرجة.

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

    الهدف هو قيمة الحديد بعد `horizon_h` ساعة. كل خاصية تُحسب من بيانات
    اللحظة الحالية أو ما قبلها فقط؛ والحديد نفسه يدخل بصيغة مُزاحة للماضي
    وحدها — وهو مسموح ومفيد، إذ اتجاهه السابق منبئ قوي.
    """
    fcfg = cfg["features"]
    freq = cfg["data"]["resample_freq"]
    target_col = cfg["model"]["target"]
    horizon = cfg["model"]["horizon_h"]

    if target_col not in df.columns:
        raise ValueError(
            f"عمود الهدف '{target_col}' غير موجود في البيانات. "
            f"الأعمدة المتاحة: {sorted(df.columns)[:12]}…"
        )

    base = pd.concat([df, corrosion_indicators(df, cfg)], axis=1, sort=False)

    time_cols = [c for c in base.columns if c != target_col]
    feats = add_time_features(base, time_cols, fcfg["lags_h"], fcfg["rolling_windows_h"], freq)
    feats = add_time_features(
        feats, [target_col], fcfg["lags_h"], fcfg["rolling_windows_h"], freq
    ).drop(columns=[target_col])

    steps = int(horizon * (pd.Timedelta(hours=1) / pd.Timedelta(freq)))
    y = df[target_col].shift(-steps).rename(f"{target_col}__plus{horizon}h")

    idx = y.dropna().index.intersection(feats.dropna(how="all").index)
    return feats.loc[idx], y.loc[idx]
