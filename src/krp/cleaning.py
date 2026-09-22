"""تنظيف بيانات الـ DCS: حدود فيزيائية، حسّاسات مجمّدة، شواذ، وحالة التشغيل المستقر.

هذه الخطوات هي ما يُفشل معظم نماذج المصافي عند إهمالها:
حسّاس معطوب يعطي قيمة ثابتة تماماً، أو فترة إيقاف/بدء تُدرَّب عليها
فيتعلّم النموذج سلوكاً لا علاقة له بالتشغيل الطبيعي.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import TagMap


def apply_physical_limits(df: pd.DataFrame, tag_map: TagMap) -> pd.DataFrame:
    """يُفرغ أي قيمة خارج المدى الفيزيائي المعقول للوسم."""
    out = df.copy()
    specs = tag_map.by_canonical()
    for col in out.columns:
        spec = specs.get(col)
        if spec is None:
            continue
        lo, hi = spec.limits
        out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan
    return out


def flag_frozen(series: pd.Series, min_repeats: int = 6) -> pd.Series:
    """يعلّم القراءات المتطابقة تماماً المتتالية كقيم حسّاس مجمّد.

    القيمة المتطابقة بدقة تامة لعدة ساعات ليست استقراراً بل حسّاس معلّق؛
    الضوضاء الطبيعية تمنع التطابق التام في عملية حقيقية.
    """
    if series.dropna().empty:
        return pd.Series(False, index=series.index)
    # مجموعات متتالية من نفس القيمة
    group = (series.diff() != 0).cumsum()
    sizes = series.groupby(group).transform("size")
    return (sizes >= min_repeats) & series.notna()


def remove_frozen(df: pd.DataFrame, min_repeats: int = 6,
                  skip: Sequence[str] = ()) -> pd.DataFrame:
    """يزيل قيم الحسّاسات المجمّدة من أعمدة الـ DCS فقط.

    أعمدة المختبر تُستثنى عبر `skip`: ثباتها بين عيّنتين نتيجة السحب
    الأمامي المقصود، لا عطل حسّاس.
    """
    out = df.copy()
    skip = set(skip)
    for col in out.columns:
        if col in skip:
            continue
        out.loc[flag_frozen(out[col], min_repeats), col] = np.nan
    return out


def remove_outliers_mad(df: pd.DataFrame, sigma: float = 6.0,
                        skip: Sequence[str] = ()) -> pd.DataFrame:
    """استبعاد الشواذ بانحراف مطلق متوسط (MAD) — أمتن من الانحراف المعياري.

    السحب الأمامي لقراءات المختبر يضخّم كثافة القيم حول الوسيط ويجعل MAD
    صغيراً بغير وجه حق، فتُستثنى تلك الأعمدة عبر `skip`.
    """
    out = df.copy()
    skip = set(skip)
    for col in out.columns:
        if col in skip:
            continue
        s = out[col]
        med = s.median()
        mad = (s - med).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        # 1.4826 يحوّل MAD إلى ما يعادل الانحراف المعياري لتوزيع طبيعي
        out.loc[(s - med).abs() > sigma * 1.4826 * mad, col] = np.nan
    return out


def steady_state_mask(df: pd.DataFrame, tag: str, threshold: float,
                      min_hours: int, freq: str = "1h") -> pd.Series:
    """يعيد قناعاً صحيحاً للفترات المستقرة فقط.

    الفترة مستقرة إذا بقي الوسم المرجعي (عادةً تغذية الخام) فوق العتبة
    لمدة متصلة لا تقل عن `min_hours` — بهذا تُستبعد التوقفات وفترات البدء.
    """
    if tag not in df.columns:
        return pd.Series(True, index=df.index)

    above = df[tag] >= threshold
    block = (above != above.shift()).cumsum()
    per_step_h = pd.Timedelta(freq) / pd.Timedelta(hours=1)
    run_hours = above.groupby(block).transform("size") * per_step_h
    return above & (run_hours >= min_hours)


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """يطبّق سلسلة التنظيف كاملة ويعيد البيانات مع تقرير موجز."""
    tag_map: TagMap = cfg["_tag_map"]
    ccfg = cfg["cleaning"]
    n_before = int(df.notna().sum().sum())

    out = apply_physical_limits(df, tag_map)
    n_limits = n_before - int(out.notna().sum().sum())

    out = remove_frozen(out, ccfg["frozen_min_repeats"], skip=tag_map.lab_tags)
    n_frozen = n_before - n_limits - int(out.notna().sum().sum())

    out = remove_outliers_mad(out, ccfg["outlier_sigma"], skip=tag_map.lab_tags)
    n_outliers = n_before - n_limits - n_frozen - int(out.notna().sum().sum())

    ss = ccfg["steady_state"]
    mask = steady_state_mask(
        out,
        tag=ss["tag"],
        threshold=ss["min_fraction_of_design"] * ss["design_value"],
        min_hours=ss["min_hours"],
        freq=cfg["data"]["resample_freq"],
    )
    out = out[mask]

    report = {
        "rows_in": len(df),
        "rows_out": len(out),
        "rows_dropped_unsteady": int((~mask).sum()),
        "values_dropped_limits": n_limits,
        "values_dropped_frozen": n_frozen,
        "values_dropped_outliers": n_outliers,
    }
    return out, report
