"""قراءة بيانات الوحدة من ملفات التصدير ومواءمة معدلات أخذ العينات.

بيانات الـ DCS لحظية (دقيقة/دقائق) بينما نتائج المختبر يومية أو أقل.
هذه الوحدة توحّدها على شبكة زمنية واحدة: متوسط للقياسات اللحظية،
وسحب أمامي محدود المدة للقراءات المختبرية حتى لا تُستنسخ عينة قديمة إلى الأبد.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TagMap


def read_export(path: str | Path, tag_map: TagMap) -> pd.DataFrame:
    """يقرأ ملف تصدير (CSV/Excel) ويعيد تسمية أعمدته إلى الأسماء الموحّدة."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    ts_col = tag_map.timestamp_column
    if ts_col not in df.columns:
        raise ValueError(
            f"عمود الوقت '{ts_col}' غير موجود في {path.name}. الأعمدة المتاحة: {list(df.columns)[:10]}"
        )

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()

    df = df.rename(columns=tag_map.rename_map())
    known = [c for c in tag_map.by_canonical() if c in df.columns]
    return df[known].apply(pd.to_numeric, errors="coerce")


def align_sampling(df: pd.DataFrame, tag_map: TagMap, freq: str = "1h",
                   lab_max_staleness_h: int = 24) -> pd.DataFrame:
    """يوحّد السلاسل على شبكة زمنية واحدة.

    - وسوم الـ DCS: متوسط داخل كل فترة.
    - وسوم المختبر: آخر قراءة داخل الفترة، ثم سحب أمامي محدود بـ
      `lab_max_staleness_h` ساعة فقط — تجاوز ذلك يترك القيمة فارغة عمداً.
    """
    if df.empty:
        return df

    dcs_cols = [c for c in tag_map.dcs_tags if c in df.columns]
    lab_cols = [c for c in tag_map.lab_tags if c in df.columns]

    parts = []
    if dcs_cols:
        parts.append(df[dcs_cols].resample(freq).mean())
    if lab_cols:
        lab = df[lab_cols].resample(freq).last()
        limit = max(1, int(pd.Timedelta(hours=lab_max_staleness_h) / pd.Timedelta(freq)))
        parts.append(lab.ffill(limit=limit))

    out = pd.concat(parts, axis=1, sort=False) if parts else pd.DataFrame(index=df.index)
    return out[[c for c in df.columns if c in out.columns]]


def load_unit_data(path: str | Path, cfg: dict) -> pd.DataFrame:
    """المسار الكامل: قراءة الملف ثم مواءمة معدلات أخذ العينات."""
    tag_map: TagMap = cfg["_tag_map"]
    raw = read_export(path, tag_map)
    return align_sampling(
        raw,
        tag_map,
        freq=cfg["data"]["resample_freq"],
        lab_max_staleness_h=cfg["data"]["lab_max_staleness_h"],
    )
