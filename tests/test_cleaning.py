"""اختبارات التنظيف — الحسّاس المجمّد وحالة التشغيل المستقر."""

import numpy as np
import pandas as pd
import pytest

from krp.cleaning import (
    apply_physical_limits,
    clean,
    flag_frozen,
    remove_frozen,
    remove_outliers_mad,
    steady_state_mask,
)
from krp.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _idx(n):
    return pd.date_range("2025-01-01", periods=n, freq="1h")


def test_flag_frozen_detects_stuck_sensor():
    s = pd.Series([1.0, 1.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0], index=_idx(9))
    flags = flag_frozen(s, min_repeats=6)
    assert flags.sum() == 6
    assert not flags.iloc[0] and not flags.iloc[-1]


def test_frozen_ignores_normal_variation():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(100, 1, 200), index=_idx(200))
    assert not flag_frozen(s, min_repeats=6).any()


def test_remove_frozen_skips_lab_columns():
    """قراءات المختبر ثابتة بين عيّنتين بحكم السحب الأمامي — لا تُمسح."""
    df = pd.DataFrame({"dcs_col": [7.0] * 10, "lab_col": [3.0] * 10}, index=_idx(10))
    out = remove_frozen(df, min_repeats=6, skip=["lab_col"])
    assert out["dcs_col"].isna().all()
    assert out["lab_col"].notna().all()


def test_physical_limits_clear_impossible_values(cfg):
    tag_map = cfg["_tag_map"]
    df = pd.DataFrame({"cdu_sour_water_ph": [6.0, 99.0, -3.0]}, index=_idx(3))
    out = apply_physical_limits(df, tag_map)
    assert out["cdu_sour_water_ph"].tolist()[0] == 6.0
    assert out["cdu_sour_water_ph"].isna().sum() == 2


def test_outlier_removal_targets_spikes():
    rng = np.random.default_rng(1)
    vals = rng.normal(50, 1, 300)
    vals[100] = 5000.0
    df = pd.DataFrame({"x": vals}, index=_idx(300))
    out = remove_outliers_mad(df, sigma=6.0)
    assert np.isnan(out["x"].iloc[100])
    assert out["x"].notna().sum() >= 295


def test_steady_state_requires_sustained_operation():
    """نبضة قصيرة فوق العتبة لا تُعدّ تشغيلاً مستقراً."""
    feed = np.concatenate([np.full(20, 900.0), np.full(3, 900.0)])
    feed[20:23] = 900.0
    vals = np.concatenate([np.full(20, 900.0), np.full(2, 100.0), np.full(3, 900.0)])
    df = pd.DataFrame({"crude_feed_rate": vals}, index=_idx(25))
    mask = steady_state_mask(df, "crude_feed_rate", threshold=540, min_hours=6)
    assert mask.iloc[:20].all()          # 20 ساعة متصلة مقبولة
    assert not mask.iloc[20:22].any()    # فترة التوقف مرفوضة
    assert not mask.iloc[22:].any()      # 3 ساعات فقط بعدها — أقل من الحد


def test_clean_reports_counts(cfg):
    idx = _idx(200)
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"crude_feed_rate": rng.normal(900, 5, 200),
         "ovhd_temp": rng.normal(126, 1, 200)},
        index=idx,
    )
    out, rep = clean(df, cfg)
    assert rep["rows_in"] == 200
    assert rep["rows_out"] == len(out)
    assert set(rep) >= {"values_dropped_limits", "values_dropped_frozen",
                        "values_dropped_outliers", "rows_dropped_unsteady"}
