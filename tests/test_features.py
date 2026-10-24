"""اختبارات فيزياء الخصائص — نقطة الندى حجر الزاوية فلا تُترك بلا تحقق."""

import numpy as np
import pandas as pd
import pytest

from krp.config import load_config
from krp.features import (
    antoine_water_dewpoint_c,
    build_feature_frame,
    corrosion_indicators,
    dewpoint_at,
    water_mole_fraction,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _frame(n=5, **cols):
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(cols, index=idx)


def test_dewpoint_matches_steam_tables():
    """760 mmHg تعطي 100°م، و200 mmHg نحو 66.3°م، و92.5 نحو 50°م."""
    got = antoine_water_dewpoint_c([760.0, 200.0, 92.5])
    assert got[0] == pytest.approx(100.0, abs=0.3)
    assert got[1] == pytest.approx(66.3, abs=0.6)
    assert got[2] == pytest.approx(50.0, abs=0.6)


def test_dewpoint_rejects_invalid_pressure():
    assert np.isnan(antoine_water_dewpoint_c([0.0, -5.0, np.nan])).all()


def test_mole_fraction_rises_with_stripping_steam():
    df = _frame(2, stripping_steam_rate=[2000.0, 6000.0],
                atm_ovhd_vapor_rate=[40000.0, 40000.0])
    y = water_mole_fraction(df)
    assert y.iloc[1] > y.iloc[0]
    assert (y.between(0, 1)).all()


def test_hot_drum_dewpoint_below_design_temperature():
    """الحوض الساخن 01-V-003 يعمل تصميماً فوق نقطة الندى — وإلا تكثّف الحمض."""
    df = _frame(1, stripping_steam_rate=[3200.0], atm_ovhd_vapor_rate=[37000.0],
                hot_drum_pressure=[1.2], hot_drum_temp=[138.0])
    dew = dewpoint_at(df, "hot_drum_pressure").iloc[0]
    assert 70.0 < dew < 120.0            # مدى معقول فيزيائياً
    assert df["hot_drum_temp"].iloc[0] - dew > 14.0


def test_dewpoint_missing_pressure_column_is_nan():
    df = _frame(3, stripping_steam_rate=3000.0, atm_ovhd_vapor_rate=40000.0)
    assert dewpoint_at(df, "hot_drum_pressure").isna().all()


def test_corrosion_indicators_cover_unit_design(cfg):
    """المؤشرات تغطي نظام الحوضين ومرحلتَي الديسولتر ونقاط الحقن الثلاث."""
    df = _frame(
        5,
        stripping_steam_rate=3200.0, atm_ovhd_vapor_rate=37000.0,
        atm_ovhd_pressure=1.45, atm_ovhd_temp=172.0,
        hot_drum_pressure=1.2, hot_drum_temp=138.0,
        crude_feed_rate=900.0, salt_in_crude_ptb=1.5,
        wash_water_rate=36.0, wash_water_recycle_rate=13.5,
        desalter1_voltage=26.0, desalter1_dp=0.8,
        desalter2_voltage=24.0, desalter2_dp=0.9,
        demulsifier1_rate=6.0, demulsifier2_rate=5.0,
        cdu_sour_water_ph=5.0, cdu_sour_water_chloride=40.0,
        neutralizer_tower_rate=12.0, neutralizer_drum_rate=10.0,
        filmer_rate=9.0, corrosion_inhibitor_rate=14.0,
        crude_tan=1.0, vac_transfer_line_temp=404.0,
        crude_sulfur=2.0, furnace_coil_outlet_temp=355.0, vac_heater_cot=407.0,
    )
    ci = corrosion_indicators(df, cfg)
    expected = {
        "hot_drum_dewpoint_c", "hot_drum_margin_c", "hot_drum_margin_breach",
        "ovhd_margin_c", "chloride_load", "wash_water_ratio_pct",
        "desalter1_health", "desalter2_health", "demulsifier_dose_ppm",
        "ph_below_band", "ph_above_band", "neutralizer_total_rate",
        "neutralizer_per_chloride", "filmer_per_chloride",
        "inhibitor_per_chloride", "nac_severity",
        "sulfidic_severity_atm", "sulfidic_severity_vac",
    }
    assert expected <= set(ci.columns)

    # المحايدان يُجمعان لأنهما يعملان على الحموضة نفسها
    assert ci["neutralizer_total_rate"].iloc[0] == pytest.approx(22.0)
    # الحموضة 5.0 تحت النطاق [5.5, 6.5] بمقدار 0.5، ولا تجاوز للحد الأعلى
    assert ci["ph_below_band"].iloc[0] == pytest.approx(0.5)
    assert ci["ph_above_band"].iloc[0] == 0.0
    # ماء الغسيل يشمل المُعاد تدويره: (36 + 13.5) / 900 = 5.5%
    assert ci["wash_water_ratio_pct"].iloc[0] == pytest.approx(5.5)
    # NAC تنشط فوق 220°م: 1.0 × (404 − 220) / 100
    assert ci["nac_severity"].iloc[0] == pytest.approx(1.84)


def test_hot_drum_breach_flag_triggers_below_threshold(cfg):
    """حرارة حوض منخفضة تُشعل علَم تجاوز هامش الأمان."""
    df = _frame(1, stripping_steam_rate=[3200.0], atm_ovhd_vapor_rate=[37000.0],
                hot_drum_pressure=[1.2], hot_drum_temp=[95.0])
    ci = corrosion_indicators(df, cfg)
    assert ci["hot_drum_margin_c"].iloc[0] < cfg["features"]["dewpoint_margin_alarm"]
    assert ci["hot_drum_margin_breach"].iloc[0] == 1.0


def test_target_is_shifted_into_the_future(cfg):
    """الهدف قيمة مستقبلية — وإلا فالنموذج يتنبأ بالحاضر."""
    idx = pd.date_range("2025-01-01", periods=300, freq="1h")
    rng = np.random.default_rng(0)
    target = cfg["model"]["target"]
    df = pd.DataFrame(
        {"crude_feed_rate": 900.0, "hot_drum_temp": 138.0, "hot_drum_pressure": 1.2,
         "stripping_steam_rate": 3200.0, "atm_ovhd_vapor_rate": 37000.0,
         target: rng.normal(2, 0.3, len(idx))},
        index=idx,
    )
    X, y = build_feature_frame(df, cfg)
    horizon = cfg["model"]["horizon_h"]
    assert y.name.endswith(f"plus{horizon}h")
    assert target not in X.columns          # لا تسريب مباشر للهدف
    ts = y.index[0]
    assert y.loc[ts] == pytest.approx(df[target].loc[ts + pd.Timedelta(hours=horizon)])


def test_missing_target_column_fails_loudly(cfg):
    """غياب عمود الهدف خطأ صريح لا نتيجة فارغة صامتة."""
    df = _frame(50, crude_feed_rate=900.0)
    with pytest.raises(ValueError, match=cfg["model"]["target"]):
        build_feature_frame(df, cfg)
