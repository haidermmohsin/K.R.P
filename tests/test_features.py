"""اختبارات فيزياء الخصائص — نقطة الندى هي حجر الزاوية فلا تُترك بلا تحقق."""

import numpy as np
import pandas as pd
import pytest

from krp.config import load_config
from krp.features import (
    antoine_water_dewpoint_c,
    build_feature_frame,
    corrosion_indicators,
    water_dewpoint,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_dewpoint_matches_steam_tables():
    """760 mmHg يجب أن تعطي 100°م، و200 mmHg نحو 66.3°م."""
    got = antoine_water_dewpoint_c([760.0, 200.0, 92.5])
    assert got[0] == pytest.approx(100.0, abs=0.3)
    assert got[1] == pytest.approx(66.3, abs=0.6)
    assert got[2] == pytest.approx(50.0, abs=0.6)


def test_dewpoint_rejects_invalid_pressure():
    got = antoine_water_dewpoint_c([0.0, -5.0, np.nan])
    assert np.isnan(got).all()


def test_dewpoint_rises_with_steam():
    """زيادة بخار التعرية ترفع الكسر المولي للماء فترتفع نقطة الندى."""
    idx = pd.date_range("2025-01-01", periods=2, freq="1h")
    df = pd.DataFrame(
        {"stripping_steam_rate": [2000.0, 6000.0],
         "ovhd_vapor_rate": [40000.0, 40000.0],
         "ovhd_pressure": [1.1, 1.1]},
        index=idx,
    )
    dew = water_dewpoint(df)
    assert dew.iloc[1] > dew.iloc[0]


def test_corrosion_indicators_present(cfg):
    idx = pd.date_range("2025-01-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {"stripping_steam_rate": 3000.0, "ovhd_vapor_rate": 40000.0,
         "ovhd_pressure": 1.1, "ovhd_temp": 126.0, "ovhd_water_ph": 5.0,
         "salt_in_crude_ptb": 10.0, "crude_feed_rate": 900.0,
         "wash_water_rate": 50.0, "neutralizer_rate": 20.0,
         "ovhd_water_chloride": 40.0},
        index=idx,
    )
    ci = corrosion_indicators(df, cfg)
    assert {"water_dewpoint_c", "dewpoint_margin_c", "chloride_load",
            "wash_water_ratio_pct", "ph_below_band"} <= set(ci.columns)
    # الحموضة 5.0 تقع تحت النطاق المستهدف [5.5, 6.5] بمقدار 0.5
    assert ci["ph_below_band"].iloc[0] == pytest.approx(0.5)
    assert ci["ph_above_band"].iloc[0] == 0.0


def test_target_is_shifted_into_the_future(cfg):
    """الهدف يجب أن يكون قيمة مستقبلية — وإلا فالنموذج يتنبأ بالحاضر."""
    idx = pd.date_range("2025-01-01", periods=300, freq="1h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"crude_feed_rate": 900.0, "ovhd_temp": 126.0, "ovhd_pressure": 1.1,
         "stripping_steam_rate": 3000.0, "ovhd_vapor_rate": 40000.0,
         "ovhd_water_iron": rng.normal(2, 0.3, len(idx))},
        index=idx,
    )
    X, y = build_feature_frame(df, cfg)
    horizon = cfg["model"]["horizon_h"]
    assert y.name.endswith(f"plus{horizon}h")
    # لا يجوز أن يظهر الهدف الخام بين الخصائص (تسريب مباشر)
    assert cfg["model"]["target"] not in X.columns
    # قيمة الهدف عند لحظة ما = قيمة الحديد بعد الأفق
    ts = y.index[0]
    assert y.loc[ts] == pytest.approx(df["ovhd_water_iron"].loc[ts + pd.Timedelta(hours=horizon)])
