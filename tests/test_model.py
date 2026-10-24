"""اختبارات النمذجة: التقسيم الزمني، أهمية التبديل، وتقرير الانجراف."""

import numpy as np
import pandas as pd
import pytest

from krp.config import load_config
from krp.model import (
    build_candidates,
    classify_alarm,
    drift_report,
    extract_importances,
    time_split,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def fitted(cfg):
    """نموذج مُدرَّب على بيانات يقودها متغيّر واحد معلوم مسبقاً."""
    rng = np.random.default_rng(0)
    n, n_noise = 400, 30
    signal = rng.normal(0, 1, n)
    data = {"signal": signal}
    data.update({f"noise_{i}": rng.normal(0, 1, n) for i in range(n_noise)})
    X = pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=n, freq="1h"))
    y = pd.Series(3.0 * signal + rng.normal(0, 0.1, n), index=X.index, name="y")

    model = build_candidates(cfg)["random_forest"]
    cut = 300
    model.fit(X.iloc[:cut], y.iloc[:cut])
    return model, X.iloc[cut:], y.iloc[cut:]


def test_time_split_preserves_order(cfg):
    idx = pd.date_range("2025-01-01", periods=100, freq="1h")
    X = pd.DataFrame({"a": range(100)}, index=idx)
    y = pd.Series(range(100), index=idx)
    X_tr, X_te, y_tr, y_te = time_split(X, y, test_fraction=0.25)
    assert len(X_tr) == 75 and len(X_te) == 25
    # لا تداخل زمني: كل عيّنة اختبار لاحقة لكل عيّنة تدريب
    assert X_tr.index.max() < X_te.index.min()


def test_time_split_rejects_tiny_datasets():
    idx = pd.date_range("2025-01-01", periods=8, freq="1h")
    X = pd.DataFrame({"a": range(8)}, index=idx)
    with pytest.raises(ValueError):
        time_split(X, pd.Series(range(8), index=idx), test_fraction=0.25)


def test_permutation_importance_finds_the_real_driver(fitted):
    model, X_te, y_te = fitted
    imp = extract_importances(model, X_te, y_te, seed=1, n_repeats=3)
    assert imp.index[0] == "signal"
    assert imp.loc["signal"] > 0.5          # يهيمن على الضوضاء
    assert imp.sum() == pytest.approx(1.0)  # مطبَّعة


def test_importance_filtering_matches_unfiltered_leader(fitted):
    """الترشيح يقلّص التكلفة ولا يُسقط الخاصية الحقيقية."""
    model, X_te, y_te = fitted
    full = extract_importances(model, X_te, y_te, seed=1, n_repeats=3, max_features=None)
    filtered = extract_importances(model, X_te, y_te, seed=1, n_repeats=3, max_features=5)
    assert full.index[0] == filtered.index[0] == "signal"
    # الخصائص خارج الترشيح تُسجَّل صفراً لا قيمة مفقودة
    assert set(filtered.index) == set(X_te.columns)
    assert filtered.notna().all()


def test_drift_report_flags_shifted_target():
    train = pd.Series(np.linspace(0, 1, 100))
    test = pd.Series(np.linspace(5, 6, 50))      # خارج مدى التدريب تماماً
    rep = drift_report(train, test)
    assert rep["test_fraction_outside_train_range"] == 1.0
    assert rep["severe"]
    assert rep["mean_shift"] > 4

    same = drift_report(train, pd.Series(np.linspace(0.2, 0.8, 50)))
    assert same["test_fraction_outside_train_range"] == 0.0
    assert not same["severe"]


def test_alarm_thresholds(cfg):
    warn = cfg["alarm"]["iron_warning_ppm"]
    crit = cfg["alarm"]["iron_critical_ppm"]
    levels = classify_alarm(pd.Series([warn - 0.1, warn + 0.1, crit + 0.1]), cfg)
    assert list(levels) == ["طبيعي", "تنبيه", "حرج"]
