"""تدريب وتقييم نموذج التنبؤ بالحديد.

القاعدة الحاكمة هنا: التقسيم زمني دائماً، لا عشوائي.
التقسيم العشوائي في سلسلة زمنية يسرّب المستقبل إلى التدريب فيعطي
دقة وهمية عالية تنهار عند التشغيل الفعلي.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:  # pragma: no cover - اعتماد اختياري
    HAS_LGBM = False


@dataclass
class Evaluation:
    name: str
    mae: float
    rmse: float
    r2: float
    cv_mae: float | None = None

    def as_row(self) -> dict:
        return {"model": self.name, "MAE": self.mae, "RMSE": self.rmse,
                "R2": self.r2, "CV_MAE": self.cv_mae}


@dataclass
class TrainResult:
    best_name: str
    best_model: Pipeline
    evaluations: list[Evaluation]
    importances: pd.Series
    predictions: pd.DataFrame
    drift: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame([e.as_row() for e in self.evaluations]).sort_values("MAE")


def time_split(X: pd.DataFrame, y: pd.Series, test_fraction: float):
    """تقسيم زمني: الأقدم للتدريب والأحدث للاختبار."""
    cut = int(len(X) * (1 - test_fraction))
    if cut < 10 or len(X) - cut < 5:
        raise ValueError(f"بيانات غير كافية للتقسيم الزمني (عدد الصفوف = {len(X)})")
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def drift_report(y_train: pd.Series, y_test: pd.Series) -> dict:
    """يقيس انجراف الهدف بين فترتي التدريب والاختبار.

    نماذج الأشجار (RandomForest / LightGBM) تتنبأ بمتوسطات أوراقها، فلا
    تتجاوز أبداً المدى الذي رأته في التدريب. إذا وقع جزء كبير من فترة
    الاختبار خارج ذلك المدى فالنموذج مكبوت سقفياً مهما بلغت جودته —
    والعلاج إعادة تدريب دورية أو نموذج خطي قادر على الاستقراء.
    """
    lo, hi = float(y_train.min()), float(y_train.max())
    outside = float(((y_test < lo) | (y_test > hi)).mean())
    return {
        "train_range": (lo, hi),
        "test_range": (float(y_test.min()), float(y_test.max())),
        "test_fraction_outside_train_range": outside,
        "mean_shift": float(y_test.mean() - y_train.mean()),
        "severe": outside > 0.25,
    }


def build_candidates(cfg: dict) -> dict[str, Pipeline]:
    """نماذج مرشّحة من الأبسط إلى الأقوى — الأساس أولاً دائماً."""
    seed = cfg["model"]["seed"]
    imp = lambda: SimpleImputer(strategy="median")

    models: dict[str, Pipeline] = {
        "baseline_mean": Pipeline([("imp", imp()), ("m", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([("imp", imp()), ("sc", StandardScaler()),
                           ("m", Ridge(alpha=5.0))]),
        "random_forest": Pipeline([("imp", imp()),
                                   ("m", RandomForestRegressor(
                                       n_estimators=300, min_samples_leaf=3,
                                       random_state=seed, n_jobs=-1))]),
    }
    if HAS_LGBM:
        p = cfg["model"]["lightgbm"]
        models["lightgbm"] = Pipeline([("imp", imp()), ("m", LGBMRegressor(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
            subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
            random_state=seed, n_jobs=-1, verbose=-1))])
    return models


def _metrics(name: str, y_true: pd.Series, y_pred: np.ndarray, cv_mae=None) -> Evaluation:
    err = y_true.to_numpy() - y_pred
    return Evaluation(
        name=name,
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        r2=float(r2_score(y_true, y_pred)),
        cv_mae=cv_mae,
    )


def model_internal_importances(model: Pipeline, feature_names: list[str]) -> pd.Series:
    """الأهمية الداخلية للنموذج — سريعة لكنها متحيّزة.

    أهمية الأشجار تَعُدّ مرات الانقسام، فتُضخّم الخصائص عالية التباين
    (وكثير منها ضوضاء) وتشتّت الأهمية بين خصائص مترابطة. تُستخدم هنا
    كبديل احتياطي فقط.
    """
    est = model.named_steps["m"]
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    else:
        return pd.Series(dtype=float)
    s = pd.Series(vals, index=feature_names)
    total = s.sum()
    return (s / total if total > 0 else s).sort_values(ascending=False)


def extract_importances(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                        seed: int = 42, n_repeats: int = 5,
                        max_features: int | None = 60) -> pd.Series:
    """أهمية التبديل على فترة الاختبار — المقياس الأمين للتفسير.

    تُخلط قيم كل خاصية عشوائياً ويُقاس مقدار تدهور الخطأ الفعلي. ما يرفع
    الخطأ حقاً هو المهم؛ والخاصية التي لا يضرّ خلطُها ليست منبئة مهما
    كثُر استخدامها في الانقسامات. هذا ما يُعرض على مهندس التشغيل.

    التكلفة خطية في عدد الخصائص: كل خاصية تتطلب `n_repeats` تنبؤاً كاملاً،
    فستّمئة خاصية تعني نحو ثلاثة آلاف تنبؤ. لذلك تُرشَّح أولاً بالأهمية
    الداخلية للنموذج — وهي مخزّنة مسبقاً فتكلفتها صفر — ويُبدَّل أعلى
    `max_features` منها فقط. الترشيح متساهل عمداً: الأهمية الداخلية متحيّزة
    لكنها نادراً ما تُنزل خاصيةً منبئة فعلاً إلى ما دون المرتبة الستين.
    مرّر `max_features=None` لتبديل كل الخصائص بلا ترشيح.
    """
    columns = list(X_test.columns)
    candidates = columns
    if max_features is not None and len(columns) > max_features:
        internal = model_internal_importances(model, columns)
        if not internal.empty:
            candidates = list(internal.head(max_features).index)

    base = mean_absolute_error(y_test, model.predict(X_test))
    rng = np.random.default_rng(seed)
    scores: dict[str, float] = {}

    for col in candidates:
        permuted = X_test.copy()
        original = permuted[col].to_numpy()
        drops = []
        for _ in range(n_repeats):
            permuted[col] = rng.permutation(original)
            drops.append(mean_absolute_error(y_test, model.predict(permuted)) - base)
        permuted[col] = original
        scores[col] = float(np.mean(drops))

    s = pd.Series(scores).reindex(columns).fillna(0.0)
    s = s.clip(lower=0)  # قيمة سالبة تعني "لا أثر"، فتُثبّت عند الصفر
    total = s.sum()
    return (s / total if total > 0 else s).sort_values(ascending=False)


def train(X: pd.DataFrame, y: pd.Series, cfg: dict) -> TrainResult:
    mcfg = cfg["model"]
    X_tr, X_te, y_tr, y_te = time_split(X, y, mcfg["test_fraction"])

    n_splits = min(mcfg["n_splits"], max(2, len(X_tr) // 50))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    evaluations, fitted = [], {}
    for name, model in build_candidates(cfg).items():
        # تحقق زمني متدرج على جزء التدريب وحده
        cv_scores = []
        for tr_idx, va_idx in splitter.split(X_tr):
            model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            cv_scores.append(mean_absolute_error(
                y_tr.iloc[va_idx], model.predict(X_tr.iloc[va_idx])))

        model.fit(X_tr, y_tr)
        fitted[name] = model
        evaluations.append(
            _metrics(name, y_te, model.predict(X_te), cv_mae=float(np.mean(cv_scores)))
        )

    scored = [e for e in evaluations if e.name != "baseline_mean"]
    best = min(scored, key=lambda e: e.mae)
    best_model = fitted[best.name]

    preds = pd.DataFrame(
        {"actual": y_te, "predicted": best_model.predict(X_te)}, index=y_te.index
    )
    return TrainResult(
        best_name=best.name,
        best_model=best_model,
        evaluations=evaluations,
        importances=extract_importances(
            best_model, X_te, y_te, seed=mcfg["seed"],
            n_repeats=mcfg.get("importance_n_repeats", 5),
            max_features=mcfg.get("importance_max_features", 60),
        ),
        predictions=preds,
        drift=drift_report(y_tr, y_te),
        feature_names=list(X.columns),
    )


def classify_alarm(values: pd.Series, cfg: dict) -> pd.Series:
    """يحوّل قيمة الحديد المتوقعة إلى مستوى إنذار تشغيلي."""
    warn = cfg["alarm"]["iron_warning_ppm"]
    crit = cfg["alarm"]["iron_critical_ppm"]
    return pd.cut(
        values,
        bins=[-np.inf, warn, crit, np.inf],
        labels=["طبيعي", "تنبيه", "حرج"],
    )
