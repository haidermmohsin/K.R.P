"""مولّد بيانات اصطناعية واقعية لوحدة CDU/VDU.

الغرض منه تشغيل خط التحليل والتحقق منه قبل وصول بيانات الـ DCS الحقيقية.
ليس بديلاً عن البيانات الحقيقية — لكنه يحاكي الآليات الصحيحة:
الملح المتبقي يولّد كلورايد، والكلورايد مع هامش ندى منخفض وحموضة متدنية
يرفع الحديد، والمحايد يخفّضه، مع تأخّر زمني وضوضاء قياس وأعطال حسّاسات.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TagMap


def _walk(rng: np.random.Generator, n: int, start: float, vol: float,
          lo: float, hi: float) -> np.ndarray:
    """مسير عشوائي مرتد بين حدين — يحاكي انجراف متغير تشغيلي."""
    x = np.empty(n)
    x[0] = start
    for i in range(1, n):
        pull = 0.01 * (start - x[i - 1])          # ارتداد نحو نقطة الضبط
        x[i] = x[i - 1] + pull + rng.normal(0, vol)
    return np.clip(x, lo, hi)


def simulate_unit(n_days: int = 180, freq: str = "1h", seed: int = 42,
                  start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=int(n_days * 24 * (pd.Timedelta(hours=1) / pd.Timedelta(freq))),
                        freq=freq)
    n = len(idx)
    hours = np.arange(n) * (pd.Timedelta(freq) / pd.Timedelta(hours=1))

    df = pd.DataFrame(index=idx)
    df.index.name = "timestamp"

    # ---------- التغذية وحملة الوحدة ----------
    feed = _walk(rng, n, 880, 3.0, 500, 980)
    # توقفان مخططان لإنتاج فترات غير مستقرة يجب أن يستبعدها التنظيف
    for t0 in (int(n * 0.35), int(n * 0.72)):
        feed[t0:t0 + 30] *= np.linspace(1.0, 0.05, 30)
        feed[t0 + 30:t0 + 60] *= np.linspace(0.05, 1.0, 30)
    df["crude_feed_rate"] = feed

    # ---------- الخام: تغيّر خلطة كل ~3 أسابيع ----------
    blend = np.repeat(rng.uniform(0, 1, n // 500 + 1), 500)[:n]
    df["crude_api"] = 28 + 6 * blend + rng.normal(0, 0.3, n)
    df["crude_sulfur"] = 2.6 - 1.4 * blend + rng.normal(0, 0.05, n)
    df["crude_tan"] = 0.3 + 1.6 * blend + rng.normal(0, 0.05, n)

    # ---------- الديسولتر ----------
    df["desalter_temp"] = _walk(rng, n, 130, 0.4, 100, 150)
    df["desalter_voltage"] = _walk(rng, n, 26, 0.3, 12, 32)
    df["wash_water_rate"] = 0.055 * feed + rng.normal(0, 1.5, n)
    df["desalter_dp"] = _walk(rng, n, 0.8, 0.02, 0.2, 2.2)
    df["caustic_rate"] = _walk(rng, n, 8, 0.15, 0, 20)

    # كفاءة إزالة الملح: ترتفع بماء الغسيل والجهد والحرارة، وتنهار مع تراكم الترسبات.
    # التدهور سنّي (sawtooth): يتراكم ثم يعود إلى الصفر عند كل تنظيف دوري —
    # وهذا هو السلوك الحقيقي، لا انحداراً أحادي الاتجاه. الفرق جوهري للنمذجة:
    # الانجراف أحادي الاتجاه يجعل فترة الاختبار خارج مدى التدريب،
    # ونماذج الأشجار لا تستقرئ خارج ذلك المدى أبداً.
    cleaning_interval_h = 45 * 24
    degradation = (hours % cleaning_interval_h) / cleaning_interval_h * 0.12
    eff = (0.985
           + 0.10 * (df["wash_water_rate"] / feed - 0.055)
           + 0.002 * (df["desalter_voltage"] - 26)
           + 0.0008 * (df["desalter_temp"] - 130)
           - degradation)
    eff = np.clip(eff, 0.80, 0.995)
    salt_in = 18 + 14 * blend                      # ملح الخام الوارد
    df["salt_in_crude_ptb"] = np.clip(salt_in * (1 - eff) * 10 + rng.normal(0, 0.4, n), 0.2, 40)
    df["bsw_percent"] = np.clip(0.25 + 0.5 * (1 - eff) * 10 + rng.normal(0, 0.05, n), 0, 3)

    # ---------- الفرن والبرج الجوي ----------
    df["furnace_coil_outlet_temp"] = _walk(rng, n, 355, 0.5, 320, 380)
    df["furnace_o2"] = _walk(rng, n, 3.0, 0.08, 1.0, 8.0)
    df["atm_flash_zone_temp"] = df["furnace_coil_outlet_temp"] - rng.normal(4, 1, n)
    df["stripping_steam_rate"] = _walk(rng, n, 3200, 25, 1500, 6000)

    # ---------- قمة البرج ----------
    df["ovhd_pressure"] = _walk(rng, n, 1.10, 0.006, 0.7, 1.8)
    ovhd_t = _walk(rng, n, 126, 0.5, 100, 150)

    # حملات خفض حرارة القمة لزيادة استخلاص النافثا (توفير طاقة).
    # تضيّق هامش نقطة الندى وتقترب به من حدّ الخطر — وهي الفترات التي
    # يُختبر فيها النموذج فعلياً، إذ بدونها تبقى آلية التكثّف الحمضي خاملة.
    for frac in (0.18, 0.47, 0.83):
        t0 = int(n * frac)
        span = min(int(9 * 24 * (pd.Timedelta(hours=1) / pd.Timedelta(freq))), n - t0)
        if span <= 0:
            continue
        ramp = np.concatenate([
            np.linspace(0, 1, span // 4),
            np.ones(span - 2 * (span // 4)),
            np.linspace(1, 0, span // 4),
        ])[:span]
        ovhd_t[t0:t0 + span] -= 18.0 * ramp
    df["ovhd_temp"] = np.clip(ovhd_t, 95, 150)
    df["ovhd_vapor_rate"] = 42 * feed + rng.normal(0, 600, n)
    df["reflux_rate"] = _walk(rng, n, 210, 2.0, 80, 350)
    df["condenser_outlet_temp"] = _walk(rng, n, 48, 0.4, 30, 80)

    # ---------- الحقن الكيميائي ----------
    df["neutralizer_rate"] = _walk(rng, n, 22, 0.25, 4, 50)
    df["filmer_rate"] = _walk(rng, n, 9, 0.12, 1, 20)

    # ---------- كيمياء ماء القمة ----------
    # الكلورايد الناتج من التحلل المائي للأملاح المتبقية
    chloride = np.clip(6 + 3.2 * df["salt_in_crude_ptb"] + rng.normal(0, 3, n), 1, 400)
    df["ovhd_water_chloride"] = chloride
    df["ovhd_water_h2s"] = np.clip(18 * df["crude_sulfur"] + rng.normal(0, 4, n), 1, 400)

    # الحموضة: يرفعها المحايد ويخفضها الكلورايد
    ph = 6.0 + 0.055 * (df["neutralizer_rate"] - 22) - 0.010 * (chloride - 60)
    df["ovhd_water_ph"] = np.clip(ph + rng.normal(0, 0.12, n), 3.2, 9.0)

    # ---------- VDU ----------
    df["vac_heater_cot"] = _walk(rng, n, 390, 0.5, 350, 410)
    df["vac_transfer_line_temp"] = df["vac_heater_cot"] - rng.normal(3, 0.8, n)
    df["vac_tower_top_pressure"] = _walk(rng, n, 25, 0.4, 8, 90)
    df["vac_bottom_temp"] = _walk(rng, n, 360, 0.6, 320, 395)

    # ---------- الحديد: الآلية الحقيقية ----------
    from .features import water_dewpoint  # استيراد متأخر لتفادي الدوران

    dew = water_dewpoint(df).to_numpy()
    margin = df["ovhd_temp"].to_numpy() - dew            # هامش نقطة الندى
    acid_drive = np.clip(20.0 - margin, 0, None) / 20.0  # يشتد كلما ضاق الهامش
    ph_drive = np.clip(6.0 - df["ovhd_water_ph"].to_numpy(), 0, None)
    cl_drive = chloride.to_numpy() / 100.0
    film_protect = np.clip(df["filmer_rate"].to_numpy() / 12.0, 0.2, 1.6)

    inst = (0.45
            + 2.6 * acid_drive * cl_drive
            + 1.5 * ph_drive
            + 0.35 * cl_drive) / film_protect

    # التآكل تراكمي وبطيء الاستجابة: مرشّح أسّي بثابت زمني ~10 ساعات
    iron = np.empty(n)
    iron[0] = inst[0]
    alpha = 0.10
    for i in range(1, n):
        iron[i] = (1 - alpha) * iron[i - 1] + alpha * inst[i]
    iron = np.clip(iron + rng.normal(0, 0.18, n), 0.05, 30)
    df["ovhd_water_iron"] = iron

    # ---------- واقعية القياس ----------
    # نتائج المختبر مرة كل 24 ساعة فقط
    lab_cols = ["salt_in_crude_ptb", "bsw_percent", "crude_api", "crude_sulfur",
                "crude_tan", "ovhd_water_ph", "ovhd_water_chloride",
                "ovhd_water_h2s", "ovhd_water_iron"]
    step_h = int(pd.Timedelta(hours=24) / pd.Timedelta(freq))
    lab_mask = np.ones(n, dtype=bool)
    lab_mask[::step_h] = False
    df.loc[df.index[lab_mask], lab_cols] = np.nan

    # حسّاس مجمّد لمدة يومين
    f0 = int(n * 0.55)
    df.iloc[f0:f0 + 48, df.columns.get_loc("ovhd_temp")] = df["ovhd_temp"].iloc[f0]

    # قيم شاذة متفرقة
    for col in ("reflux_rate", "furnace_o2", "ovhd_pressure"):
        spikes = rng.choice(n, size=max(3, n // 2000), replace=False)
        df.iloc[spikes, df.columns.get_loc(col)] *= 12

    return df


def to_dcs_export(df: pd.DataFrame, tag_map: TagMap) -> pd.DataFrame:
    """يحوّل الأسماء الموحّدة إلى وسوم DCS — لمحاكاة شكل ملف التصدير الحقيقي."""
    inverse = {s.canonical: s.dcs_tag for s in tag_map.specs}
    out = df.rename(columns=inverse)
    out.index.name = tag_map.timestamp_column
    return out.reset_index()
