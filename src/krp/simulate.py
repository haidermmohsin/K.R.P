"""مولّد بيانات اصطناعية لوحدة 01 CDU/VDU — مصفى كربلاء.

يحاكي تصميم الوحدة كما في IONE-AA01-PFD-0010-001: إزالة ملح على مرحلتين،
نظام قمة بحوضين (ساخن 01-V-003 ثم بارد 01-V-005)، وثلاث نقاط حقن كيميائي.
الغرض التحقق من الخط البرمجي قبل وصول بيانات الـ DCS — لا تمثيل الوحدة.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TagMap


def _walk(rng: np.random.Generator, n: int, start: float, vol: float,
          lo: float, hi: float) -> np.ndarray:
    """مسير عشوائي مرتد حول نقطة ضبط — يحاكي انجراف متغير تشغيلي."""
    x = np.empty(n)
    x[0] = start
    for i in range(1, n):
        x[i] = x[i - 1] + 0.01 * (start - x[i - 1]) + rng.normal(0, vol)
    return np.clip(x, lo, hi)


def _campaigns(n: int, fracs: tuple[float, ...], span: int, depth: float) -> np.ndarray:
    """ملف حملات تشغيلية: صعود ثم ثبات ثم هبوط، بعمق `depth`."""
    out = np.zeros(n)
    for frac in fracs:
        t0 = int(n * frac)
        width = min(span, n - t0)
        if width <= 4:
            continue
        q = width // 4
        ramp = np.concatenate([np.linspace(0, 1, q), np.ones(width - 2 * q),
                               np.linspace(1, 0, q)])[:width]
        out[t0:t0 + width] += depth * ramp
    return out


def simulate_unit(n_days: int = 180, freq: str = "1h", seed: int = 42,
                  start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps_per_h = pd.Timedelta(hours=1) / pd.Timedelta(freq)
    n = int(n_days * 24 * steps_per_h)
    idx = pd.date_range(start, periods=n, freq=freq)
    hours = np.arange(n) / steps_per_h

    df = pd.DataFrame(index=idx)
    df.index.name = "timestamp"

    # ---------------- التغذية ----------------
    feed = _walk(rng, n, 880, 3.0, 500, 980)
    for t0 in (int(n * 0.35), int(n * 0.72)):          # توقفان مخططان
        feed[t0:t0 + 30] *= np.linspace(1.0, 0.05, 30)
        feed[t0 + 30:t0 + 60] *= np.linspace(0.05, 1.0, 30)
    df["crude_feed_rate"] = feed

    # ---------------- خلطة الخام ----------------
    blend = np.repeat(rng.uniform(0, 1, n // 500 + 1), 500)[:n]
    df["crude_api"] = 28 + 6 * blend + rng.normal(0, 0.3, n)
    df["crude_sulfur"] = 2.6 - 1.4 * blend + rng.normal(0, 0.05, n)
    df["crude_tan"] = 0.3 + 1.6 * blend + rng.normal(0, 0.05, n)

    # ------- إزالة الملح على مرحلتين (01-V-001A ثم 01-V-001B) -------
    # الترسّب يتراكم ثم يعود إلى الصفر عند كل تنظيف دوري. التدهور سنّي لا
    # أحادي الاتجاه، وهو الفرق الذي يُبقي فترة الاختبار داخل مدى التدريب.
    clean_h = 45 * 24
    fouling = (hours % clean_h) / clean_h
    df["wash_water_rate"] = 0.040 * feed + rng.normal(0, 1.2, n)
    df["wash_water_recycle_rate"] = 0.015 * feed + rng.normal(0, 0.8, n)
    df["caustic_rate"] = _walk(rng, n, 8, 0.15, 0, 20)

    # ملح الخام الوارد بوحدة PTB (رطل لكل ألف برميل). خام البصرة ثقيل
    # الملوحة، والمدى 60–140 PTB شائع قبل المعالجة.
    salt_in = 60.0 + 80.0 * blend
    remaining = salt_in.copy()
    for stage, (v0, t0_, demul0) in enumerate(((26.0, 130.0, 6.0), (24.0, 136.0, 5.0)), 1):
        volt = _walk(rng, n, v0, 0.3, 12, 32)
        temp = _walk(rng, n, t0_, 0.4, 100, 150)
        demul = _walk(rng, n, demul0, 0.08, 0.5, 20)
        df[f"desalter{stage}_voltage"] = volt
        df[f"desalter{stage}_temp"] = temp
        df[f"desalter{stage}_dp"] = _walk(rng, n, 0.8, 0.02, 0.2, 2.2) + 0.6 * fouling
        df[f"demulsifier{stage}_rate"] = demul

        wash_ratio = (df["wash_water_rate"] + df["wash_water_recycle_rate"]) / feed
        # كفاءة المرحلة الواحدة 80–96%؛ المرحلتان معاً تبلغان ~99% عند
        # النظافة التامة فتُنزلان 100 PTB إلى نحو 0.2، وتهبطان إلى ~96%
        # عند أقصى ترسّب فيقفز المتبقي إلى عدة وحدات — وهذا هو المدى
        # الذي تظهر فيه مشكلة الحديد فعلاً.
        eff = (0.940
               + 0.60 * (wash_ratio - 0.055)
               + 0.0020 * (volt - v0)
               + 0.0008 * (temp - t0_)
               + 0.0030 * (demul - demul0)
               - 0.140 * fouling)
        remaining = remaining * (1 - np.clip(eff, 0.80, 0.96))

    df["salt_in_crude_ptb"] = np.clip(remaining + rng.normal(0, 0.12, n), 0.05, 45)
    df["bsw_percent"] = np.clip(0.2 + 0.9 * fouling + rng.normal(0, 0.05, n), 0, 3)

    # ---------------- الفرن والبرج الجوي ----------------
    df["furnace_coil_outlet_temp"] = _walk(rng, n, 355, 0.5, 320, 380)
    df["furnace_o2"] = _walk(rng, n, 3.0, 0.08, 1.0, 8.0)
    df["atm_flash_zone_temp"] = df["furnace_coil_outlet_temp"] - rng.normal(4, 1, n)
    df["stripping_steam_rate"] = _walk(rng, n, 3200, 25, 1500, 6000)
    df["atm_reflux_rate"] = _walk(rng, n, 210, 2.0, 80, 350)
    df["atm_ovhd_vapor_rate"] = 42 * feed + rng.normal(0, 600, n)

    # ---------------- قمة البرج ونظام الحوضين ----------------
    df["atm_ovhd_pressure"] = _walk(rng, n, 1.45, 0.006, 0.9, 2.2)
    df["atm_ovhd_temp"] = _walk(rng, n, 172, 0.5, 140, 195)
    df["hot_drum_pressure"] = df["atm_ovhd_pressure"] - _walk(rng, n, 0.25, 0.004, 0.05, 0.6)

    # حملات خفض حرارة الحوض الساخن لزيادة استرجاع الحرارة في 01-E-003.
    # تضيّق هامش نقطة الندى وتقترب به من حدّ الخطر — وهي الفترات التي
    # تُفعَّل فيها آلية التكثّف الحمضي أصلاً.
    hot_drum = _walk(rng, n, 138, 0.4, 110, 165)
    hot_drum -= _campaigns(n, (0.18, 0.47, 0.83), int(9 * 24 * steps_per_h), 20.0)
    df["hot_drum_temp"] = np.clip(hot_drum, 105, 170)

    df["ovhd_air_cooler_outlet_temp"] = _walk(rng, n, 62, 0.4, 35, 110)
    df["accumulator_pressure"] = df["hot_drum_pressure"] - 0.5
    df["accumulator_temp"] = _walk(rng, n, 38, 0.3, 28, 70)

    # ---------------- الحقن الكيميائي: ثلاث نقاط ----------------
    df["corrosion_inhibitor_rate"] = _walk(rng, n, 14, 0.18, 2, 35)
    df["neutralizer_tower_rate"] = _walk(rng, n, 12, 0.2, 2, 40)
    df["neutralizer_drum_rate"] = _walk(rng, n, 10, 0.2, 2, 40)
    df["filmer_rate"] = _walk(rng, n, 9, 0.12, 1, 20)

    # ---------------- كيمياء الماء الحامض (01-V-005) ----------------
    # تحوّل الأملاح المتبقية إلى HCl ثم إلى كلورايد في الماء المتكثّف.
    # القاعدة التقريبية: كل PTB يعطي نحو 20 ppm كلورايد في ماء القمة.
    chloride = np.clip(6 + 20.0 * df["salt_in_crude_ptb"] + rng.normal(0, 3, n), 1, 400)
    df["cdu_sour_water_chloride"] = chloride
    df["cdu_sour_water_h2s"] = np.clip(18 * df["crude_sulfur"] + rng.normal(0, 4, n), 1, 400)

    neutralizer = df["neutralizer_tower_rate"] + df["neutralizer_drum_rate"]
    ph = 6.2 + 0.050 * (neutralizer - 22) - 0.014 * (chloride - 60)
    df["cdu_sour_water_ph"] = np.clip(ph + rng.normal(0, 0.12, n), 3.2, 9.0)

    # ---------------- الـ VDU ----------------
    df["vac_heater_cot"] = _walk(rng, n, 407, 0.5, 360, 425)
    df["velocity_steam_rate"] = _walk(rng, n, 2400, 20, 800, 5000)
    df["vac_transfer_line_temp"] = df["vac_heater_cot"] - rng.normal(3, 0.8, n)
    df["vac_flash_zone_temp"] = df["vac_transfer_line_temp"] - rng.normal(10, 1.5, n)
    df["vac_tower_top_pressure"] = _walk(rng, n, 25, 0.4, 8, 90)
    df["vac_residue_temp"] = _walk(rng, n, 200, 0.5, 160, 380)
    df["vac_corrosion_inhibitor_rate"] = _walk(rng, n, 10, 0.15, 1, 30)
    df["vac_neutralizer_rate"] = _walk(rng, n, 8, 0.15, 1, 30)
    df["vdu_sour_water_ph"] = np.clip(_walk(rng, n, 6.4, 0.03, 4, 9), 4, 9)

    # ---------------- الحديد: الآلية المرتبطة بالحوض الساخن ----------------
    from .features import dewpoint_at  # استيراد متأخر لتفادي الدوران

    # في وحدة ذات حوض ساخن، التكثّف المائي الروتيني لا يقع في 01-V-003 بل
    # في 01-EA-004/01-E-020 حيث يُبرَّد البخار إلى 38°م — أي دون نقطة الندى
    # بكثير. فالقائد الروتيني للتآكل هو حموضة أول متكثّف: الكلورايد مقابل
    # جرعة المحايد. أما هامش الحوض الساخن فيبقى واسعاً (نحو 40°م) في
    # التشغيل الطبيعي، ولا يصير قائداً إلا حين ينهار في اضطراب تشغيلي —
    # وعندها يتكثّف الحمض داخل الحوض نفسه فيقفز الحديد قفزة حادة.
    dew = dewpoint_at(df, "hot_drum_pressure").to_numpy()
    margin = df["hot_drum_temp"].to_numpy() - dew
    acid_drive = np.clip(25.0 - margin, 0, None) / 25.0
    ph_drive = np.clip(6.0 - df["cdu_sour_water_ph"].to_numpy(), 0, None)
    cl_drive = chloride.to_numpy() / 100.0
    protect = np.clip(
        0.6 + 0.4 * (df["filmer_rate"].to_numpy() / 12.0)
        + 0.25 * (df["corrosion_inhibitor_rate"].to_numpy() / 14.0),
        0.5, 1.8,
    )
    inst = (0.60
            + 1.80 * cl_drive                 # القائد الروتيني
            + 2.00 * ph_drive                 # الحموضة الفعلية للمتكثّف
            + 4.00 * acid_drive * cl_drive    # تصعيد حاد عند انهيار الهامش
            ) / protect

    # التآكل تراكمي بطيء الاستجابة: مرشّح أسّي بثابت زمني نحو عشر ساعات
    iron = np.empty(n)
    iron[0] = inst[0]
    for i in range(1, n):
        iron[i] = 0.90 * iron[i - 1] + 0.10 * inst[i]
    df["cdu_sour_water_iron"] = np.clip(iron + rng.normal(0, 0.18, n), 0.05, 30)

    # حديد الـ VDU يقوده التآكل الكبريتي وحمضية الأحماض النفثينية
    nac = df["crude_tan"].to_numpy() * np.clip(
        df["vac_transfer_line_temp"].to_numpy() - 220.0, 0, None) / 100.0
    df["vdu_sour_water_iron"] = np.clip(
        0.3 + 0.35 * nac + 0.12 * df["crude_sulfur"].to_numpy() + rng.normal(0, 0.1, n),
        0.02, 30)

    # ---------------- واقعية القياس ----------------
    lab_cols = ["salt_in_crude_ptb", "bsw_percent", "crude_api", "crude_sulfur",
                "crude_tan", "cdu_sour_water_ph", "cdu_sour_water_chloride",
                "cdu_sour_water_h2s", "cdu_sour_water_iron",
                "vdu_sour_water_ph", "vdu_sour_water_iron"]
    step = max(1, int(24 * steps_per_h))                 # عيّنة مختبرية يومية
    mask = np.ones(n, dtype=bool)
    mask[::step] = False
    df.loc[df.index[mask], lab_cols] = np.nan

    # حسّاس مجمّد يومين على حرارة الحوض الساخن
    f0 = int(n * 0.55)
    df.iloc[f0:f0 + int(48 * steps_per_h), df.columns.get_loc("hot_drum_temp")] = \
        df["hot_drum_temp"].iloc[f0]

    # قيم شاذة متفرقة
    for col in ("atm_reflux_rate", "furnace_o2", "atm_ovhd_pressure"):
        spikes = rng.choice(n, size=max(3, n // 2000), replace=False)
        df.iloc[spikes, df.columns.get_loc(col)] *= 12

    return df


def to_dcs_export(df: pd.DataFrame, tag_map: TagMap) -> pd.DataFrame:
    """يحوّل الأسماء الموحّدة إلى وسوم DCS — لمحاكاة شكل ملف التصدير."""
    inverse = {s.canonical: s.dcs_tag for s in tag_map.specs}
    out = df.rename(columns=inverse)
    out.index.name = tag_map.timestamp_column
    return out.reset_index()
