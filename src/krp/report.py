"""رسوم تشخيصية لنتائج النموذج.

لوحة الألوان مأخوذة من نظام ألوان مُتحقَّق منه لعمى الألوان في الوضعين
الفاتح والداكن، وكل رسم يحمل تسميات مباشرة فلا يعتمد على اللون وحده.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# لوحات مُتحقَّق منها (ΔE للرؤية الطبيعية ولعمى الألوان فوق الحدود المطلوبة)
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
        "seq": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
        "seq": "#3987e5",
    },
}
# ألوان الحالة ثابتة في الوضعين، وتُرافق دائماً بنص يوضح المعنى
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}


def _style(ax, theme: dict, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """شبكة وحدود خافتة — المحاور لا تنافس البيانات على الانتباه."""
    ax.set_facecolor(theme["surface"])
    ax.set_title(title, color=theme["ink"], fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=theme["ink2"], fontsize=9)
    ax.set_ylabel(ylabel, color=theme["ink2"], fontsize=9)
    ax.tick_params(colors=theme["muted"], labelsize=8)
    ax.grid(True, color=theme["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(0.8)


def plot_prediction_timeline(preds: pd.DataFrame, cfg: dict, out: Path,
                             theme_name: str = "light") -> Path:
    """القيمة الفعلية مقابل المتوقعة على فترة الاختبار، مع حدّي الإنذار."""
    th = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=th["surface"])

    ax.plot(preds.index, preds["actual"], color=th["s1"], linewidth=2, label="الفعلي")
    ax.plot(preds.index, preds["predicted"], color=th["s2"], linewidth=2,
            linestyle="--", label="المتوقّع")

    warn, crit = cfg["alarm"]["iron_warning_ppm"], cfg["alarm"]["iron_critical_ppm"]
    for level, color, text in ((warn, STATUS["warning"], "حد التنبيه"),
                               (crit, STATUS["critical"], "الحد الحرج")):
        ax.axhline(level, color=color, linewidth=1.4, linestyle=":")
        ax.text(preds.index[0], level, f"  {text}", color=color,
                fontsize=8, va="bottom")

    _style(ax, th, "الحديد في ماء القمة — الفعلي مقابل المتوقّع", ylabel="Fe (ppm)")
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9, ncols=2)
    for t in leg.get_texts():
        t.set_color(th["ink2"])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=th["surface"])
    plt.close(fig)
    return out


def plot_feature_importance(importances: pd.Series, out: Path, top_n: int = 15,
                            theme_name: str = "light") -> Path:
    """أهم الخصائص — كل شريط يحمل قيمته مباشرة، بلا وسيلة إيضاح."""
    th = THEMES[theme_name]
    top = importances.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(top) + 1.6), facecolor=th["surface"])
    ax.barh(range(len(top)), top.to_numpy(), color=th["seq"], height=0.62)
    ax.set_yticks(range(len(top)), [t[:46] for t in top.index], fontsize=8)

    span = top.max() if len(top) else 1
    for i, v in enumerate(top.to_numpy()):
        ax.text(v + span * 0.015, i, f"{v:.3f}", va="center",
                fontsize=8, color=th["ink2"])
    ax.set_xlim(0, span * 1.18)

    _style(ax, th, "الخصائص الأكثر تأثيراً في التنبؤ", xlabel="الأهمية النسبية")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=th["surface"])
    plt.close(fig)
    return out


def plot_risk_scatter(df: pd.DataFrame, cfg: dict, out: Path, x_col: str,
                      title: str, xlabel: str, threshold: float | None = None,
                      threshold_label: str = "حد الأمان",
                      theme_name: str = "light") -> Path | None:
    """الحديد مقابل متغيّر تشغيلي، مصنّفاً بمستوى الإنذار.

    كل نقطة فترة تشغيل. اللون والشكل والنص تحمل مستوى الخطر معاً، فلا
    يعتمد التمييز على اللون وحده. يعيد None إذا لم تتوفر الأعمدة.
    """
    th = THEMES[theme_name]
    target = cfg["model"]["target"]
    if x_col not in df.columns or target not in df.columns:
        return None
    sub = df[[x_col, target]].dropna()
    if sub.empty:
        return None

    warn, crit = cfg["alarm"]["iron_warning_ppm"], cfg["alarm"]["iron_critical_ppm"]
    bands = [
        (sub[target] <= warn, STATUS["good"], "طبيعي", "o"),
        ((sub[target] > warn) & (sub[target] <= crit), STATUS["warning"], "تنبيه", "s"),
        (sub[target] > crit, STATUS["critical"], "حرج", "^"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=th["surface"])
    for mask, color, label, marker in bands:
        if mask.any():
            ax.scatter(sub.loc[mask, x_col], sub.loc[mask, target],
                       s=26, c=color, marker=marker, label=label,
                       edgecolors=th["surface"], linewidths=0.6, alpha=0.85)

    if threshold is not None:
        ax.axvline(threshold, color=th["ink2"], linewidth=1.4, linestyle="--")
        ax.text(threshold, ax.get_ylim()[1], f" {threshold_label} ",
                color=th["ink2"], fontsize=8, va="top")

    _style(ax, th, title, xlabel=xlabel, ylabel="Fe (ppm)")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(th["ink2"])
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=th["surface"])
    plt.close(fig)
    return out


def plot_hot_drum_risk(df: pd.DataFrame, cfg: dict, out: Path,
                       theme_name: str = "light") -> Path | None:
    """الحديد مقابل هامش الحوض الساخن 01-V-003.

    في وحدة ذات حوض ساخن يبقى هذا الهامش واسعاً في التشغيل الطبيعي، فليس
    هو القائد الروتيني للتآكل. قيمة الرسم أنه يكشف الفترات القليلة التي
    يضيق فيها — وهي الأخطر، إذ يعني ضيقه تكثّفاً حمضياً داخل الحوض نفسه.
    """
    return plot_risk_scatter(
        df, cfg, out, x_col="hot_drum_margin_c",
        title="الحديد مقابل هامش الحوض الساخن 01-V-003",
        xlabel="هامش نقطة الندى (°م)",
        threshold=cfg["features"]["dewpoint_margin_alarm"],
        theme_name=theme_name,
    )


def plot_chloride_risk(df: pd.DataFrame, cfg: dict, out: Path,
                       theme_name: str = "light") -> Path | None:
    """الحديد مقابل الكلورايد في الماء الحامض — القائد الروتيني للتآكل.

    هذا هو الرسم الذي يُعرض على مهندس التشغيل: الكلورايد متغيّر يملك
    السيطرة عليه عبر أداء الديسولتر وماء الغسيل، والعلاقة مباشرة.
    """
    return plot_risk_scatter(
        df, cfg, out, x_col="cdu_sour_water_chloride",
        title="الحديد مقابل الكلورايد في الماء الحامض",
        xlabel="الكلورايد (ppm)",
        theme_name=theme_name,
    )
