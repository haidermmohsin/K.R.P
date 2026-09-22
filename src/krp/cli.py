"""واجهة سطر الأوامر لخط تحليل وحدة CDU/VDU.

    python -m krp.cli simulate --days 180 --out data/raw/unit01_sim.csv
    python -m krp.cli analyze  --input data/raw/unit01_sim.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, load_config
from .features import build_feature_frame, corrosion_indicators
from .cleaning import clean
from .ingest import load_unit_data
from .model import classify_alarm, train
from . import report
from .simulate import simulate_unit, to_dcs_export


def _p(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def cmd_simulate(args) -> None:
    cfg = load_config(args.config)
    df = simulate_unit(n_days=args.days, freq=cfg["data"]["resample_freq"], seed=args.seed)
    export = to_dcs_export(df, cfg["_tag_map"])
    out = _p(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out, index=False)
    print(f"✓ بيانات اصطناعية: {len(export):,} صف × {len(export.columns)} وسم → {out}")


def cmd_analyze(args) -> None:
    cfg = load_config(args.config)
    outdir = _p(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("① قراءة البيانات ومواءمة معدلات أخذ العينات…")
    raw = load_unit_data(_p(args.input), cfg)
    print(f"   {len(raw):,} فترة زمنية × {len(raw.columns)} متغير"
          f"  ({raw.index.min()} → {raw.index.max()})")

    print("② التنظيف…")
    df, creport = clean(raw, cfg)
    for k, v in creport.items():
        print(f"   {k:32s} {v:,}")
    if df.empty:
        raise SystemExit("لا توجد بيانات بعد التنظيف — راجع إعدادات steady_state.")

    print("③ بناء الخصائص…")
    X, y = build_feature_frame(df, cfg)
    print(f"   {X.shape[0]:,} عيّنة × {X.shape[1]:,} خاصية"
          f"  | الهدف: {y.name}")

    print("④ التدريب (تقسيم زمني)…")
    result = train(X, y, cfg)
    board = result.leaderboard()
    print(board.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"   ★ الأفضل: {result.best_name}")

    d = result.drift
    print(f"   انجراف الهدف: مدى التدريب [{d['train_range'][0]:.2f}, {d['train_range'][1]:.2f}]"
          f" | مدى الاختبار [{d['test_range'][0]:.2f}, {d['test_range'][1]:.2f}]")
    print(f"   نسبة الاختبار خارج مدى التدريب: {d['test_fraction_outside_train_range']:.1%}")
    if d["severe"]:
        print("   ⚠ انجراف شديد: نماذج الأشجار لا تستقرئ خارج مدى التدريب —"
              " أعد التدريب دورياً أو اعتمد نموذجاً خطياً.")

    print("⑤ الرسوم والتقارير…")
    enriched = pd.concat([df, corrosion_indicators(df, cfg)], axis=1, sort=False)
    report.plot_prediction_timeline(result.predictions, cfg, outdir / "prediction_timeline.png")
    report.plot_feature_importance(result.importances, outdir / "feature_importance.png")
    report.plot_dewpoint_risk(enriched, cfg, outdir / "dewpoint_risk.png")

    board.to_csv(outdir / "model_leaderboard.csv", index=False)
    result.importances.head(40).to_csv(outdir / "feature_importance.csv", header=["importance"])

    preds = result.predictions.copy()
    preds["alarm"] = classify_alarm(preds["predicted"], cfg)
    preds.to_csv(outdir / "predictions.csv")

    summary = {
        "rows_analyzed": int(len(df)),
        "n_features": int(X.shape[1]),
        "horizon_h": cfg["model"]["horizon_h"],
        "best_model": result.best_name,
        "metrics": board.iloc[0].to_dict(),
        "drift": result.drift,
        "cleaning": creport,
        "top_features": result.importances.head(10).round(4).to_dict(),
        "alarm_distribution": preds["alarm"].value_counts().to_dict(),
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"✓ المخرجات في {outdir}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="krp", description="تحليل ظروف وحدة CDU/VDU")
    ap.add_argument("--config", default="configs/config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="توليد بيانات اصطناعية بشكل تصدير DCS")
    s.add_argument("--days", type=int, default=180)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="data/raw/unit01_sim.csv")
    s.set_defaults(func=cmd_simulate)

    a = sub.add_parser("analyze", help="تشغيل خط التحليل الكامل")
    a.add_argument("--input", required=True)
    a.add_argument("--outdir", default="outputs")
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
