"""تحميل ملفات الإعداد وخريطة وسوم الـ DCS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class TagSpec:
    """مواصفة وسم واحد: اسمه في الـ DCS، اسمه الموحّد، وحدوده الفيزيائية."""

    canonical: str
    dcs_tag: str
    unit: str
    limits: tuple[float, float]
    source: str
    desc: str = ""

    @property
    def is_lab(self) -> bool:
        return self.source == "lab"


@dataclass(frozen=True)
class TagMap:
    timestamp_column: str
    specs: tuple[TagSpec, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "TagMap":
        raw = load_yaml(path)
        specs = tuple(
            TagSpec(
                canonical=t["canonical"],
                dcs_tag=t["dcs_tag"],
                unit=t.get("unit", ""),
                limits=(float(t["limits"][0]), float(t["limits"][1])),
                source=t.get("source", "dcs"),
                desc=t.get("desc", ""),
            )
            for t in raw["tags"]
        )
        return cls(timestamp_column=raw.get("timestamp_column", "timestamp"), specs=specs)

    def by_canonical(self) -> dict[str, TagSpec]:
        return {s.canonical: s for s in self.specs}

    def rename_map(self) -> dict[str, str]:
        """خريطة {وسم DCS -> الاسم الموحّد} لإعادة تسمية أعمدة الملف الخام."""
        return {s.dcs_tag: s.canonical for s in self.specs}

    @property
    def lab_tags(self) -> list[str]:
        return [s.canonical for s in self.specs if s.is_lab]

    @property
    def dcs_tags(self) -> list[str]:
        return [s.canonical for s in self.specs if not s.is_lab]


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    cfg = load_yaml(path)
    cfg["_tag_map"] = TagMap.from_file(cfg["data"]["tags_file"])
    return cfg
