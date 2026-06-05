#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"^\s*epoch\s*:\s*(.+?)\s*$", re.IGNORECASE)
METRIC_RE = re.compile(
    r"^\s*(?P<prefix>[^\/:]+)\s*\/\s*(?P<metric>[^\/:]+)\s*:\s*(?P<value>.+?)\s*$"
)


def safe_filename(name: str) -> str:
    # чтобы metric_name вроде "loss/cls" не ломал пути
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "metric"


def parse_epoch(raw: str) -> Optional[int]:
    raw = raw.strip()
    if raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_value(raw: str) -> Optional[float]:
    raw = raw.strip()
    # поддержка NaN/inf тоже ок, но такие точки лучше не рисовать
    try:
        v = float(raw)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def is_test_prefix(prefix: str) -> bool:
    return prefix.strip().lower().startswith("test")


def read_all_metrics(path: Path):
    """
    Возвращает структуру:
    metric_name -> {
        "series": { prefix -> list[(epoch, value)] },
        "tests":  { prefix -> list[value] }  # обычно 1 значение, но если много — усредним
    }
    """
    data: Dict[str, Dict[str, Dict[str, List]]] = {}

    current_epoch: Optional[int] = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue

        m_epoch = EPOCH_RE.match(line)
        if m_epoch:
            current_epoch = parse_epoch(m_epoch.group(1))
            continue

        m = METRIC_RE.match(line)
        if not m:
            # строка не похожа на метрику — пропускаем
            continue

        prefix = m.group("prefix").strip()
        metric = m.group("metric").strip()
        value = parse_value(m.group("value"))
        if value is None:
            continue

        bucket = data.setdefault(metric, {"series": {}, "tests": {}})

        if is_test_prefix(prefix):
            bucket["tests"].setdefault(prefix, []).append(value)
            continue

        # не тест: должен быть epoch!=None, иначе игнор
        if current_epoch is None:
            continue

        bucket["series"].setdefault(prefix, []).append((current_epoch, value))

    # сортировка точек по эпохе
    for metric, bucket in data.items():
        for prefix, pts in list(bucket["series"].items()):
            pts.sort(key=lambda x: x[0])
            # если вдруг дубли epoch — оставим как есть (matplotlib нормально рисует),
            # но можно усреднить при желании

    return data


def plot_metric(
    metric_name: str,
    series: Dict[str, List[Tuple[int, float]]],
    tests: Dict[str, List[float]],
    out_path: Path,
):
    plt.figure(figsize=(14, 7), dpi=160)

    ax = plt.gca()

    # Рисуем тренировочные/валидационные серии
    prefixes = sorted(series.keys())
    for prefix in prefixes:
        pts = series[prefix]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", markersize=3.0, linewidth=1.6, label=prefix)

    # Рисуем тестовые горизонтальные линии разными цветами
    test_prefixes = sorted(tests.keys())
    cmap = plt.get_cmap("tab20")  # много разных цветов

    for i, tprefix in enumerate(test_prefixes):
        vals = [v for v in tests[tprefix] if v is not None]
        if not vals:
            continue

        y = sum(vals) / len(vals)
        color = cmap(i % cmap.N)

        ax.axhline(
            y=y,
            color=color,
            linestyle="--",
            linewidth=2.4,
            alpha=0.95,
            label=f"TEST:{tprefix}",
            zorder=3,
        )

        # Если легенду не показываем — подпишем линию справа, чтобы не гадать
        # x=0.995 в координатах осей, y в координатах данных
        # (не искажает значения и видна даже при наложении линий)
        # legend_count посчитай ниже или просто используй условие len(prefixes)+len(test_prefixes) > 4
        if (len(series) + len(test_prefixes)) > 4:
            ax.text(
                0.995, y, f"TEST:{tprefix}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom",
                fontsize=9, color=color,
                alpha=0.95,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6),
                zorder=4,
            )
    # Легенда: показываем только если <= 4 разных prefix (включая test-серии тоже считаем)
    legend_count = len(prefixes) + len(tests)
    if 1 <= legend_count <= 4:
        plt.legend(loc="best", fontsize=10, frameon=True)

    # Нормальная обрезка краёв
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def find_all_metrics_files(root: Path) -> List[Path]:
    # “рядом с файлом которого лежат папки” — значит root = папка скрипта,
    # а all_metrics.txt лежат внутри подпапок. Рекурсивно это покрывает.
    return sorted(root.rglob("all_metrics.txt"))


def main():
    script_dir = Path(__file__).resolve().parent
    metrics_files = find_all_metrics_files(script_dir)

    if not metrics_files:
        print(f"[!] Не найдено ни одного all_metrics.txt в {script_dir}")
        return

    for mf in metrics_files:
        try:
            parsed = read_all_metrics(mf)
        except Exception as e:
            print(f"[!] Ошибка чтения/парсинга {mf}: {e}")
            continue

        if not parsed:
            print(f"[-] Метрик не найдено в {mf}")
            continue

        plots_dir = mf.parent / "plots"
        for metric_name, bucket in parsed.items():
            out_file = plots_dir / f"{safe_filename(metric_name)}.png"
            plot_metric(
                metric_name=metric_name,
                series=bucket["series"],
                tests=bucket["tests"],
                out_path=out_file,
            )

        print(f"[+] Готово: {mf.parent} -> {plots_dir} (метрик: {len(parsed)})")


if __name__ == "__main__":
    main()
