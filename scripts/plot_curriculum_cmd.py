"""Plot curriculum command ranges from a TensorBoard event file.

Purpose:
- Read Curriculum scalar tags from one tfevents file and visualize how command ranges
  change over training iterations.

Main contents:
- `load_scalar_series`: load one scalar tag as (steps, values).
- `align_series`: align min/max series on a shared step axis.
- `plot_curriculum_ranges`: draw 3 subplots with min/max boundary lines and range band.
- CLI entrypoint for event-file path, output directory, and optional step range.
- per-metric color themes for clearer visual separation across x/y/z command groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: tensorboard. "
        "Please install it in your environment before running this script."
    ) from exc


CURRICULUM_TAGS = {
    "cmd_lin_vel_x": (
        "Curriculum / cmd_lin_vel_x_min",
        "Curriculum / cmd_lin_vel_x_max",
    ),
    "cmd_lin_vel_y": (
        "Curriculum / cmd_lin_vel_y_min",
        "Curriculum / cmd_lin_vel_y_max",
    ),
    "cmd_ang_vel_z": (
        "Curriculum / cmd_ang_vel_z_min",
        "Curriculum / cmd_ang_vel_z_max",
    ),
}

METRIC_THEME_COLORS = {
    "cmd_lin_vel_x": {
        "theme": "Ocean Blue",
        "max": "#1D4ED8",
        "min": "#60A5FA",
        "band": "#93C5FD",
    },
    "cmd_lin_vel_y": {
        "theme": "Sunset Orange",
        "max": "#C2410C",
        "min": "#FB923C",
        "band": "#FDBA74",
    },
    "cmd_ang_vel_z": {
        "theme": "Forest Green",
        "max": "#166534",
        "min": "#4ADE80",
        "band": "#86EFAC",
    },
}


def load_scalar_series(event_accumulator: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one scalar tag and return (steps, values)."""
    events = event_accumulator.Scalars(tag)
    if not events:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    steps = np.array([item.step for item in events], dtype=np.int64)
    values = np.array([item.value for item in events], dtype=np.float64)
    return steps, values


def align_series(
    steps_a: np.ndarray,
    values_a: np.ndarray,
    steps_b: np.ndarray,
    values_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two series by step and return (steps, values_a, values_b)."""
    if steps_a.size == 0 and steps_b.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if np.array_equal(steps_a, steps_b):
        return steps_a, values_a, values_b

    merged_steps = np.union1d(steps_a, steps_b)
    map_a = dict(zip(steps_a.tolist(), values_a.tolist()))
    map_b = dict(zip(steps_b.tolist(), values_b.tolist()))
    aligned_a = np.array([map_a.get(int(step), np.nan) for step in merged_steps], dtype=np.float64)
    aligned_b = np.array([map_b.get(int(step), np.nan) for step in merged_steps], dtype=np.float64)
    return merged_steps, aligned_a, aligned_b


def plot_curriculum_ranges(
    event_file: Path,
    output_dir: Path,
    dpi: int = 200,
    step_min: int | None = None,
    step_max: int | None = None,
) -> Path:
    """Create and save curriculum range plot for x/y/z commands."""
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    all_scalar_tags = set(accumulator.Tags().get("scalars", []))
    required_tags = {tag for pair in CURRICULUM_TAGS.values() for tag in pair}
    missing_tags = sorted(required_tags - all_scalar_tags)
    if missing_tags:
        missing_display = "\n".join(f"- {tag}" for tag in missing_tags)
        raise ValueError(f"Missing required scalar tags:\n{missing_display}")

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    figure.suptitle("Curriculum Command Ranges vs Iteration", fontsize=14)

    for axis, (metric_name, (min_tag, max_tag)) in zip(axes, CURRICULUM_TAGS.items()):
        color_theme = METRIC_THEME_COLORS.get(metric_name, {})
        min_color = color_theme.get("min", "#1f77b4")
        max_color = color_theme.get("max", "#d62728")
        band_color = color_theme.get("band", "#2ca02c")
        theme_name = color_theme.get("theme", "Default")

        min_steps, min_values = load_scalar_series(accumulator, min_tag)
        max_steps, max_values = load_scalar_series(accumulator, max_tag)
        steps, min_aligned, max_aligned = align_series(min_steps, min_values, max_steps, max_values)

        if steps.size == 0:
            axis.set_title(f"{metric_name} (no data)")
            axis.grid(alpha=0.25)
            continue

        valid = np.isfinite(min_aligned) & np.isfinite(max_aligned)
        axis.plot(steps, max_aligned, label="max", linewidth=1.7, color=max_color)
        axis.plot(steps, min_aligned, label="min", linewidth=1.7, color=min_color)
        axis.fill_between(
            steps,
            min_aligned,
            max_aligned,
            where=valid,
            alpha=0.24,
            color=band_color,
            label="range",
        )

        axis.set_ylabel(metric_name)
        axis.set_title(f"{metric_name} ({theme_name})")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        order = ["max", "min", "range"]
        order_index = {label: idx for idx, label in enumerate(order)}
        ordered = sorted(
            zip(handles, labels),
            key=lambda item: order_index.get(item[1], len(order_index)),
        )
        ordered_handles = [item[0] for item in ordered]
        ordered_labels = [item[1] for item in ordered]
        axis.legend(ordered_handles, ordered_labels, loc="center right")

    if step_min is not None or step_max is not None:
        left = step_min if step_min is not None else None
        right = step_max if step_max is not None else None
        axes[-1].set_xlim(left=left, right=right)

    axes[-1].set_xlabel("Iteration (step)")
    figure.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "curriculum_cmd_ranges.png"
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Curriculum command ranges (min/max and range band) from TensorBoard events.",
    )
    parser.add_argument(
        "--event_file",
        type=Path,
        required=True,
        help="Path to one events.out.tfevents.* file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for the chart. Default: <event_file_parent>/charts",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for saved figure.",
    )
    parser.add_argument(
        "--step_min",
        type=int,
        default=None,
        help="Optional lower bound of iteration step shown on x-axis.",
    )
    parser.add_argument(
        "--step_max",
        type=int,
        default=None,
        help="Optional upper bound of iteration step shown on x-axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_file = args.event_file
    output_dir = args.output_dir if args.output_dir is not None else event_file.parent / "charts"

    if not event_file.exists():
        raise FileNotFoundError(f"Event file does not exist: {event_file}")

    output_path = plot_curriculum_ranges(
        event_file=event_file,
        output_dir=output_dir,
        dpi=args.dpi,
        step_min=args.step_min,
        step_max=args.step_max,
    )
    print(f"Chart saved to: {output_path}")


if __name__ == "__main__":
    main()
