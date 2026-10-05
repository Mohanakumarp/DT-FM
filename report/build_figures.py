"""Regenerate the report's vector diagrams and Git-history progress chart."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
BLUE = "#244a74"
PALE = "#edf3f8"
GRAY = "#52606d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


def box(ax, x, y, w, h, text, fill=PALE, size=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025",
                               linewidth=1.3, edgecolor=BLUE, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color="#172b3a", fontsize=size, linespacing=1.5)


def arrow(ax, start, end, text=None, offset=(0, 0.12), color=BLUE):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=15, linewidth=1.5, color=color))
    if text:
        ax.text((start[0]+end[0])/2+offset[0], (start[1]+end[1])/2+offset[1],
                text, ha="center", va="bottom", fontsize=12, color=color)


def architecture():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set(xlim=(0, 12), ylim=(0, 6.8))
    ax.axis("off")
    ax.text(6, 6.55, "Implemented scheduling and training path", ha="center",
            fontsize=16, weight="bold", color=BLUE)
    box(ax, .15, 4.95, 3.15, .95, "Network probes\nRTT + bandwidth")
    box(ax, 4.40, 4.95, 3.20, .95, "Compute + memory\nProfiles + budgets")
    box(ax, 8.70, 4.95, 3.15, .95, "Launch manifest\nOrder + counts")
    arrow(ax, (3.35, 5.43), (4.35, 5.43))
    arrow(ax, (7.65, 5.43), (8.65, 5.43))
    box(ax, .45, 2.45, 2.75, 1.40, "Rank 0\nEmbed + layers\nCPU / GPU")
    box(ax, 4.60, 2.45, 2.75, 1.40, "Middle rank(s)\nModel layers\nCPU / GPU")
    box(ax, 8.75, 2.45, 2.75, 1.40, "Last rank\nLayers + head\nLoss / labels")
    arrow(ax, (10.28, 4.9), (10.28, 3.9), "Launch", offset=(.47, 0))
    for left, right in [(3.25, 4.55), (7.4, 8.7)]:
        arrow(ax, (left, 3.45), (right, 3.45), "A", offset=(0,.12))
        arrow(ax, (right, 2.8), (left, 2.8), "G", offset=(0,-.38))
    ax.text(6, 2.07, "A: activations. G: gradients. Transport: CPU-staged Gloo.",
            ha="center", color=GRAY, fontsize=11)
    box(ax, .45, .40, 5.05, .95, "Boundary controller\nReprofile / propose / migrate", size=12)
    box(ax, 6.55, .40, 4.95, .95, "Per-rank metrics\nJSON / MLflow / lab reports", size=12)
    arrow(ax, (1.82, 2.40), (1.82, 1.40))
    arrow(ax, (10.12, 2.40), (10.12, 1.40))
    ax.text(6, .03, "Fixed membership; automatic recovery from a crashed training rank is future work.",
            ha="center", fontsize=11, color=GRAY)
    fig.savefig(FIGURES / "implemented_architecture.pdf")
    plt.close(fig)


def migration():
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set(xlim=(0, 12), ylim=(0, 5.3))
    ax.axis("off")
    ax.text(6, 5.03, "Transactional layer migration at an optimizer-step boundary",
            ha="center", fontsize=13.5, weight="bold", color=BLUE)
    steps = [("Drain pipeline\nFinish step", .2),
             ("Prepare stage\nReserve RAM", 3.25),
             ("Transfer state\nCheck SHA-256", 6.3),
             ("All-rank vote\nCommit", 9.35)]
    for text, x in steps:
        box(ax, x, 3.0, 2.45, 1.25, text, size=12)
    for x in [2.7, 5.75, 8.8]:
        arrow(ax, (x, 3.62), (x+.48, 3.62))
    box(ax, .75, .90, 4.70, 1.15, "Preparation / capacity failure\nKeep old stage + SGD state",
        fill="#fff3e4", size=12)
    box(ax, 6.65, .90, 4.70, 1.15, "Successful commit\nSwap stage + optimizer\nContinue training", size=12)
    arrow(ax, (4.48, 2.95), (3.1, 2.1), "Rollback", offset=(-.6, .02), color="#a85a10")
    arrow(ax, (10.57, 2.95), (9.0, 2.1), "Commit", offset=(.5, .02))
    ax.text(6, .32, "Preserve trained weights, supported SGD momentum, RNG and completed-step count.",
            ha="center", fontsize=11, color=GRAY)
    fig.savefig(FIGURES / "migration_transaction.pdf")
    plt.close(fig)


def progress():
    evidence_path = ROOT / "evidence.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        counts = Counter(commit["date"] for commit in evidence["commits"])
    else:
        log = subprocess.check_output(["git", "log", "--format=%ad",
                                       "--date=short", "cda948f..e0c0507"],
                                      cwd=ROOT.parent, text=True)
        counts = Counter(log.splitlines())
    dates, cumulative, total = [], [], 0
    for date, count in sorted(counts.items()):
        total += count
        dates.append(datetime.fromisoformat(date))
        cumulative.append(total)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(dates, cumulative, where="post", color=BLUE, linewidth=2.2)
    ax.scatter(dates, cumulative, color=BLUE, s=23, zorder=3)
    ax.annotate(f"{total} team commits", (dates[-1], cumulative[-1]),
                xytext=(-115, -27), textcoords="offset points", weight="bold", color=BLUE)
    ax.set(title="Recorded development progress beyond the original upstream baseline",
           ylabel="Cumulative team commits", xlabel="2026")
    ax.set_ylim(0, total+4)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(rotation=25)
    fig.savefig(FIGURES / "development_progress.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    architecture()
    migration()
    progress()
