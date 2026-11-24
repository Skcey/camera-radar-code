"""Validate ablation experiments and regenerate the paper table from three seeds."""
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEEDS = (42, 43, 44)
DATASETS = {"mini": 81, "nuscenes_unseen": 1191, "astyx": 546}
METRICS = ("ap", "ap50", "ap75")
LABELS = {
    "full": "完整方法",
    "no_echo": "去除回波属性",
    "no_visual_roi": "去除图像 RoI 特征",
    "no_sharpness": "去除清晰度特征",
    "no_regression": "去除边框回归",
    "no_quality": "去除质量加权",
    "radar_geometry_only": "雷达候选（统一输入）",
    "image_geometry_only": "图像候选（统一输入）",
}


def read(relative):
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def archived(seed, dataset):
    if dataset == "mini":
        return read(f"seed_{seed}.json")["metrics"]
    path = "astyx.json" if dataset == "astyx" else "nuscenes_unseen.json"
    return read(f"expansion/{path}")["seeds"][str(seed)]


def verify(seed, dataset, scores):
    reference = archived(seed, dataset)
    for run, original in (("full", "radar_refined"), ("image_geometry_only", "image_refined")):
        for metric in (*METRICS, "ar100"):
            if abs(scores[run][metric] - reference[original][metric]) > 1e-8:
                raise ValueError(f"Baseline mismatch in {seed=} {dataset=} {run=} {metric=}")


def difference(delta):
    if abs(delta) < .005:
        return "持平"
    return f"{'高' if delta > 0 else '低'} {abs(delta):.2f} 个百分点"


def main():
    runs = [read(f"ablation/seed_{seed}.json") for seed in SEEDS]
    for seed, run in zip(SEEDS, runs):
        if run["seed"] != seed or run["blur"] != 4 or run["train_frames"] != 323:
            raise ValueError(f"Unexpected evaluation settings for seed {seed}")
        for name, frames in DATASETS.items():
            block = run["datasets"][name]
            if block["frames"] != frames or set(block["scores"]) != set(LABELS):
                raise ValueError(f"Incomplete ablation in seed {seed} on {name}")
            verify(seed, name, block["scores"])

    def score(dataset, variant, metric):
        return 100 * mean(run["datasets"][dataset]["scores"][variant][metric] for run in runs)

    rows = []
    for variant, label in LABELS.items():
        if variant == "radar_geometry_only":
            rows.append(r"\midrule")
        values = [score(dataset, variant, metric) for dataset in DATASETS
                  for metric in (METRICS if dataset == "mini" else METRICS[:2])]
        rows.append(" & ".join((label, *(f"{value:.2f}" for value in values))) + r" \\")

    def delta(dataset, variant, metric="ap"):
        return score(dataset, "full", metric) - score(dataset, variant, metric)

    core = ("no_echo", "no_visual_roi", "no_sharpness", "no_regression", "no_quality")
    mini_effects = [
        f"{LABELS[name]}后，mAP50--95 较完整方法{difference(-delta('mini', name))}"
        for name in core
    ]
    candidate_gaps = [
        difference(score(dataset, "radar_geometry_only", "ap")
                   - score(dataset, "image_geometry_only", "ap"))
        for dataset in DATASETS
    ]
    analysis = (
        "在 mini 上，" + "。".join(mini_effects) + "。"
        f"关闭边框回归后的 AP75 为 {score('mini', 'no_regression', 'ap75'):.2f}\\%，"
        f"完整方法为 {score('mini', 'full', 'ap75'):.2f}\\%。"
    )
    transfer = (
        "使用统一的精修输入时，雷达候选组在 mini、nuScenes 未见场景和 Astyx 上的"
        " AP50--95 分别比图像候选组" + "、".join(candidate_gaps)
        + "。该对照衡量候选来源的影响，候选数量随各自生成规则变化。"
    )
    external = (
        "在 nuScenes 未见场景和 Astyx 上，去除边框回归后，AP75 分别比完整方法"
        + "、".join(difference(-delta(dataset, "no_regression", "ap75"))
                    for dataset in ("nuscenes_unseen", "astyx"))
        + "。去除质量加权后，AP50--95 分别比完整方法"
        + "、".join(difference(-delta(dataset, "no_quality"))
                    for dataset in ("nuscenes_unseen", "astyx"))
        + "。去除回波属性后，两套扩展数据的 AP50--95 分别比完整方法"
        + "、".join(difference(-delta(dataset, "no_echo"))
                    for dataset in ("nuscenes_unseen", "astyx"))
        + "。回波属性的收益随验证集而变。Astyx 在完整组已经将 RCS 和速度置零，"
        + "该集的回波属性消融仅额外移除了精修头的深度输入。"
    )
    zero_scale = []
    for name, label in LABELS.items():
        seeds = [str(seed) for seed, run in zip(SEEDS, runs)
                 if run["variants"].get(name, {}).get("scale") == 0]
        if seeds:
            zero_scale.append(f"{label}（种子 {'、'.join(seeds)}）")
    if zero_scale:
        analysis += (
            "开发集为" + "、".join(zero_scale)
            + "选出了零分数系数，这些运行在验证时回退为纯相机输出。"
        )
    output = ROOT / "paper/figures/ablation.tex"
    commands = {"AblationRows": "\n".join(rows), "AblationMiniFinding": analysis,
                "AblationExternalFinding": external, "AblationCandidateFinding": transfer}
    output.write_text("\n".join(
        f"\\newcommand{{\\{name}}}{{%\n{value}\n}}" for name, value in commands.items()
    ) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
