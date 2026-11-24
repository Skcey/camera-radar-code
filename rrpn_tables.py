"""Rebuild the manuscript rows from the three server runs."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def average(rows, metric):
    return mean(item[metric] for item in rows) * 100


def comparison_row(method, camera, fusion):
    measures = ("ap", "ap50")
    camera_scores = [average(camera, name) for name in measures]
    fusion_scores = [average(fusion, name) for name in measures]
    gains = [score - baseline for score, baseline in zip(fusion_scores, camera_scores)]
    return row(method, [" / ".join(f"{v:.2f}" for v in camera_scores),
                        " / ".join(f"{v:.2f}" for v in fusion_scores),
                        " / ".join(f"{v:+.2f}" for v in gains)])


def verify_camera(retest, original):
    for name in ("ap", "ap50", "ap75", "ar100"):
        if abs(retest[name] - original[name]) > 1e-8:
            raise ValueError(f"The frozen camera differs from the archived baseline for {name}")


def row(name, values):
    return name + " & " + " & ".join(values) + r" \\"


def gap(ours, reference, name):
    delta = average(ours, name) - average(reference, name)
    if abs(delta) < .005:
        return "持平"
    return ("高 " if delta > 0 else "低 ") + f"{abs(delta):.2f} 个百分点"


def build(results, output):
    archived_external = read(ROOT / "results/expansion/nuscenes_unseen.json")["seeds"]
    audit = read(results / "candidate_audit.json")
    mini, unseen, mini_ours, unseen_ours = [], [], [], []
    mini_camera, unseen_camera = [], []
    counts = []
    for seed in SEEDS:
        archived_mini = read(ROOT / f"results/seed_{seed}.json")["metrics"]
        data = read(results / f"seed_{seed}.json")
        if (data["seed"], data["blur"]) != (seed, 4):
            raise ValueError(f"Expected seed {seed} and sigma=4")
        if data["anchor_source_revision"] != audit["revision"]:
            raise ValueError(f"Unexpected RRPN code revision for seed {seed}")
        mini_result = data["evaluation"]["mini"]
        unseen_result = data["evaluation"]["nuscenes_unseen"]
        if (mini_result["frames"], unseen_result["frames"]) != (81, 1191):
            raise ValueError(f"Unexpected frame counts for seed {seed}")
        verify_camera(mini_result["rgb"], archived_mini["rgb"])
        verify_camera(unseen_result["rgb"], archived_external[str(seed)]["rgb"])
        mini.append(mini_result["rrpn_refined"])
        counts.append(mini_result["box_count_mean"])
        unseen.append(unseen_result["rrpn_refined"])
        mini_ours.append(archived_mini["radar_refined"])
        unseen_ours.append(archived_external[str(seed)]["radar_refined"])
        mini_camera.append(archived_mini["rgb"])
        unseen_camera.append(archived_external[str(seed)]["rgb"])

    full_label = "RRPN 候选＋同型精修"
    short_label = "RRPN 候选精修"
    cols = ("ap", "ap50", "ap75", "ar100")
    scores = [item["ap"] * 100 for item in mini]
    paper_count = audit["paper_templates"]["mini"]["box_count_mean"]
    comparison_label = "RRPN 候选复测"
    commands = {
        "RrpnMiniRow": row(full_label, [f"{average(mini, col):.2f}" for col in cols]),
        "RrpnSeedRow": row(short_label, [f"{score:.2f}" for score in scores]
                          + [fr"${mean(scores):.2f}\pm{stdev(scores):.2f}$"]),
        "RrpnExternalRow": row(short_label, [f"{average(unseen, col):.2f}" for col in cols[:2]]
                               + ["—", "—"]),
        "StrongBlurComparisonRows": "\n".join((
            r"\multicolumn{4}{l}{nuScenes mini} \\",
            comparison_row(comparison_label, mini_camera, mini),
            comparison_row("本文方法", mini_camera, mini_ours),
            r"\midrule",
            r"\multicolumn{4}{l}{nuScenes 未见场景} \\",
            comparison_row(comparison_label, unseen_camera, unseen),
            comparison_row("本文方法", unseen_camera, unseen_ours),
        )),
        "RrpnMiniFinding": (
            f"在 mini 上，本文模板和 RRPN 对照平均每帧分别生成 {paper_count:.1f} 和 {mean(counts):.1f} 个雷达候选。"
            f"相对该对照，本文方法的 mAP50--95 {gap(mini_ours, mini, 'ap')}，"
            f"AP50 {gap(mini_ours, mini, 'ap50')}。两组均使用相同的最终输出上限。"
        ),
        "RrpnExternalFinding": (
            f"在 30 个未见 nuScenes 场景中，相对 RRPN 候选对照，本文方法的 mAP50--95 "
            f"{gap(unseen_ours, unseen, 'ap')}，AP50 {gap(unseen_ours, unseen, 'ap50')}。"
            "两组均沿用 mini 的权重与分数系数。"
        ),
        "RrpnConclusion": (
            f"相对 RRPN 候选对照，本文方法的 mAP50--95 在 mini 上"
            f"{gap(mini_ours, mini, 'ap')}，在未见 nuScenes 场景上{gap(unseen_ours, unseen, 'ap')}。"
        ),
    }
    output.write_text("\n".join(
        f"\\newcommand{{\\{name}}}{{%\n{value}\n}}" for name, value in commands.items()
    ) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/rrpn")
    parser.add_argument("--out", type=Path, default=ROOT / "paper/figures/rrpn_reference.tex")
    args = parser.parse_args()
    build(args.results, args.out)
