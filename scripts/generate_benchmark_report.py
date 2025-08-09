#!/usr/bin/env python3
"""Generate benchmark report from ground truth and detection results.

This script compares detection results against ground truth definitions
to calculate accuracy metrics and generate a human-readable report.
"""

import json
from datetime import datetime
from pathlib import Path

import yaml


def load_ground_truth(yaml_path: Path) -> dict:
    """Load ground truth definitions from YAML."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_detection_results(json_path: Path) -> list[dict]:
    """Load detection results from JSON."""
    if not json_path.exists():
        return []
    with open(json_path) as f:
        return json.load(f)


def calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union for two bounding boxes."""
    r1_start, r1_end, c1_start, c1_end = box1
    r2_start, r2_end, c2_start, c2_end = box2

    # Calculate intersection
    r_inter_start = max(r1_start, r2_start)
    r_inter_end = min(r1_end, r2_end)
    c_inter_start = max(c1_start, c2_start)
    c_inter_end = min(c1_end, c2_end)

    if r_inter_end < r_inter_start or c_inter_end < c_inter_start:
        return 0.0  # No intersection

    # Calculate areas
    inter_area = (r_inter_end - r_inter_start + 1) * (c_inter_end - c_inter_start + 1)
    box1_area = (r1_end - r1_start + 1) * (c1_end - c1_start + 1)
    box2_area = (r2_end - r2_start + 1) * (c2_end - c2_start + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_detection(detected_tables: list[dict], ground_truth_tables: list[dict]) -> dict:
    """Evaluate detection against ground truth."""
    metrics = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "ious": [],
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }

    # Convert to coordinate tuples
    detected_coords = [(t["start_row"], t["end_row"], t["start_col"], t["end_col"]) for t in detected_tables]
    gt_coords = [(t["start_row"], t["end_row"], t["start_col"], t["end_col"]) for t in ground_truth_tables]

    # Track matches
    matched_gt = set()
    matched_det = set()

    # Find best matches
    for i, det_box in enumerate(detected_coords):
        best_iou = 0.0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_coords):
            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        # Consider it a match if IoU > 0.5
        if best_iou > 0.5:
            matched_det.add(i)
            matched_gt.add(best_gt_idx)
            metrics["ious"].append(best_iou)
            metrics["true_positives"] += 1
        else:
            metrics["false_positives"] += 1

    # Count false negatives
    metrics["false_negatives"] = len(gt_coords) - len(matched_gt)

    # Calculate precision, recall, F1
    if metrics["true_positives"] + metrics["false_positives"] > 0:
        metrics["precision"] = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_positives"])

    if metrics["true_positives"] + metrics["false_negatives"] > 0:
        metrics["recall"] = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_negatives"])

    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1_score"] = (
            2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
        )

    # Calculate average IoU
    metrics["avg_iou"] = sum(metrics["ious"]) / len(metrics["ious"]) if metrics["ious"] else 0.0

    return metrics


def generate_markdown_report(ground_truth: dict, results: list[dict]) -> str:
    """Generate a markdown benchmark report."""
    report = []
    report.append("# Table Detection Benchmark Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Group results by model
    model_results = {}
    for result in results:
        model = result["model"]
        if model not in model_results:
            model_results[model] = []
        model_results[model].append(result)

    # Overall summary
    report.append("## Summary")
    report.append(f"- Total detection runs: {len(results)}")
    report.append(f"- Models tested: {', '.join(model_results.keys())}")
    report.append(f"- Sheets evaluated: {len(ground_truth['sheets'])}\n")

    # Model comparison table
    report.append("## Model Performance")
    report.append("\n| Model | Runs | Avg Precision | Avg Recall | Avg F1 | Avg IoU | Avg Tokens | Avg Cost |")
    report.append("|-------|------|---------------|------------|--------|---------|------------|----------|")

    for model, model_runs in model_results.items():
        # Calculate aggregate metrics
        all_metrics = []
        total_tokens = 0
        total_cost = 0.0

        for run in model_runs:
            sheet_idx = run["sheet_index"]

            # Find ground truth for this sheet
            gt_sheet = None
            for sheet in ground_truth["sheets"]:
                if sheet["sheet_index"] == sheet_idx:
                    gt_sheet = sheet
                    break

            if gt_sheet:
                metrics = evaluate_detection(run["tables_detected"], gt_sheet["expected_tables"])
                all_metrics.append(metrics)

            total_tokens += run["token_usage"]["total"]
            total_cost += run["cost_usd"]

        # Calculate averages
        if all_metrics:
            avg_precision = sum(m["precision"] for m in all_metrics) / len(all_metrics)
            avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
            avg_f1 = sum(m["f1_score"] for m in all_metrics) / len(all_metrics)
            avg_iou = sum(m["avg_iou"] for m in all_metrics) / len(all_metrics)
        else:
            avg_precision = avg_recall = avg_f1 = avg_iou = 0.0

        avg_tokens = total_tokens / len(model_runs) if model_runs else 0
        avg_cost = total_cost / len(model_runs) if model_runs else 0.0

        report.append(
            f"| {model} | {len(model_runs)} | {avg_precision:.3f} | "
            f"{avg_recall:.3f} | {avg_f1:.3f} | {avg_iou:.3f} | "
            f"{avg_tokens:.0f} | ${avg_cost:.4f} |"
        )

    # Sheet-by-sheet analysis
    report.append("\n## Sheet-by-Sheet Analysis")

    for sheet in ground_truth["sheets"]:
        sheet_idx = sheet["sheet_index"]
        sheet_name = sheet["sheet_name"]
        expected_tables = len(sheet["expected_tables"])

        report.append(f"\n### Sheet {sheet_idx}: {sheet_name}")
        report.append(f"- Expected tables: {expected_tables}")
        if sheet.get("notes"):
            report.append(f"- Notes: {sheet['notes']}")

        # Find all runs for this sheet
        sheet_runs = [r for r in results if r["sheet_index"] == sheet_idx]

        if sheet_runs:
            report.append("\n| Model | Tables Found | Precision | Recall | F1 | Avg IoU |")
            report.append("|-------|--------------|-----------|--------|-----|---------|")

            for run in sheet_runs:
                metrics = evaluate_detection(run["tables_detected"], sheet["expected_tables"])

                report.append(
                    f"| {run['model']} | {run['table_count']} | "
                    f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
                    f"{metrics['f1_score']:.3f} | {metrics['avg_iou']:.3f} |"
                )
        else:
            report.append("\n*No detection runs for this sheet*")

    # Common issues
    report.append("\n## Common Issues")

    # Analyze failure patterns
    overlapping_tables = 0
    missed_side_by_side = 0
    over_detection = 0
    under_detection = 0

    for result in results:
        sheet_idx = result["sheet_index"]

        # Find ground truth
        gt_sheet = None
        for sheet in ground_truth["sheets"]:
            if sheet["sheet_index"] == sheet_idx:
                gt_sheet = sheet
                break

        if gt_sheet:
            expected = len(gt_sheet["expected_tables"])
            detected = result["table_count"]

            if detected > expected:
                over_detection += 1
            elif detected < expected:
                under_detection += 1

                # Check if it's a side-by-side case
                if expected == 2 and detected == 1:
                    if "side-by-side" in gt_sheet.get("notes", "").lower():
                        missed_side_by_side += 1

    report.append(f"- Over-detection (detected more tables than expected): {over_detection} cases")
    report.append(f"- Under-detection (missed tables): {under_detection} cases")
    report.append(f"- Missed side-by-side tables: {missed_side_by_side} cases")

    return "\n".join(report)


def main():
    """Generate benchmark report."""
    # Paths
    ground_truth_path = Path("docs/benchmarks/ground_truth/business_accounting.yaml")
    results_path = Path("outputs/benchmarks/detection_results.json")
    report_path = Path("docs/benchmarks/reports/benchmark_report.md")

    # Load data
    ground_truth = load_ground_truth(ground_truth_path)
    results = load_detection_results(results_path)

    if not results:
        print("No detection results found. Run some detections first.")
        return

    # Generate report
    report = generate_markdown_report(ground_truth, results)

    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Benchmark report generated: {report_path}")

    # Also print summary to console
    print("\n=== Quick Summary ===")
    model_counts = {}
    for result in results:
        model = result["model"]
        model_counts[model] = model_counts.get(model, 0) + 1

    for model, count in model_counts.items():
        print(f"- {model}: {count} runs")


if __name__ == "__main__":
    main()
