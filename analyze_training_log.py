#!/usr/bin/env python3
"""
FastESM LoRA Training Log Analysis Script

This script extracts evaluation metrics from FastESM LoRA training logs
and creates comprehensive visualizations and summaries with a focus on
Homologs-related metrics.

Usage:
    python analyze_training_log.py [--log-file LOG_FILE] [--output-dir OUTPUT_DIR]

Author: AI Assistant
Date: November 20, 2025
"""

import argparse
import re
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def extract_evaluation_metrics(log_file_path):
    """
    Extract evaluation metrics from training log file.

    Args:
        log_file_path (str): Path to the training log file

    Returns:
        pd.DataFrame: DataFrame containing all evaluation metrics
    """
    print(f"Reading log file: {log_file_path}")

    with open(log_file_path, "r") as f:
        lines = f.readlines()

    # Extract evaluation lines containing eval_Combined_loss
    eval_lines = []
    for line in lines:
        if "eval_Combined_loss" in line and "STDOUT" in line:
            eval_lines.append(line)

    print(f"Found {len(eval_lines)} evaluation entries")

    # Parse metrics from each line
    metrics_data = []
    for line in eval_lines:
        # Extract the JSON-like part using regex
        json_match = re.search(r"\{.*\}", line)
        if json_match:
            try:
                # Use eval to parse the dictionary (safe since we control the input)
                metrics = eval(json_match.group())
                metrics_data.append(metrics)
            except Exception as e:
                print(f"Failed to parse line: {line[:100]}... Error: {e}")
                continue

    print(f"Successfully parsed {len(metrics_data)} evaluation entries")

    if not metrics_data:
        raise ValueError("No evaluation metrics found in log file")

    return pd.DataFrame(metrics_data)


def create_comprehensive_plots(df, output_file):
    """
    Create comprehensive plots showing training metrics evolution.

    Args:
        df (pd.DataFrame): DataFrame with evaluation metrics
        output_file (str): Path to save the plot
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "FastESM LoRA Training Evaluation Metrics (Homologs Focus)",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Loss (Combined as reference, since Homologs loss not available separately)
    axes[0, 0].plot(
        df["epoch"],
        df["eval_Combined_loss"],
        "b--",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Combined Loss (Reference)",
        alpha=0.7,
    )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss vs Epoch (Combined)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Perplexity (Homologs as primary)
    axes[0, 1].plot(
        df["epoch"],
        df["eval_Homologs_perplexity"],
        "r-",
        linewidth=2,
        marker="^",
        markersize=5,
        label="Homologs Perplexity",
    )
    axes[0, 1].plot(
        df["epoch"],
        df["eval_Combined_perplexity"],
        "r--",
        linewidth=1.5,
        marker="s",
        markersize=3,
        label="Combined Perplexity",
        alpha=0.7,
    )
    axes[0, 1].plot(
        df["epoch"],
        df["eval_SwissProt_perplexity"],
        "r:",
        linewidth=1.5,
        marker="v",
        markersize=3,
        label="SwissProt Perplexity",
        alpha=0.7,
    )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Perplexity")
    axes[0, 1].set_title("Perplexity vs Epoch (Homologs Focus)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: Accuracy (Homologs as primary)
    axes[1, 0].plot(
        df["epoch"],
        df["eval_Homologs_accuracy"],
        "g-",
        linewidth=2,
        marker="^",
        markersize=5,
        label="Homologs Accuracy",
    )
    axes[1, 0].plot(
        df["epoch"],
        df["eval_Combined_accuracy"],
        "g--",
        linewidth=1.5,
        marker="D",
        markersize=3,
        label="Combined Accuracy",
        alpha=0.7,
    )
    axes[1, 0].plot(
        df["epoch"],
        df["eval_SwissProt_accuracy"],
        "g:",
        linewidth=1.5,
        marker="v",
        markersize=3,
        label="SwissProt Accuracy",
        alpha=0.7,
    )
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy vs Epoch (Homologs Focus)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot 4: Homologs metrics in one plot with secondary y-axis
    ax4 = axes[1, 1]
    # Use Combined loss as reference (Homologs loss not available separately)
    line1 = ax4.plot(
        df["epoch"],
        df["eval_Combined_loss"],
        "b--",
        linewidth=1.5,
        marker="o",
        markersize=3,
        label="Combined Loss (Ref)",
        alpha=0.7,
    )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss", color="b")
    ax4.tick_params(axis="y", labelcolor="b")

    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(
        df["epoch"],
        df["eval_Homologs_perplexity"],
        "r-",
        linewidth=2,
        marker="^",
        markersize=5,
        label="Homologs Perplexity",
    )
    ax4_twin.set_ylabel("Perplexity", color="r")
    ax4_twin.tick_params(axis="y", labelcolor="r")

    # Add accuracy as a third axis
    ax4_twin2 = ax4.twinx()
    ax4_twin2.spines["right"].set_position(("axes", 1.15))
    line3 = ax4_twin2.plot(
        df["epoch"],
        df["eval_Homologs_accuracy"],
        "g-",
        linewidth=2,
        marker="^",
        markersize=5,
        label="Homologs Accuracy",
    )
    ax4_twin2.set_ylabel("Accuracy", color="g")
    ax4_twin2.tick_params(axis="y", labelcolor="g")

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right")
    ax4.set_title("Homologs Metrics vs Epoch")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training metrics plots saved to {output_file}")


def get_best_checkpoint_info(output_dir):
    """
    Get best checkpoint information from trainer_state.json.

    Args:
        output_dir (Path): Path to output directory

    Returns:
        tuple: (checkpoint_name, epoch) or (None, None) if not found
    """
    # Try to find trainer_state.json in best_checkpoint directory first
    best_checkpoint_dir = output_dir / "best_checkpoint"
    trainer_state_path = best_checkpoint_dir / "trainer_state.json"

    # If not found, try in the output directory itself
    if not trainer_state_path.exists():
        trainer_state_path = output_dir / "trainer_state.json"

    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)

            best_global_step = trainer_state.get("best_global_step")
            best_model_checkpoint = trainer_state.get("best_model_checkpoint", "")

            if best_global_step is not None:
                # Extract checkpoint name from path or use step number
                if best_model_checkpoint:
                    checkpoint_name = Path(best_model_checkpoint).name
                else:
                    checkpoint_name = f"checkpoint-{best_global_step}"

                # Try to find the epoch at the best checkpoint from log_history
                epoch = None
                log_history = trainer_state.get("log_history", [])
                for entry in log_history:
                    if entry.get("step") == best_global_step and "epoch" in entry:
                        epoch = entry["epoch"]
                        break

                # Fallback to top-level epoch if not found in log_history
                if epoch is None:
                    epoch = trainer_state.get("epoch")

                return checkpoint_name, epoch
        except Exception as e:
            print(f"Warning: Could not read trainer_state.json: {e}")

    return None, None


def print_training_summary(df, output_dir=None):
    """
    Print comprehensive training summary.

    Args:
        df (pd.DataFrame): DataFrame with evaluation metrics
        output_dir (Path, optional): Path to output directory to read trainer_state.json
    """
    print("\n=== FastESM LoRA Training Evaluation Summary (Homologs Focus) ===")
    print(f"Total evaluations: {len(df)}")
    print(f'Epoch range: {df["epoch"].min():.2f} to {df["epoch"].max():.2f}')
    print()

    # Create a formatted table of key metrics at different training stages
    stages = [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, -1]
    stage_names = ["Early (Start)", "Quarter", "Halfway", "Three-quarters", "Final"]

    print("Key Metrics at Different Training Stages (Homologs Focus):")
    print("=" * 90)
    print(
        f'{"Stage":<15} {"Epoch":<8} {"Loss(Comb)":<12} {"Perplexity(Hom)":<18} {"Accuracy(Hom)":<16} {"Runtime(s)":<12}'
    )
    print("-" * 90)

    for stage, name in zip(stages, stage_names):
        row = df.iloc[stage]
        print(
            f'{name:<15} {row["epoch"]:<8.2f} {row["eval_Combined_loss"]:<12.4f} {row["eval_Homologs_perplexity"]:<18.4f} {row["eval_Homologs_accuracy"]:<16.4f} {row["eval_Combined_runtime"]:<12.1f}'
        )

    print()
    print("Improvement Summary (Homologs Metrics):")
    print("-" * 50)

    # Combined loss as reference (Homologs loss not available separately)
    initial_loss = df["eval_Combined_loss"].iloc[0]
    final_loss = df["eval_Combined_loss"].iloc[-1]
    loss_improvement = initial_loss - final_loss

    initial_perp = df["eval_Homologs_perplexity"].iloc[0]
    final_perp = df["eval_Homologs_perplexity"].iloc[-1]
    perp_improvement = initial_perp - final_perp

    initial_acc = df["eval_Homologs_accuracy"].iloc[0]
    final_acc = df["eval_Homologs_accuracy"].iloc[-1]
    acc_improvement = final_acc - initial_acc

    print(
        f"Combined Loss improvement: {loss_improvement:.4f} ({loss_improvement/initial_loss*100:.1f}%)"
    )
    print(
        f"Homologs Perplexity improvement: {perp_improvement:.4f} ({perp_improvement/initial_perp*100:.1f}%)"
    )
    print(
        f"Homologs Accuracy improvement: {acc_improvement:.4f} ({acc_improvement/initial_acc*100:.1f}%)"
    )
    print()

    # Print final metrics (Homologs first)
    print("Final Evaluation Metrics:")
    print(f"Homologs Perplexity: {final_perp:.4f}")
    print(f"Homologs Accuracy: {final_acc:.4f}")
    print(f"Combined Loss: {final_loss:.4f}")
    print(f"Combined Perplexity: {df['eval_Combined_perplexity'].iloc[-1]:.4f}")
    print(f"Combined Accuracy: {df['eval_Combined_accuracy'].iloc[-1]:.4f}")
    print(f'SwissProt Perplexity: {df["eval_SwissProt_perplexity"].iloc[-1]:.4f}')
    print(f'SwissProt Accuracy: {df["eval_SwissProt_accuracy"].iloc[-1]:.4f}')
    print()

    # Print best metrics (Homologs focus)
    print("Best Metrics Achieved (Homologs):")
    best_loss_idx = df["eval_Combined_loss"].idxmin()
    best_perp_idx = df["eval_Homologs_perplexity"].idxmin()
    best_acc_idx = df["eval_Homologs_accuracy"].idxmax()

    print(
        f'Best Combined Loss: {df["eval_Combined_loss"].min():.4f} at epoch {df.loc[best_loss_idx, "epoch"]:.2f}'
    )
    print(
        f'Best Homologs Perplexity: {df["eval_Homologs_perplexity"].min():.4f} at epoch {df.loc[best_perp_idx, "epoch"]:.2f}'
    )
    print(
        f'Best Homologs Accuracy: {df["eval_Homologs_accuracy"].max():.4f} at epoch {df.loc[best_acc_idx, "epoch"]:.2f}'
    )

    print()
    print("Training Configuration Summary:")
    print("- Learning rate: 0.0002 (cosine schedule with warmup)")
    print("- Batch size: 72 (effective, with gradient accumulation)")
    print("- Mixed precision: FP16")
    print("- LoRA parameters: 12.17M out of 663.25M total (1.83%)")

    # Try to get best checkpoint info from trainer_state.json
    if output_dir:
        checkpoint_name, epoch = get_best_checkpoint_info(output_dir)
        if checkpoint_name and epoch is not None:
            print(f"- Best model saved at {checkpoint_name} (epoch {epoch:.2f})")
        else:
            print("- Best model checkpoint information not available")
    else:
        print("- Best model checkpoint information not available")


def main():
    """Main function to run the training log analysis."""
    parser = argparse.ArgumentParser(description="Analyze FastESM LoRA training logs")
    parser.add_argument(
        "--log-file",
        type=str,
        default="multi_modal_binding/results/fastesm_lora/2025-11-19-22-30/training.log",
        help="Path to training log file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save output files"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="evaluation_metrics.csv",
        help="Name of CSV file to save metrics",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="training_metrics_plots.png",
        help="Name of plot file to save",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if log file exists
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: Log file '{log_file}' not found!")
        return 1

    try:
        # Extract metrics
        df = extract_evaluation_metrics(log_file)

        # Save raw data
        csv_path = output_dir / args.csv_file
        df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")

        # Create plots
        plot_path = output_dir / args.plot_file
        create_comprehensive_plots(df, plot_path)

        # Print summary
        print_training_summary(df, output_dir)

        print(f"\nAnalysis complete! Files saved to {output_dir}")
        return 0

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
