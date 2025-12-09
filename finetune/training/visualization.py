import torch
import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    precision_score,
)
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_training_curves(metrics_tracker, fold=None, save_dir=None, stage=None):
    """Plot and save training curves for each ion separately.

    Args:
        metrics_tracker (MetricsTracker): Object containing training history per ion
        fold (int, optional): Current fold number. If None, uses "global" in filenames
        save_dir (str or Path): Directory to save plots
        stage (str, optional): Training stage name for two-stage training (e.g., 'combined', 'stage1', 'stage2')
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if save_dir is None:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Handle missing fold number
    fold_str = str(fold) if fold is not None else "global"
    fold_display = f"Fold {fold}" if fold is not None else "Global Training"

    # Create filename suffix based on stage
    stage_suffix = f"_stage_{stage}" if stage else ""
    title_suffix = f" - {stage.capitalize()} Stage" if stage else ""

    # Get all ions
    ions = metrics_tracker.get_all_ions()

    if not ions:
        logger.warning("No metrics found to plot")
        return

    # Plot curves for each ion separately
    for ion in ions:
        ion_metrics = metrics_tracker.get_ion_metrics(ion)
        train_metrics = ion_metrics["train_metrics"]
        val_metrics = ion_metrics["val_metrics"]

        if not train_metrics and not val_metrics:
            continue

        # Plot 1: Loss curves
        train_losses = [m.get("loss", 0) for m in train_metrics]
        val_losses = [m.get("loss", 0) for m in val_metrics]

        if train_losses or val_losses:
            plt.figure(figsize=(10, 6))
            if train_losses:
                plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
            if val_losses:
                plt.plot(val_losses, label="Validation Loss", marker="s", markersize=3)
            plt.title(
                f"Training and Validation Loss - {fold_display} [{ion}]{title_suffix}"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                save_dir / f"loss_fold_{fold_str}_{ion}{stage_suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

        # Plot 2: AUPRC and other metrics (if available)
        # Check all metric entries to find which metrics are available
        # (Stage 1 only has loss, Stage 2 has loss + classification metrics)
        if train_metrics and val_metrics:
            available_metrics = []

            # Check all metric entries to find available classification metrics
            # This handles the case where Stage 1 metrics don't have classification metrics
            all_metric_keys = set()
            for m in train_metrics + val_metrics:
                all_metric_keys.update(m.keys())

            # Check for binary classification metrics
            if "auprc" in all_metric_keys:
                available_metrics.append(("auprc", "AUPRC"))
            if "auc" in all_metric_keys:
                available_metrics.append(("auc", "AUC"))
            if "roc_auc" in all_metric_keys:
                available_metrics.append(("roc_auc", "ROC AUC"))

            # Plot available threshold-independent metrics
            # Only plot epochs where the metric exists (typically Stage 2 epochs)
            for metric_name, display_name in available_metrics:
                plt.figure(figsize=(10, 6))

                # Extract metrics only for epochs where they exist
                train_metric = []
                val_metric = []
                train_epochs = []
                val_epochs = []

                for epoch_idx, m in enumerate(train_metrics):
                    if metric_name in m:
                        train_metric.append(m[metric_name])
                        train_epochs.append(epoch_idx + 1)  # 1-based epoch numbering

                for epoch_idx, m in enumerate(val_metrics):
                    if metric_name in m:
                        val_metric.append(m[metric_name])
                        val_epochs.append(epoch_idx + 1)  # 1-based epoch numbering

                if train_metric:
                    plt.plot(
                        train_epochs,
                        train_metric,
                        label=f"Train {display_name}",
                        marker="o",
                        markersize=3,
                    )
                if val_metric:
                    plt.plot(
                        val_epochs,
                        val_metric,
                        label=f"Validation {display_name}",
                        marker="s",
                        markersize=3,
                    )
                plt.title(f"{display_name} - {fold_display} [{ion}]{title_suffix}")
                plt.xlabel("Epoch")
                plt.ylabel(display_name)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(
                    save_dir / f"{metric_name}_fold_{fold_str}_{ion}{stage_suffix}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()


def plot_threshold_analysis(
    outputs,
    labels,
    save_dir,
    fold_number=None,
    threshold_methods=None,
    target_recalls=None,
    optimal_threshold=None,  # Pre-calculated optimal threshold to mark on plots
    threshold_method_used=None,  # Method used to calculate the optimal_threshold
    logger=None,
    ion=None,  # Optional ion identifier for filename
):
    """
    Plot ROC and Precision-Recall curves with threshold analysis.

    This helps visualize the precision-recall trade-off and choose appropriate
    target_recall values for the recall_constrained threshold method.

    Args:
        outputs: Model outputs (logits or probabilities)
        labels: Ground truth labels
        save_dir: Directory to save plots
        fold_number: Current fold number for filename
        threshold_methods: List of threshold methods to compare
        target_recalls: List of target recall values to mark on plots
        optimal_threshold: Pre-calculated optimal threshold to mark prominently
        threshold_method_used: Method used to calculate optimal_threshold
        logger: Logger instance for logging results
    """
    if save_dir is None:
        return

    try:
        from .metrics import find_optimal_threshold

        # Convert outputs to numpy arrays first
        if isinstance(outputs, tuple):
            logits = outputs[0]  # First element is always logits
        else:
            logits = outputs

        # Convert to numpy if it's a tensor
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()

        # Handle variant dimensions - use only the first variant (original)
        if logits.ndim == 3:  # [batch_size, num_variants, num_classes]
            logits = logits[:, 0]  # Use original sequence (variant 0)
        elif logits.ndim == 2 and logits.shape[1] == 1:  # [batch_size, 1]
            logits = logits.squeeze(1)  # [batch_size]

        # Convert to probabilities if needed
        if np.max(logits) > 1.0 or np.min(logits) < 0.0:
            # Assume logits, apply sigmoid
            probs = 1.0 / (1.0 + np.exp(-logits))
        else:
            # Already probabilities
            probs = logits

        # Convert labels to numpy arrays and ensure they are integers
        if torch.is_tensor(labels):
            labels_np = labels.cpu().numpy().ravel()
        else:
            labels_np = np.array(labels).ravel()

        # Ensure labels are integers (not float32)
        labels_np = labels_np.astype(np.int64)

        probs = probs.ravel()

        # Default parameters
        if threshold_methods is None:
            threshold_methods = ["f1", "mcc", "youden"]
        if target_recalls is None:
            target_recalls = [0.7, 0.8]

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fold_str = f"_fold_{fold_number}" if fold_number is not None else ""
        ion_str = f"_{ion}" if ion is not None else ""

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(labels_np, probs)
        roc_auc = roc_auc_score(labels_np, probs)

        ax1.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax1.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random classifier",
        )

        # Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(labels_np, probs)
        avg_precision = average_precision_score(labels_np, probs)

        # Calculate high-recall AUPRC metrics
        try:
            from .metrics import high_recall_auprc

            hr_auprc_07 = high_recall_auprc(labels_np, probs, recall_threshold=0.7)
            hr_auprc_08 = high_recall_auprc(labels_np, probs, recall_threshold=0.8)

            # Create enhanced label with high-recall AUPRC info
            label = f"PR curve (AUPRC = {avg_precision:.3f})"
            label += (
                f"\nHigh-recall AUPRC: R≥0.7={hr_auprc_07:.3f}, R≥0.8={hr_auprc_08:.3f}"
            )
        except ImportError:
            label = f"PR curve (AUPRC = {avg_precision:.3f})"

        ax2.plot(
            recall,
            precision,
            color="darkorange",
            lw=2,
            label=label,
        )

        # Baseline precision (random classifier)
        baseline_precision = np.sum(labels_np) / len(labels_np)
        ax2.axhline(
            y=baseline_precision,
            color="navy",
            linestyle="--",
            lw=2,
            label=f"Random classifier (P = {baseline_precision:.3f})",
        )

        # Calculate and mark optimal thresholds
        colors = ["red", "green", "blue", "purple", "orange"]
        threshold_info = []

        # First, mark the pre-calculated optimal threshold if provided
        if optimal_threshold is not None:
            # Find corresponding points on curves
            roc_idx = np.argmin(np.abs(roc_thresholds - optimal_threshold))
            pr_idx = (
                np.argmin(np.abs(pr_thresholds - optimal_threshold))
                if len(pr_thresholds) > 0
                else 0
            )

            # Calculate metrics at this threshold
            y_pred = (probs >= optimal_threshold).astype(int)
            achieved_recall = recall_score(labels_np, y_pred)
            achieved_precision = precision_score(labels_np, y_pred)

            # Mark prominently on ROC curve
            if roc_idx < len(fpr):
                ax1.plot(
                    fpr[roc_idx],
                    tpr[roc_idx],
                    "*",
                    color="black",
                    markersize=15,
                    markeredgewidth=2,
                    markeredgecolor="white",
                    label=f"USED {threshold_method_used or 'threshold'}: T={optimal_threshold:.3f}",
                )

            # Mark prominently on PR curve
            ax2.plot(
                achieved_recall,
                achieved_precision,
                "*",
                color="black",
                markersize=15,
                markeredgewidth=2,
                markeredgecolor="white",
                label=f"USED {threshold_method_used or 'threshold'}: T={optimal_threshold:.3f}",
            )

            threshold_info.append(
                {
                    "method": f"USED_{threshold_method_used or 'threshold'}",
                    "threshold": optimal_threshold,
                    "precision": achieved_precision,
                    "recall": achieved_recall,
                }
            )

            if logger:
                logger.info(
                    f"Marked optimal threshold on plot: {optimal_threshold:.4f} "
                    f"(method: {threshold_method_used}, P={achieved_precision:.3f}, R={achieved_recall:.3f})"
                )

        for i, method in enumerate(threshold_methods):
            try:
                # find_optimal_threshold returns a dict with 'threshold' key
                threshold_result = find_optimal_threshold(
                    probs, labels_np, method=method
                )
                optimal_thresh = threshold_result["threshold"]

                # Define color for method
                color = colors[i % len(colors)]

                # Find corresponding points on curves
                roc_idx = np.argmin(np.abs(roc_thresholds - optimal_thresh))

                # Calculate metrics at this threshold
                y_pred = (probs >= optimal_thresh).astype(int)
                achieved_recall = recall_score(labels_np, y_pred)
                achieved_precision = precision_score(labels_np, y_pred)

                # Mark on ROC curve
                if roc_idx < len(fpr):
                    ax1.plot(
                        fpr[roc_idx],
                        tpr[roc_idx],
                        "o",
                        color=color,
                        markersize=8,
                        label=f"{method}: T={optimal_thresh:.3f}",
                    )

                # Mark on PR curve
                ax2.plot(
                    achieved_recall,
                    achieved_precision,
                    "o",
                    color=color,
                    markersize=8,
                    label=f"{method}: T={optimal_thresh:.3f}",
                )

                threshold_info.append(
                    {
                        "method": method,
                        "threshold": optimal_thresh,
                        "precision": achieved_precision,
                        "recall": achieved_recall,
                    }
                )

            except Exception as e:
                if logger:
                    logger.warning(f"Error calculating {method} threshold: {e}")

        # Format ROC plot
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve with Optimal Thresholds")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Format PR plot
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve with Optimal Thresholds")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = f"threshold_analysis{fold_str}{ion_str}.png"
        plt.savefig(save_dir / plot_filename, dpi=150, bbox_inches="tight")
        plt.close()

        # Log threshold analysis results
        if logger:
            ion_label = f" [{ion}]" if ion is not None else ""
            logger.info(f"\nThreshold Analysis Results{ion_label}:")
            logger.info("=" * 60)
            for info in threshold_info:
                logger.info(
                    f"{info['method']:20}: Threshold={info['threshold']:.4f}, "
                    f"Precision={info['precision']:.3f}, Recall={info['recall']:.3f}"
                )

        # Save threshold analysis to CSV
        if threshold_info:
            import pandas as pd

            df = pd.DataFrame(threshold_info)
            csv_filename = f"threshold_analysis{fold_str}{ion_str}.csv"
            df.to_csv(save_dir / csv_filename, index=False)
            if logger:
                logger.info(f"Threshold analysis saved to {save_dir / csv_filename}")

        if logger:
            logger.info(f"Threshold analysis plots saved to {save_dir / plot_filename}")

    except ImportError as e:
        if logger:
            logger.warning(f"Missing dependencies for threshold analysis plotting: {e}")
    except Exception as e:
        if logger:
            logger.warning(f"Error in threshold analysis plotting: {e}")
            import traceback

            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
