"""
E1 Binding Ensemble Inference Script.

This script performs inference on test data using K-fold cross-validation models:
1. Loads K best models from cross-validation
2. Runs inference on test set using each model
3. Aggregates predictions (mean probabilities across models)
4. Applies global threshold from OOF predictions
5. Evaluates and visualizes results

Usage:
    python finetune/infer_e1_binding.py \
        --run_dir results/e1_binding/2025-12-09-13-11 \
        --test_fasta /path/to/test_data \
        --output_dir results/test_evaluation

    # Or use config's data paths for test:
    python finetune/infer_e1_binding.py \
        --run_dir results/e1_binding/2025-12-09-13-11 \
        --use_val_as_test
"""

import os
import sys

# Add src directory to Python path so E1 module can be imported without installing the package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from torch.amp import autocast
from torch.utils.data import DataLoader

from training.e1_checkpoint_utils import load_ensemble_models
from training.e1_binding_dataset import (
    create_binding_datasets_from_config,
    E1BindingDataset,
    process_binding_fasta_file,
)
from training.e1_binding_collator import E1DataCollatorForResidueClassification
from training.e1_joint_collator import E1DataCollatorForJointBindingMLM
from training.metrics import find_optimal_threshold, high_recall_auprc
from training.visualization import plot_threshold_analysis
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryRecall,
    BinaryPrecision,
    BinaryConfusionMatrix,
)


def setup_logging(output_dir: str):
    """Setup logging to file and console."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir) / "inference.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def load_config(run_dir: Path) -> Dict[str, Any]:
    """Load training config from run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loaded config from {config_path}")
    return config


def load_global_thresholds(run_dir: Path) -> Dict[str, Any]:
    """Load global thresholds from run directory."""
    thresholds_path = run_dir / "global_thresholds.yaml"
    if not thresholds_path.exists():
        logging.warning(f"Global thresholds not found at {thresholds_path}")
        return None

    with open(thresholds_path, "r") as f:
        thresholds_data = yaml.safe_load(f)

    logging.info(f"Loaded global thresholds from {thresholds_path}")
    logging.info(f"  Threshold method: {thresholds_data.get('threshold_method')}")
    for ion, data in thresholds_data.get("thresholds", {}).items():
        logging.info(f"  {ion}: threshold={data['threshold']:.4f}")

    return thresholds_data


def create_test_dataset(
    config: Dict[str, Any],
    ion_type: str,
    test_fasta: Optional[str] = None,
    msa_dir: Optional[str] = None,
) -> E1BindingDataset:
    """
    Create a test dataset from config or explicit paths.

    Args:
        config: Training configuration
        ion_type: Ion type to create dataset for
        test_fasta: Optional explicit test FASTA path (supports {ION} placeholder)
        msa_dir: Optional explicit MSA directory (supports {ION} placeholder)

    Returns:
        E1BindingDataset configured for test/inference
    """
    data_conf = config.get("data", {})
    training_conf = config.get("training", {})

    # Determine FASTA path
    if test_fasta:
        if "{ION}" in test_fasta:
            fasta_path = test_fasta.replace("{ION}", ion_type)
        else:
            fasta_path = test_fasta
    else:
        # Use training config path pattern
        base_path = data_conf.get("fasta_path", "")
        if "{ION}" in base_path:
            fasta_path = base_path.replace("{ION}", ion_type)
        else:
            fasta_path = f"{base_path}/{ion_type}_test.fasta"

    # Determine MSA directory
    if msa_dir:
        if "{ION}" in msa_dir:
            msa_path = msa_dir.replace("{ION}", ion_type)
        else:
            msa_path = msa_dir
    else:
        msa_base = data_conf.get("msa_dir")
        if msa_base:
            if "{ION}" in msa_base:
                msa_path = msa_base.replace("{ION}", ion_type)
            else:
                msa_path = f"{msa_base}/{ion_type}"
        else:
            msa_path = None

    # Use validation MSA sampling config for inference (deterministic)
    val_msa_config = training_conf.get(
        "validation_msa_sampling", training_conf.get("msa_sampling", {})
    )

    dataset = E1BindingDataset(
        fasta_path=fasta_path,
        msa_dir=msa_path,
        ion_type=ion_type,
        max_num_samples=val_msa_config.get("max_num_samples", 128),
        max_token_length=val_msa_config.get("max_token_length", 12288),
        max_query_similarity=val_msa_config.get("max_query_similarity", 0.95),
        min_query_similarity=val_msa_config.get("min_query_similarity", 0.0),
        neighbor_similarity_lower_bound=val_msa_config.get(
            "neighbor_similarity_lower_bound", 0.8
        ),
        seed=config.get("general", {}).get("seed", 42),
        is_validation=True,  # Use fixed seed for reproducibility
    )

    return dataset


def run_inference_single_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    ion: str,
    device: torch.device,
    use_bf16: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a single model.

    Returns:
        Dictionary with 'probs', 'labels', 'ids' arrays
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16 and torch.cuda.is_available(),
            ):
                outputs = model(
                    input_ids=batch_device["input_ids"],
                    within_seq_position_ids=batch_device["within_seq_position_ids"],
                    global_position_ids=batch_device["global_position_ids"],
                    sequence_ids=batch_device["sequence_ids"],
                    ion=ion,
                    labels=None,  # No labels needed for inference
                    label_mask=batch_device.get("label_mask"),
                )

            # Get predictions
            logits = outputs.logits
            probs = torch.sigmoid(logits)
            mask = batch_device["label_mask"]

            # Extract valid predictions
            for sample_idx in range(probs.shape[0]):
                sample_mask = mask[sample_idx].cpu()
                sample_probs = probs[sample_idx][sample_mask].cpu().numpy()
                sample_labels = batch["binding_labels"][sample_idx][
                    sample_mask.cpu()
                ].numpy()

                # Get protein ID
                protein_ids = batch.get("protein_ids", [])
                if protein_ids and sample_idx < len(protein_ids):
                    protein_id = protein_ids[sample_idx]
                else:
                    protein_id = f"sample_{sample_idx}"

                # Create residue IDs
                residue_ids = [
                    f"{protein_id}:{pos}" for pos in range(len(sample_probs))
                ]

                all_probs.extend(sample_probs.tolist())
                all_labels.extend(sample_labels.tolist())
                all_ids.extend(residue_ids)

    return {
        "probs": np.array(all_probs),
        "labels": np.array(all_labels),
        "ids": np.array(all_ids, dtype=object),
    }


def aggregate_predictions(
    model_predictions: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Aggregate predictions from multiple models by averaging probabilities.

    Args:
        model_predictions: List of prediction dicts from each model

    Returns:
        Aggregated predictions with mean probabilities
    """
    if not model_predictions:
        raise ValueError("No model predictions to aggregate")

    # All models should have the same IDs
    reference_ids = model_predictions[0]["ids"]
    labels = model_predictions[0]["labels"]

    # Stack probabilities from all models
    all_probs = np.stack([pred["probs"] for pred in model_predictions], axis=0)

    # Compute mean probability
    mean_probs = np.mean(all_probs, axis=0)

    return {
        "probs": mean_probs,
        "labels": labels,
        "ids": reference_ids,
        "num_models": len(model_predictions),
        "std_probs": np.std(all_probs, axis=0),  # Also track uncertainty
    }


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute classification metrics at a given threshold."""
    probs_tensor = torch.tensor(probs).float()
    labels_tensor = torch.tensor(labels).long()

    # Threshold-independent metrics
    auc_metric = BinaryAUROC(thresholds=None)
    auprc_metric = BinaryAveragePrecision(thresholds=None)

    # Threshold-dependent metrics
    f1_metric = BinaryF1Score(threshold=threshold)
    mcc_metric = BinaryMatthewsCorrCoef(threshold=threshold)
    recall_metric = BinaryRecall(threshold=threshold)
    precision_metric = BinaryPrecision(threshold=threshold)
    cm_metric = BinaryConfusionMatrix(threshold=threshold)

    metrics = {
        "auc": auc_metric(probs_tensor, labels_tensor).item(),
        "auprc": auprc_metric(probs_tensor, labels_tensor).item(),
        "high_recall_auprc_07": high_recall_auprc(
            labels_tensor, probs_tensor, recall_threshold=0.7
        ),
        "high_recall_auprc_08": high_recall_auprc(
            labels_tensor, probs_tensor, recall_threshold=0.8
        ),
        "threshold": threshold,
        "f1": f1_metric(probs_tensor, labels_tensor).item(),
        "mcc": mcc_metric(probs_tensor, labels_tensor).item(),
        "recall": recall_metric(probs_tensor, labels_tensor).item(),
        "precision": precision_metric(probs_tensor, labels_tensor).item(),
        "confusion_matrix": cm_metric(probs_tensor, labels_tensor).numpy().tolist(),
        "num_samples": len(labels),
        "num_positive": int(np.sum(labels)),
        "num_negative": int(len(labels) - np.sum(labels)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="E1 Binding Ensemble Inference on Test Data"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (contains fold_*/, config.yaml, global_thresholds.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save test metrics and plots (default: {run_dir}/test_results)",
    )
    parser.add_argument(
        "--test_fasta",
        type=str,
        default=None,
        help="Path to test FASTA file (supports {ION} placeholder, e.g., /data/{ION}_test.fasta)",
    )
    parser.add_argument(
        "--msa_dir",
        type=str,
        default=None,
        help="Path to MSA directory for test data (supports {ION} placeholder)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=None,
        help="Number of folds to load (default: from config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size (default: 4)",
    )
    parser.add_argument(
        "--use_val_as_test",
        action="store_true",
        help="Use validation data as test data (for debugging/sanity check)",
    )

    args = parser.parse_args()

    # Setup paths
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(str(output_dir))

    logging.info("=" * 80)
    logging.info("E1 Binding Ensemble Inference")
    logging.info("=" * 80)
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Output directory: {output_dir}")

    # Load config and thresholds
    config = load_config(run_dir)
    thresholds_data = load_global_thresholds(run_dir)

    # Get ion list and num_folds from config
    training_conf = config.get("training", {})
    ion_list = training_conf.get("ions", ["CA", "ZN", "MG"])
    num_folds = args.num_folds or training_conf.get("num_folds", 5)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Ions: {ion_list}")
    logging.info(f"Num folds: {num_folds}")

    # Load ensemble models
    logging.info("\nLoading ensemble models...")
    models = load_ensemble_models(
        checkpoint_dir=str(run_dir),
        num_folds=num_folds,
        device=device,
        model_dtype="bfloat16",
        mlm_weight=training_conf.get("mlm_weight", 0.0),
    )
    logging.info(f"Loaded {len(models)} models")

    # Create collator
    mlm_weight = training_conf.get("mlm_weight", 0.0)
    val_msa_config = training_conf.get(
        "validation_msa_sampling", training_conf.get("msa_sampling", {})
    )

    if mlm_weight and mlm_weight > 0:
        collator = E1DataCollatorForJointBindingMLM(
            mlm_probability=training_conf.get("mlm_probability", 0.15),
            max_total_tokens=val_msa_config.get("max_token_length", 12288),
            max_query_tokens=val_msa_config.get("max_query_length", 2048),
            ignore_index=-100,
            label_smoothing=0.0,
        )
    else:
        collator = E1DataCollatorForResidueClassification(
            max_total_tokens=val_msa_config.get("max_token_length", 12288),
            max_query_tokens=val_msa_config.get("max_query_length", 2048),
            label_smoothing=0.0,
        )

    # Run inference for each ion
    all_metrics = []
    all_predictions = {}

    for ion in ion_list:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing ion: {ion}")
        logging.info(f"{'='*60}")

        # Create test dataset
        try:
            dataset = create_test_dataset(
                config=config,
                ion_type=ion,
                test_fasta=args.test_fasta,
                msa_dir=args.msa_dir,
            )
        except FileNotFoundError as e:
            logging.warning(f"Could not create dataset for {ion}: {e}")
            continue

        if len(dataset) == 0:
            logging.warning(f"Empty dataset for {ion}, skipping")
            continue

        logging.info(f"Test dataset size: {len(dataset)} samples")

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

        # Run inference for each model
        logging.info(f"Running inference with {len(models)} models...")
        model_predictions = []

        for fold_idx, model in enumerate(models, 1):
            logging.info(f"  Fold {fold_idx}...")
            preds = run_inference_single_model(
                model=model,
                dataloader=dataloader,
                ion=ion,
                device=device,
                use_bf16=True,
            )
            model_predictions.append(preds)
            logging.info(f"    Samples: {len(preds['probs'])}")

        # Aggregate predictions
        logging.info("Aggregating predictions...")
        aggregated = aggregate_predictions(model_predictions)

        # Get threshold for this ion
        if thresholds_data and ion in thresholds_data.get("thresholds", {}):
            threshold = thresholds_data["thresholds"][ion]["threshold"]
            threshold_method = thresholds_data.get("threshold_method", "youden")
        else:
            # Compute threshold from test data if not available
            logging.warning(f"No threshold found for {ion}, computing from test data")
            threshold_result = find_optimal_threshold(
                predictions=aggregated["probs"],
                labels=aggregated["labels"],
                method=training_conf.get("threshold_method", "youden"),
            )
            threshold = threshold_result["threshold"]
            threshold_method = training_conf.get("threshold_method", "youden")

        logging.info(f"Using threshold: {threshold:.4f} (method: {threshold_method})")

        # Compute metrics
        metrics = compute_metrics(
            probs=aggregated["probs"],
            labels=aggregated["labels"],
            threshold=threshold,
        )
        metrics["ion"] = ion
        metrics["threshold_method"] = threshold_method
        all_metrics.append(metrics)

        # Log metrics
        logging.info(f"\nTest Metrics for {ion}:")
        logging.info(f"  AUC: {metrics['auc']:.4f}")
        logging.info(f"  AUPRC: {metrics['auprc']:.4f}")
        logging.info(
            f"  High-recall AUPRC (R>=0.7): {metrics['high_recall_auprc_07']:.4f}"
        )
        logging.info(
            f"  High-recall AUPRC (R>=0.8): {metrics['high_recall_auprc_08']:.4f}"
        )
        logging.info(f"  F1 @ {threshold:.3f}: {metrics['f1']:.4f}")
        logging.info(f"  MCC @ {threshold:.3f}: {metrics['mcc']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        cm = metrics["confusion_matrix"]
        logging.info(
            f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}"
        )

        # Store predictions
        all_predictions[ion] = {
            "probs": aggregated["probs"],
            "labels": aggregated["labels"],
            "ids": aggregated["ids"],
            "std_probs": aggregated["std_probs"],
        }

        # Generate visualization
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Generating threshold analysis plots...")
        plot_threshold_analysis(
            outputs=aggregated["probs"],
            labels=aggregated["labels"],
            save_dir=plots_dir,
            fold_number=None,
            optimal_threshold=threshold,
            threshold_method_used=threshold_method,
            logger=logging,
            ion=ion,
        )

    # Save metrics to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_csv_path = output_dir / "test_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        logging.info(f"\nMetrics saved to {metrics_csv_path}")

    # Save summary to YAML
    summary = {
        "run_dir": str(run_dir),
        "num_folds": len(models),
        "threshold_method": (
            thresholds_data.get("threshold_method") if thresholds_data else "computed"
        ),
        "ions": {},
    }

    for metrics in all_metrics:
        ion = metrics["ion"]
        summary["ions"][ion] = {
            "auprc": metrics["auprc"],
            "high_recall_auprc_07": metrics["high_recall_auprc_07"],
            "threshold": metrics["threshold"],
            "f1": metrics["f1"],
            "mcc": metrics["mcc"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
        }

    # Compute overall averages
    if all_metrics:
        summary["overall"] = {
            "mean_auprc": float(np.mean([m["auprc"] for m in all_metrics])),
            "mean_hra_07": float(
                np.mean([m["high_recall_auprc_07"] for m in all_metrics])
            ),
            "mean_f1": float(np.mean([m["f1"] for m in all_metrics])),
            "mean_mcc": float(np.mean([m["mcc"] for m in all_metrics])),
        }

    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    logging.info(f"Summary saved to {summary_path}")

    # Save raw predictions
    predictions_path = output_dir / "test_predictions.npz"
    save_dict = {}
    for ion, preds in all_predictions.items():
        for key, value in preds.items():
            save_dict[f"{ion}_{key}"] = value
    np.savez(predictions_path, **save_dict)
    logging.info(f"Predictions saved to {predictions_path}")

    logging.info("\n" + "=" * 80)
    logging.info("Inference completed successfully!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
