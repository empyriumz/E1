import random
import argparse
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path

# E1 imports
from E1.batch_preparer import E1BatchPreparer
from E1.modeling import E1ForMaskedLM
from E1.msa_sampling import sample_context


# --- Helper Functions ---
def create_masked_sequence_variants(
    sequence,
    num_variants=2,
    mask_token="?",
    mask_prob_range=(0.05, 0.15),
):
    """
    Create masked variants of a protein sequence for contrastive learning.

    This function creates masked variants using ? token for E1 models.
    This enables unsupervised contrastive learning where different masked versions
    of the same sequence serve as positive pairs.

    Args:
        sequence (str): Original protein sequence
        num_variants (int): Number of masked variants to create (default: 2)
        mask_token (str): Token to use for masking (default: "?" for E1)
        mask_prob_range (tuple): Range for masking probability (min, max). If min equals max,
                               a fixed probability is used.

    Returns:
        list: List of masked sequence variants (includes original + masked versions)
    """
    # Parameter validation
    if num_variants < 1:
        raise ValueError("num_variants must be >= 1")
    if not (
        isinstance(mask_prob_range, (list, tuple))
        and len(mask_prob_range) == 2
        and 0 <= mask_prob_range[0] <= mask_prob_range[1] <= 1
    ):
        raise ValueError(
            "mask_prob_range must be a tuple or list of two floats [min, max] between 0 and 1, with min <= max"
        )

    variants = [sequence]  # Always include the original sequence

    # Create masked variants
    for _ in range(num_variants - 1):
        # Determine masking probability for this variant
        variant_mask_prob = random.uniform(mask_prob_range[0], mask_prob_range[1])

        masked_sequence_parts = []
        for amino_acid in sequence:
            if random.random() < variant_mask_prob:
                masked_sequence_parts.append(mask_token)
            else:
                masked_sequence_parts.append(amino_acid)
        variants.append("".join(masked_sequence_parts))

    return variants


def extract_e1_embeddings(
    model,
    batch_preparer,
    sequences,
    output_dir,
    msa_dir,
    device_str="auto",
    num_variants=1,
    mask_prob_range=(0.05, 0.15),
):
    """
    Extract per-residue embeddings for each sequence using E1 model.
    Uses random masking to create stochastic variants for contrastive learning.
    Preserves full per-residue embeddings without pooling.

    Args:
        model: The loaded E1ForMaskedLM model
        batch_preparer: E1BatchPreparer instance
        sequences: List of (id, sequence) tuples
        output_dir: Directory to save the embeddings
        msa_dir: Directory containing MSA files (.a3m format)
        device_str (str): Device specification ('cpu', 'cuda:N', or 'auto')
        num_variants: Number of variants per sequence (1 = no augmentation, >1 = include masked variants)
        mask_prob_range: Tuple with (min, max) for masking probability.
                         If min equals max, a fixed probability is used.

    Returns:
        tuple: (n_processed, n_failed, failed_ids)
            - n_processed: Number of sequences processed
            - n_failed: Number of sequences that failed
            - failed_ids: Dictionary mapping failed sequence IDs to their error messages
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Validate and get device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cpu":
        device = torch.device("cpu")
    elif device_str.startswith("cuda:"):
        device = torch.device(device_str)
    else:
        raise ValueError(
            f"Invalid device specification: {device_str}. Use 'cpu', 'cuda:N', or 'auto'"
        )

    # Track failures
    failed_ids = {}
    n_processed = 0

    print(
        f"Extracting embeddings with input masking variants: {num_variants} variant(s) per sequence"
    )
    if num_variants > 1:
        min_prob, max_prob = mask_prob_range
        if min_prob == max_prob:
            print(f"Using fixed masking probability: {min_prob*100:.2f}% with ? token")
        else:
            print(
                f"Using random masking probability between {min_prob*100:.2f}% and {max_prob*100:.2f}% with ? token"
            )
    else:
        print("Using deterministic embedding (no masking).")

    # Use tqdm to show progress with success/failure stats
    progress_bar = tqdm(sequences, desc="Extracting embeddings")
    for seq_id, query_seq in progress_bar:
        # Use .npz format when multiple variants are generated, .npy for single variant
        if num_variants > 1:
            output_path = Path(output_dir) / f"{seq_id}.npz"
        else:
            output_path = Path(output_dir) / f"{seq_id}.npy"

        # Skip if already exists
        if output_path.exists():
            n_processed += 1
            continue

        try:
            # Create masked variants if needed
            if num_variants > 1:
                sequence_variants = create_masked_sequence_variants(
                    query_seq,
                    num_variants=num_variants,
                    mask_token=batch_preparer.mask_token,
                    mask_prob_range=mask_prob_range,
                )
            else:
                sequence_variants = [query_seq]

            # Store embeddings for all variants
            all_embeddings = []

            # Prepare MSA context once using the original sequence (if available)
            # This ensures consistent context across all variants
            has_context = False
            context_str = None
            if msa_dir:
                msa_path = Path(msa_dir) / f"{seq_id}.a3m"
                if msa_path.exists():
                    try:
                        context_str, _ = sample_context(
                            msa_path=str(msa_path),
                            max_num_samples=511,
                            max_token_length=14784,
                            max_query_similarity=0.95,
                            min_query_similarity=0.0,
                            neighbor_similarity_lower_bound=0.8,
                            seed=42,
                        )
                        has_context = True
                    except Exception as e:
                        print(f"Error sampling context for {seq_id}: {e}")
                        context_str = None
                        has_context = False

            for variant_seq in sequence_variants:
                # 1. Prepare Input (Homologs + Query)
                # Use original sequence for MSA context, but masked variant for the query sequence
                if has_context and context_str is not None:
                    input_str = context_str + "," + variant_seq
                else:
                    input_str = variant_seq

                # 2. Prepare Batch
                # batch_preparer expects a list of strings
                batch = batch_preparer.get_batch_kwargs([input_str], device=device)

                # 3. Model Inference
                with torch.no_grad():
                    with torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            within_seq_position_ids=batch["within_seq_position_ids"],
                            global_position_ids=batch["global_position_ids"],
                            sequence_ids=batch["sequence_ids"],
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                        )

                # 4. Extract Embedding for the Query Sequence Only
                # The model outputs embeddings for ALL homologs + query.
                # We need to mask out the homologs and special tokens.

                embeddings = outputs.embeddings  # Shape: (1, Total_Length, Hidden_Dim)

                # Mask 1: Select tokens belonging to the LAST sequence (the query)
                # sequence_ids tracks which protein a token belongs to. The max ID is the last one.
                last_seq_mask = (
                    batch["sequence_ids"]
                    == batch["sequence_ids"].max(dim=1)[0][:, None]
                )

                # Mask 2: Select actual residues (exclude <bos>, <eos>, separators)
                residue_mask = ~(
                    batch_preparer.get_boundary_token_mask(batch["input_ids"])
                )

                # Combined Mask
                final_mask = last_seq_mask & residue_mask

                # Apply mask to get the vectors (Batch size is 1, so take index 0)
                # Resulting shape: (Length_of_Query, Hidden_Dim)
                # Convert to float32 before numpy conversion (bfloat16 is not supported by numpy)
                query_embedding = embeddings[0, final_mask[0]].cpu().float().numpy()

                # 5. Safety Check
                # The number of embedding vectors must match the original sequence length
                # Note: For masked variants, we compare against the original query_seq length
                if query_embedding.shape[0] != len(query_seq):
                    print(
                        f"Warning: {seq_id} length mismatch. Seq: {len(query_seq)}, Emb: {query_embedding.shape[0]}"
                    )

                all_embeddings.append(query_embedding)

            # Stack all variants into [num_variants, seq_len, embed_dim]
            if num_variants > 1:
                stacked_embeddings = np.stack(all_embeddings)
                # Save the stacked per-residue embeddings
                np.savez(output_path, embeddings=stacked_embeddings)
            else:
                # Save single embedding as .npy for backward compatibility
                np.save(output_path, all_embeddings[0])

            n_processed += 1

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            failed_ids[seq_id] = error_msg

        # Update progress bar description with stats
        progress_bar.set_description(
            f"Processing: {n_processed} succeeded, {len(failed_ids)} failed"
        )

    # Print summary of failures if any occurred
    if failed_ids:
        print("\nFailed sequences:")
        for seq_id, error_msg in failed_ids.items():
            print(f"  {seq_id}: {error_msg}")

    print(f"\nProcessing complete:")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Successfully processed: {n_processed}")
    print(f"  Failed: {len(failed_ids)}")

    return n_processed, len(failed_ids), failed_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract E1 embeddings from sequence file"
    )
    parser.add_argument(
        "--input", required=True, help="Input sequence file (FASTA format)"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for embeddings"
    )
    parser.add_argument(
        "--msa_dir",
        default=None,
        help="Directory containing MSA files (.a3m format). If not provided, single sequence mode will be used.",
    )
    parser.add_argument(
        "--model",
        default="Profluent-Bio/E1-600m",
        choices=[
            "Profluent-Bio/E1-150m",
            "Profluent-Bio/E1-300m",
            "Profluent-Bio/E1-600m",
        ],
        help="E1 model to use",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run the model on. Use 'cpu' for CPU, 'cuda:N' for specific GPU (N=0,1,etc), or 'auto' to auto-select",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=1,
        help="Number of variants per sequence for contrastive learning (1 = no augmentation, >1 = include masked variants)",
    )
    parser.add_argument(
        "--mask_prob_range",
        type=float,
        nargs=2,
        default=(0.05, 0.15),
        help="Range for masking probability (min, max). Use two identical values for a fixed probability (e.g., 0.1 0.1).",
    )
    args = parser.parse_args()

    # Load the model
    print(f"Loading E1 model: {args.model}")
    try:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif args.device == "cpu":
            device = torch.device("cpu")
        elif args.device.startswith("cuda:"):
            device = torch.device(args.device)
        else:
            raise ValueError(f"Invalid device specification: {args.device}")

        model = E1ForMaskedLM.from_pretrained(args.model).to(device).eval()
        print(f"Model loaded successfully on {device}")
        batch_preparer = E1BatchPreparer()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load sequences
    print(f"Loading sequences from: {args.input}")
    sequences = [
        (record.id, str(record.seq)) for record in SeqIO.parse(args.input, "fasta")
    ]
    print(f"Loaded {len(sequences)} sequences")

    # Set MSA directory (use None if not provided to indicate no MSA)
    msa_dir = args.msa_dir if args.msa_dir else None

    # Extract embeddings
    print(f"Extracting embeddings to: {args.output_dir}")
    n_processed, n_failed, failed_ids = extract_e1_embeddings(
        model=model,
        batch_preparer=batch_preparer,
        sequences=sequences,
        output_dir=args.output_dir,
        msa_dir=msa_dir,
        device_str=args.device,
        num_variants=args.num_variants,
        mask_prob_range=tuple(args.mask_prob_range),
    )

    if n_failed > 0:
        print(
            "\nWarning: Some sequences failed to process. Check the output above for details."
        )


if __name__ == "__main__":
    main()
