# HOW-TO: Generate MSAs and E1 Embeddings

This document covers only:
1. Getting MSAs (`.a3m`) with MMseqs2.
2. Getting E1 embeddings once your input FASTA and MSAs are ready.

## 1) Get MSAs (MMseqs2 GPU workflow)

### 1.1 Install MMseqs2 with GPU support

```bash
# Create build environment
conda create -n mmseqs-gpu -c conda-forge -c nvidia \
  cuda-nvcc cuda-cudart-dev cuda-version=12.6 \
  cmake gxx_linux-64 make git pkg-config zlib rust
conda activate mmseqs-gpu

# Build MMseqs2
git clone https://github.com/soedinglab/MMseqs2.git
cd MMseqs2
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=. \
  -DENABLE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="native" ..
make -j"$(nproc)"
make install
export PATH="$(pwd)/bin/:$PATH"
```

### 1.2 Build NR databases

```bash
# CPU database (source DB)
mmseqs databases nr /data/mmseqs_databases/nr/nr tmp

# GPU-padded database for search
mmseqs makepaddedseqdb /data/mmseqs_databases/nr/nr /data/mmseqs_databases/nr/nr_gpu
```

### 1.3 Search homologs

```bash
# query.fasta = your input query sequences
mmseqs createdb query.fasta queryDB

mmseqs search queryDB /data/mmseqs_databases/nr/nr_gpu results/resultDB tmp \
  --gpu 1 --max-seqs 4000
```

### 1.4 Convert results to per-sequence `.a3m`

```bash
# Use CPU NR DB here to avoid padded artifacts
mmseqs result2msa queryDB /data/mmseqs_databases/nr/nr results/resultDB results/msaDB \
  --msa-format-mode 6

mkdir -p results/a3m_files
mmseqs unpackdb results/msaDB results/a3m_files \
  --unpack-name-mode 0 --unpack-suffix .a3m
```

Output MSA directory for embedding step: `results/a3m_files`

## 2) Get embeddings once FASTA + MSA are ready

### 2.1 Required layout

- FASTA file: `/path/to/sequences.fasta`
- MSA directory: `/path/to/a3m_files`
- Every FASTA record ID must have a matching MSA filename:
  - FASTA ID `SEQ123` -> `/path/to/a3m_files/SEQ123.a3m`

### 2.2 Run embedding extraction

```bash
python embeddings.py \
  --input /path/to/sequences.fasta \
  --output_dir /path/to/embeddings_out \
  --msa_dir /path/to/a3m_files \
  --model Profluent-Bio/E1-600m \
  --device auto \
  --num_variants 1 \
  --mask_prob_range 0.05 0.15
```

### 2.3 Optional: use LoRA-finetuned adapter

```bash
python embeddings.py \
  --input /path/to/sequences.fasta \
  --output_dir /path/to/embeddings_out \
  --msa_dir /path/to/a3m_files \
  --adapter_checkpoint results/e1_lora_checkpoints/<run>/checkpoint-<step> \
  --num_variants 8 \
  --mask_prob_range 0.05 0.15 \
  --model Profluent-Bio/E1-600m \
  --device auto
```

### 2.4 Output format

For sequence ID `X`:

- `--num_variants 1` -> `X.npy` with shape `[L, D]`
- `--num_variants > 1` -> `X.npz` with key `embeddings` and shape `[V, L, D]`
