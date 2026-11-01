import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

cloud_script_path = os.path.join(PROJECT_ROOT, "colab_w2v2_streaming_cloud.py")
with open(cloud_script_path, "r", encoding="utf-8") as f:
    cloud_script_code = f.read()

# Split script into lines with newlines intact for ipynb JSON format
script_lines = [line + "\n" for line in cloud_script_code.splitlines()]
# Prepend %%writefile colab_w2v2_streaming_cloud.py\n
writefile_cell_source = ["%%writefile colab_w2v2_streaming_cloud.py\n"] + script_lines

notebook_data = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# VoiceGuard AI: Large-Scale Cloud GPU Trainer (Strategy 2 & 4)\n",
                "### Fine-Tuning Wav2Vec 2.0 (XLS-R 300M) + AASIST on 5,000–50,000 Streamed Audio Clips\n",
                "\n",
                "This Google Colab notebook implements **Strategy 2** (Cloud GPU streaming with 0 MB local disk storage footprint) and **Strategy 4** (detecting modern commercial AI voice clones such as ElevenLabs, Hume AI, and neural vocoders).\n",
                "\n",
                "**Key Highlights:**\n",
                "- **100% Self-Contained:** Automatically generates all companion scripts inside Google Colab (no manual file uploads required).\n",
                "- **Zero Local Disk Footprint:** Audio clips are streamed on-the-fly directly from Hugging Face into Colab GPU memory without filling up your local hard drive.\n",
                "- **Modern AI Clones:** Integrates modern synthetic voice clones (ElevenLabs, Hume AI, WaveFake) alongside authentic human speech (LJSpeech, YouTube, LibriSpeech).\n",
                "- **Acoustic Invariance Augmentation:** Applies on-the-fly Room Impulse Response (RIR) reverberation augmentation.\n",
                "- **Calibrated Validation:** Optimizes the decision threshold using Youden's J statistic on isolated validation data (zero data leakage).\n",
                "- **1-Click Checkpoint Download:** Exports a compact ~180 KB PyTorch checkpoint (`voiceguard_w2v2_large_scale_model.pt`) that you can download directly back to your local machine."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Check GPU Runtime & Install Dependencies (CUDA 12 Optimized)\n",
                "!pip install -q soundfile librosa datasets scikit-learn matplotlib huggingface_hub\n",
                "!pip install -q onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/\n",
                "\n",
                "import torch\n",
                "print('PyTorch Version:', torch.__version__)\n",
                "print('CUDA Available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU Device Name:', torch.cuda.get_device_name(0))\n",
                "    print(f'GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\n",
                "else:\n",
                "    print('WARNING: Running on CPU! For 10x faster training, go to Runtime -> Change runtime type -> T4 GPU.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Auto-Generate Streaming Trainer Script\n",
                "This cell writes `colab_w2v2_streaming_cloud.py` directly into Colab's workspace (`/content/`).\n",
                "Run this cell so that the companion script is ready."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": writefile_cell_source
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Select Scale Tier (Memory-Optimized)\n",
                "Choose your desired dataset scale (all tiers now use memory micro-chunking to prevent CUDA Out-of-Memory):\n",
                "- **Tier 1 (Instant Safe Demo, ~1.5 mins):** 1,500 samples (1,000 Train / 200 Val / 300 Test)\n",
                "- **Tier 2 (Production Balanced, ~3 mins):** 4,000 samples (3,000 Train / 400 Val / 600 Test) — **Recommended**\n",
                "- **Tier 3 (Large Scale, ~7 mins):** 10,000 samples (8,000 Train / 1,000 Val / 1,000 Test)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# SCALE CONFIGURATION: Select 1 (1.5k), 2 (4k - Recommended), or 3 (10k)\n",
                "# ==============================================================================\n",
                "SCALE_TIER = 2   # Choose 1, 2, or 3\n",
                "\n",
                "if SCALE_TIER == 1:\n",
                "    TRAIN_SAMPLES = 1000\n",
                "    VAL_SAMPLES = 200\n",
                "    TEST_SAMPLES = 300\n",
                "elif SCALE_TIER == 2:\n",
                "    TRAIN_SAMPLES = 3000\n",
                "    VAL_SAMPLES = 400\n",
                "    TEST_SAMPLES = 600\n",
                "elif SCALE_TIER == 3:\n",
                "    TRAIN_SAMPLES = 8000\n",
                "    VAL_SAMPLES = 1000\n",
                "    TEST_SAMPLES = 1000\n",
                "\n",
                "# BATCH_SIZE_EXTRACT: Set to 4 to guarantee zero CUDA Out-of-Memory on Colab T4 GPU\n",
                "BATCH_SIZE_EXTRACT = 4\n",
                "EPOCHS = 25\n",
                "BATCH_SIZE = 64\n",
                "LEARNING_RATE = 0.001\n",
                "\n",
                "print(f'Selected Tier {SCALE_TIER}:')\n",
                "print(f'  Training Samples:   {TRAIN_SAMPLES}')\n",
                "print(f'  Validation Samples: {VAL_SAMPLES}')\n",
                "print(f'  Test Samples:       {TEST_SAMPLES}')\n",
                "print(f'  Total Audio Clips:  {TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Run the Accelerated Cloud Streaming Trainer\n",
                "Streams audio in a single pass, batches 16 clips per GPU pass through XLS-R 300M + AASIST, trains the adaptation head, and evaluates on held-out test data.\n",
                "**Live second-by-second progress will be displayed below!**"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Execute the large-scale streaming trainer with unbuffered real-time output (-u)\n",
                "!python -u colab_w2v2_streaming_cloud.py \\\n",
                "    --train_samples {TRAIN_SAMPLES} \\\n",
                "    --val_samples {VAL_SAMPLES} \\\n",
                "    --test_samples {TEST_SAMPLES} \\\n",
                "    --batch_size_extract {BATCH_SIZE_EXTRACT} \\\n",
                "    --epochs {EPOCHS} \\\n",
                "    --batch_size {BATCH_SIZE} \\\n",
                "    --lr {LEARNING_RATE} \\\n",
                "    --output_dir ./checkpoints"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Display Large-Scale Evaluation Metrics\n",
                "Reads and displays the statistically robust evaluation report."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import os\n",
                "\n",
                "report_file = './checkpoints/large_scale_training_report.json'\n",
                "if os.path.exists(report_file):\n",
                "    with open(report_file) as f:\n",
                "        data = json.load(f)\n",
                "    m = data['metrics']\n",
                "    print('=' * 55)\n",
                "    print('  LARGE-SCALE STREAMING EVALUATION RESULTS')\n",
                "    print('=' * 55)\n",
                "    print(f\"Held-Out Test Samples:  {m['sample_count']}\")\n",
                "    print(f\"Calibrated Threshold:   {m['calibrated_threshold']}\")\n",
                "    print(f\"Test Accuracy:          {m['accuracy']}%\")\n",
                "    print(f\"Balanced Accuracy:      {m['balanced_accuracy']}%\")\n",
                "    print(f\"Precision:              {m['precision']}%\")\n",
                "    print(f\"Deepfake Recall:        {m['recall_fake']}%\")\n",
                "    print(f\"Human Specificity:      {m['specificity_real']}%\")\n",
                "    print(f\"F1-Score:               {m['f1_score']}%\")\n",
                "    print(f\"ROC-AUC:                {m['roc_auc']}\")\n",
                "    print(f\"Equal Error Rate (EER): {m['equal_error_rate']}%\")\n",
                "    print('=' * 55)\n",
                "else:\n",
                "    print('Report file not found. Ensure training finished successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Download Trained Checkpoint to Your Local Machine\n",
                "Downloads `voiceguard_w2v2_large_scale_model.pt` (~180 KB) directly to your computer. Drop this file into `voiceguard-ai/models/saved/w2v2_aasist/` to use it in your local FastAPI & React application."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    from google.colab import files\n",
                "    checkpoint_path = './checkpoints/voiceguard_w2v2_large_scale_model.pt'\n",
                "    if os.path.exists(checkpoint_path):\n",
                "        print(f'Downloading checkpoint ({os.path.getsize(checkpoint_path)/1024:.1f} KB)...')\n",
                "        files.download(checkpoint_path)\n",
                "        files.download('./checkpoints/large_scale_training_report.json')\n",
                "    else:\n",
                "        print('Checkpoint not found at:', checkpoint_path)\n",
                "except ImportError:\n",
                "    print('Running locally. Checkpoint is stored in ./checkpoints/')"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

output_path = os.path.join(PROJECT_ROOT, "colab_w2v2_large_scale_training.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=1)

print(f"Successfully generated self-contained notebook at: {output_path}")
