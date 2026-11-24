# SupportMind — Practical Fine-Tuning with Unsloth

SupportMind is an educational end-to-end fine-tuning lab for building a small, structured customer-support assistant. The project uses **Qwen3-4B**, **Unsloth**, **LoRA/QLoRA**, Hugging Face **Transformers** and **Datasets**, and **TRL's `SFTTrainer`**.

The goal is to understand and implement the main stages of an LLM fine-tuning workflow: dataset engineering, chat-template-aware tokenization, quantization, adapter configuration, supervised fine-tuning, evaluation, adapter export, and inference.

> **Important:** This is a learning project, not a production-ready customer-support system. Its dataset is small and synthetic, and its evaluation results should be treated as educational diagnostics rather than evidence of real-world performance.

## Project Goals

- Understand how a pretrained causal language model is loaded and prepared for fine-tuning.
- Compare model precision and quantization choices, including FP16, INT8, and 4-bit NF4.
- Apply LoRA adapters to a quantized base model using Unsloth.
- Prepare conversational training data in the Hugging Face message format.
- Understand chat templates, tokenization, and assistant-only loss.
- Fine-tune the model with TRL's `SFTTrainer`.
- Inspect training behavior and evaluate structured model outputs.
- Save and reload LoRA adapters, optionally export a merged 16-bit model, and run inference.
- Practice model-independent validation and unit testing.

## Technology Stack

| Component | Purpose |
|---|---|
| Python | Main implementation language |
| PyTorch | Tensor operations and model execution |
| Unsloth | Efficient model loading, LoRA setup, and fine-tuning support |
| Hugging Face Transformers | Model configuration, tokenizer, generation, and model interfaces |
| Hugging Face Datasets | Dataset representation and preparation |
| TRL | Supervised fine-tuning through `SFTTrainer` and `SFTConfig` |
| PEFT / LoRA | Parameter-efficient adapter fine-tuning |
| bitsandbytes | 4-bit and 8-bit quantization support |
| scikit-learn | Dataset splitting and evaluation metrics |
| Matplotlib | Training and evaluation visualizations |
| pandas | Human response-quality review worksheet |

## Base Model and Fine-Tuning Method

- **Base model:** `Qwen/Qwen3-4B`
- **Fine-tuning approach:** Supervised Fine-Tuning (SFT) with QLoRA
- **Base-model quantization:** 4-bit NF4 with double quantization
- **Compute precision:** FP16
- **Adapter method:** LoRA
- **Maximum sequence length:** 512 tokens
- **Training framework:** TRL `SFTTrainer`, with the model and LoRA integration provided by Unsloth

The notebook is configured for an FP16-capable CUDA environment. BF16 is disabled in the training configuration. A GPU with sufficient VRAM is required for the model-loading and training stages; the actual memory requirement depends on the runtime, library versions, and training configuration.

## SupportMind Output Contract

The assistant is trained to return a JSON object containing exactly three fields:

```json
{
  "intent": "return_request",
  "response": "Could you provide your order ID so I can guide you through the return process?",
  "needs_clarification": true
}
```

The supported intent labels are:

- `order_status`
- `shipping_issue`
- `refund_request`
- `return_request`
- `product_information`
- `technical_issue`
- `billing_issue`
- `complaint`
- `other`

The system prompt instructs the model to identify the customer's primary intent, respond professionally, ask for clarification when needed, and avoid inventing information.

## Dataset

The notebook builds a small synthetic customer-support dataset from:

- **27 seed examples**
- **18 augmentation examples**
- **45 total examples**
- **37 group identifiers**

Related paraphrases share a `group_id`. The notebook uses `GroupShuffleSplit` so examples from the same group are not distributed across training, validation, and test sets. This reduces leakage caused by near-duplicate examples appearing in different splits.

The exact split sizes are calculated and printed by the notebook. The dataset is intentionally small so the workflow can be studied without building a large data collection pipeline.

### Dataset limitations

- The examples are synthetic and do not represent real customer conversations.
- Some intent and clarification classes are imbalanced.
- A small test set cannot provide statistically reliable estimates of generalization.
- The model's responses are not connected to a real order system, CRM, payment gateway, or product database.

## Notebook Workflow

Open `SupportMind_Unsloth_Lab_Complete.ipynb` and follow the sections in order.

| Section | Topic | Main activities |
|---|---|---|
| 0 | Environment Setup | Install and inspect the required libraries; configure execution flags |
| 1 | Understanding Qwen3-4B | Inspect model configuration, tokenizer, and chat template |
| 2 | Baseline Evaluation | Optionally generate a response from the untouched base model |
| 3 | Dataset Engineering | Define the output contract, create examples, validate data, inspect balance and duplicates, and split by group |
| 4 | Tokenization & Chat Templates | Format conversations and inspect token lengths and assistant-token masking |
| 5 | Model Loading & Precision Analysis | Load the base model when enabled and inspect precision and memory considerations |
| 6 | Quantization Experiments | Explore FP16, INT8, and 4-bit quantization |
| 7 | LoRA Configuration | Configure adapter rank, alpha, dropout, target modules, and trainable parameters |
| 8 | SFT Dataset Preparation | Prepare Hugging Face datasets and inspect data quality |
| 9 | Supervised Fine-Tuning | Configure and initialize TRL's `SFTTrainer`, then train the model |
| 10 | Training Analysis | Inspect training logs and visualize training/evaluation loss |
| 11 | Model Evaluation | Evaluate JSON validity, intent prediction, clarification decisions, and response quality |
| 12 | Adapter Management & Export | Save adapter artifacts and metadata; optionally reload or merge the model |
| 13 | Final Inference | Generate a response for a new message and validate the JSON output |

## Setup

The notebook is designed to run in **Google Colab** or another compatible Python environment with a supported CUDA GPU.

1. Download or clone this project and open `SupportMind_Unsloth_Lab_Complete.ipynb`.
2. Select a CUDA-enabled runtime with enough GPU memory for Qwen3-4B fine-tuning.
3. Run the environment setup cells.
4. Review the installed versions of PyTorch, Transformers, TRL, and Unsloth.
5. Run the dataset and tokenizer sections before enabling model-weight downloads.

The notebook installs its main dependencies with:

```bash
pip install unsloth datasets scikit-learn matplotlib trl
```

The notebook also inspects the installed `SFTConfig` API before constructing the configuration. This is useful because argument names and supported options can vary between TRL versions. For reproducible experiments, record and pin the exact package versions used in the successful run.

## Execution Controls

Expensive operations are **disabled by default**. This allows the environment inspection, dataset engineering, tokenization, and model-independent tests to be explored without downloading the Qwen3-4B weights or starting GPU training.

In the environment/configuration cell, the main controls are:

```python
RUN_MODEL_DOWNLOADS = False
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_EXPORT = False
```

Enable the relevant operation only when you are ready:

| Flag | Set to `True` when you want to... |
|---|---|
| `RUN_MODEL_DOWNLOADS` | Download and load the quantized Qwen3-4B base model |
| `RUN_TRAINING` | Run supervised fine-tuning after the model and trainer are ready |
| `RUN_EVALUATION` | Generate predictions and calculate evaluation metrics |
| `RUN_EXPORT` | Save the adapter and optionally export the merged model |

The baseline model experiment and adapter reload smoke test have separate controls in their respective notebook sections. They are also disabled by default.

**Suggested order:** first run the lightweight sections and inspect the configuration. When a suitable GPU runtime is available, enable model loading, then training, evaluation, and export in that order. Run dependent cells again after changing the flags. Do not enable all operations blindly: model loading, training, evaluation, and export consume GPU memory, disk space, and runtime.

## Training Configuration

The notebook defines a LoRA configuration with the following main settings:

| Parameter | Value |
|---|---:|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| Random seed | 3407 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| RSLoRA | Disabled |

The SFT configuration uses:

| Parameter | Value |
|---|---:|
| Epochs | 3 |
| Per-device training batch size | 2 |
| Per-device evaluation batch size | 2 |
| Gradient accumulation steps | 4 |
| Learning rate | `1e-4` |
| Warmup steps | 1 |
| Scheduler | Cosine |
| Maximum sequence length | 512 |
| FP16 | Enabled |
| BF16 | Disabled |
| Packing | Disabled |
| Assistant-only loss | Enabled |

These are educational starting values for the notebook, not tuned production hyperparameters. The effective batch size and number of optimizer steps depend on the actual dataset split and training runtime.

### Assistant-only Loss

The notebook enables `assistant_only_loss=True`. This is intended to calculate training loss on assistant response tokens rather than on the system and user portions of each conversation.

This behavior depends on the chat template exposing assistant-generation regions. The notebook includes a tokenizer-level diagnostic and checks the training template after `SFTTrainer` initialization. The diagnostic probe alone does not prove how the trainer will behave; the trainer's effective template is the relevant one for training.

## Evaluation

The evaluation section includes:

- JSON validity rate
- Intent accuracy
- Per-intent precision, recall, and F1
- Clarification-decision accuracy
- Inspection of expected and generated outputs
- A human review worksheet for response relevance, factuality, clarity, and safety

The human review worksheet is written to:

```text
./outputs/supportmind/response_quality_review.csv
```

A reviewer must fill in the ratings. The notebook does not generate or assume subjective quality scores.

Because the dataset is small and synthetic, automated scores should not be interpreted as evidence that the model is ready for deployment.

## Exported Artifacts

When training and export are enabled, the notebook uses the following output locations:

```text
outputs/
└── supportmind/
    ├── final_adapter/
    │   ├── adapter configuration and weights
    │   ├── tokenizer files
    │   └── experiment_metadata.json
    ├── merged_16bit/
    │   └── optional merged model export
    └── response_quality_review.csv
```

The adapter directory stores the LoRA adapter separately from the base-model weights. The experiment metadata records the base model, quantization settings, LoRA/SFT configuration, dataset split information, dataset fingerprint, and library versions.

The merged 16-bit export is optional and can require considerably more memory and disk space than saving the adapter alone.

## Final Inference

The final inference interface accepts a customer message and returns a validated Python dictionary:

```python
prediction = predict_supportmind(
    user_message="I received my shoes, but they are damaged. I want to return them.",
    model=model,
    tokenizer=tokenizer
)

print(prediction)
```

The inference pipeline:

1. Validates the incoming message.
2. Builds the system/user conversation.
3. Applies the model's chat template.
4. Tokenizes the prompt and moves tensors to the model device.
5. Generates the response.
6. Removes the prompt tokens from the generated sequence.
7. Parses the result as strict JSON and validates the required fields and intent label.

The notebook also includes model-independent unit tests for the JSON parser. These tests can run without downloading the model.

## Current Scope and Future Improvements

This project focuses on learning the fine-tuning workflow. It does not implement a production customer-support backend.

Potential next steps include:

- Replace synthetic examples with reviewed, representative customer-support conversations.
- Expand the dataset and improve intent and clarification-class balance.
- Add a larger, held-out test set and repeated evaluation.
- Compare the base model and fine-tuned model using the same evaluation cases.
- Evaluate response quality with a documented human-review protocol.
- Add robust retry, timeout, logging, and monitoring behavior around inference.
- Connect the assistant to verified business data and authorized tools before allowing it to answer order, shipping, refund, or product-specific questions.
- Add deployment, API, privacy, and security controls before any real customer use.

## License and Model Terms

This repository is an educational project. Check the license and usage terms of the Qwen model, Unsloth, and all dependencies before redistributing model artifacts or using the system commercially. The notebook itself does not grant rights to the base model or its weights.
