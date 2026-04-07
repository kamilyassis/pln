#!/usr/bin/env python3
# coding: utf-8
#
# Em background (mata instâncias antigas e grava PID):
#   cd <raiz_do_repo_pln>
#   ./processamento/run_bertiny_nohup.sh
# Se o log disser que falta torch e usou /usr/bin/python3:
#   export PYTHON="$HOME/miniconda3/bin/python"   # ajuste ao teu conda
#   ./processamento/run_bertiny_nohup.sh
# Parar: kill $(cat bertiny.nohup.pid)  ou  pkill -f processamento/bertiny.py
#
# Token Hugging Face (opcional): export HF_TOKEN=... antes de rodar.

# In[1]:


import os
import sys
import pandas as pd

# In[2]:


# pip install torch transformers scikit-learn pandas numpy tqdm accelerate imbalanced-learn
# Opcional: export HF_TOKEN=... para limites maiores no Hugging Face Hub

# In[3]:


try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from imblearn.over_sampling import RandomOverSampler
except ImportError as e:
    print(
        "Falta dependência para o treino. Ative seu conda/venv e instale:\n"
        "  pip install torch transformers scikit-learn pandas numpy tqdm accelerate imbalanced-learn\n"
        "Ou, na raiz do repo:\n"
        "  pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

# ──────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────
# google/… has model.safetensors (prajjwal1/bert-tiny is .bin only → torch>=2.6 required on recent transformers)
MODEL_NAME   = "DeepWokLab/bert-tiny"
NUM_LABELS   = 8          # ← change to your number of classes
MAX_LEN      = 512        # max token length
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 5e-5
WARMUP_RATIO = 0.1        # fraction of steps used for LR warm-up
WEIGHT_DECAY = 0.01
SEED         = 42
# Checkpoint sempre ao lado deste arquivo (não depende do cwd)
_SAVE_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_SAVE_ROOT, "bert_tiny_classifier")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

from pathlib import Path

#local_dataset_path = "data/conexao-labels-full-4.1nano-p2.csv"
#online_dataset_path = "https://media.githubusercontent.com/media/kamilyassis/pln/refs/heads/main/2026-03-29_label_datasets/data/out/conexao-labels-full-4.1nano-p2.csv"
local_dataset_path = "datasets/combined_dataset.csv"
online_dataset_path = "https://media.githubusercontent.com/media/kamilyassis/pln/refs/heads/main/2026-03-30_bert/data/out/semeval-treated.csv"

in_colab = False


def resolve_project_csv(relative_path: str) -> Path:
    """Localiza o CSV: raiz do repo (pai de processamento/), cwd e pai do cwd."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    cwd = Path.cwd().resolve()
    bases = []
    for b in (repo_root, cwd, cwd.parent, script_dir):
        b = b.resolve()
        if b not in bases:
            bases.append(b)
    for base in bases:
        candidate = (base / relative_path).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"CSV não encontrado: {relative_path!r}\n"
        f"  repo_root={repo_root}\n"
        f"  cwd={cwd}\n"
        "Coloque o ficheiro em datasets/ na raiz do projeto ou defina in_colab=True e URL."
    )


if in_colab:
    DATASET_ARGS = {"filepath_or_buffer": online_dataset_path, "sep": "|"}
else:
    DATASET_ARGS = {
        "filepath_or_buffer": str(resolve_project_csv(local_dataset_path)),
        "sep": ",",
    }


# In[4]:


# ──────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────
class TextClassificationDataset(Dataset):
    """
    Generic dataset for text classification.

    Args:
        texts  : list[str]  – raw input sentences
        labels : list[int]  – integer class indices (0 … NUM_LABELS-1)
        tokenizer           – HuggingFace tokenizer
        max_len : int       – max sequence length
    """

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# In[ ]:


# ──────────────────────────────────────────────
# 3. METRICS
# ──────────────────────────────────────────────
def compute_metrics(preds, labels):
    preds = np.argmax(preds, axis=1)
    acc   = accuracy_score(labels, preds)
    report = classification_report(labels, preds, zero_division=0)
    return acc, report


# ──────────────────────────────────────────────
# 4. TRAIN ONE EPOCH
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training batches"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ──────────────────────────────────────────────
# 5. EVALUATE
# ──────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            all_preds.append(outputs.logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    acc, report = compute_metrics(all_preds, all_labels)
    return total_loss / len(loader), acc, report


# ──────────────────────────────────────────────
# 6. INFERENCE / PREDICT
# ──────────────────────────────────────────────
def predict(texts, model, tokenizer, id2label=None):
    """
    Run inference on a list of strings.

    Returns:
        predicted class indices (and labels if id2label is provided)
    """
    model.eval()
    all_preds = []

    dataset = TextClassificationDataset(
        texts, [0] * len(texts), tokenizer  # dummy labels
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
            preds    = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    if id2label:
        return [id2label[p] for p in all_preds]
    return all_preds



# In[6]:


# ── 7a. Load your data ──────────────────────
# Replace this block with your actual data loading logic.
# `texts`  → list of raw strings
# `labels` → list of int class indices (0-based)

df = pd.read_csv(**DATASET_ARGS)
# Optionally define a human-readable mapping (used at inference time)
id2label = {
    0: "No_Label",
    1: "Loaded_Language",
    2: "Name_Calling-Labeling",
    3: "Doubt",
    4: "Repetition",
    5: "Appeal_to_Fear-Prejudice",
    6: "Flag_Waving",
    7: "Exaggeration-Minimisation"
}
df = df[df["label"].isin(id2label.values())].reset_index(drop=True)
label2id = {v: k for k, v in id2label.items()}

texts = df["sentence"].tolist()
labels = df["label"].tolist()
labels_int = [label2id[l] for l in labels]

texts_array = np.asarray(texts, dtype=object).reshape(-1, 1)

ros = RandomOverSampler(random_state=SEED)
texts_resampled, labels_resampled = ros.fit_resample(texts_array, labels_int)

texts = [str(t).strip() for t in np.asarray(texts_resampled, dtype=object).ravel()]
labels = np.asarray(labels_resampled).tolist()

# ── 7b. Train / val / test split ────────────
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)


# In[ ]:


print(train_texts[:5])


# In[ ]:


# ── 7c. Tokenizer & model ───────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
).to(DEVICE)


# In[ ]:


# ── 7d. DataLoaders ─────────────────────────
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
val_dataset   = TextClassificationDataset(val_texts,   val_labels,   tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)


# In[ ]:


# ── 7e. Optimizer & scheduler ───────────────
total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


# In[ ]:


# ── 7f. Training loop ───────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc, val_report = evaluate(model, val_loader)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )
    print(val_report)

    # Salva só o melhor checkpoint localmente (modelo + tokenizer + config)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ✓ Melhor modelo salvo em: {SAVE_DIR}")

print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


# In[ ]:


# ── 7g. Inference example ───────────────────
best_model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(DEVICE)
sample     = ["Você prefere o brasil ou cuba?", "Caralho porra que merda de politica do caralho"]
preds      = predict(sample, best_model, tokenizer, id2label=id2label)
for text, pred in zip(sample, preds):
    print(f"  [{pred}] {text}")

