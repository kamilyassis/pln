#!/usr/bin/env python
# coding: utf-8
# pip install transformers torch scikit-learn tqdm imbalanced-learn matplotlib

import os
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

# ──────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────
MODEL_NAME = "DeepWokLab/bert-tiny"
NUM_LABELS = 8
MAX_LEN = 512
BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42
# Frações: teste 15%, validação ~17.65% do restante → ~70% treino / ~15% val / ~15% test
TEST_SIZE = 0.15
VAL_SIZE_OF_REMAINING = 0.15 / (1.0 - TEST_SIZE)

_SAVE_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_SAVE_ROOT, "bert_tiny_classifier")
CHECKPOINTS_DIR = os.path.join(_SAVE_ROOT, "checkpoints")
DATASET_CSV = os.path.join(_SAVE_ROOT, "datasets", "combined_dataset.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)


def _loss_to_ckpt_slug(val_loss: float) -> str:
    """Nome de pasta seguro: substitui '.' por 'p' (ex.: 1p234567)."""
    s = f"{val_loss:.6f}"
    return s.replace(".", "p").replace("-", "neg")


# ──────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(preds, labels):
    preds_cls = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds_cls)
    report = classification_report(labels, preds_cls, zero_division=0)
    return acc, report


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_every = max(1, n_batches // 20)

    for step, batch in enumerate(tqdm(loader, desc="Training batches", mininterval=5.0)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

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
        if (step + 1) % log_every == 0 or (step + 1) == n_batches:
            print(
                f"  [train] batch {step + 1}/{n_batches}  loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / n_batches


def evaluate(model, loader, return_preds=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"  [eval] {len(loader)} batches…", flush=True)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            all_preds.append(outputs.logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    y_true = np.array(all_labels)
    y_pred = np.argmax(all_preds, axis=1)
    acc, report = compute_metrics(all_preds, y_true)
    avg_loss = total_loss / len(loader)
    print("  [eval] done.", flush=True)
    if return_preds:
        return avg_loss, acc, report, y_pred, y_true
    return avg_loss, acc, report


def predict(texts, model, tokenizer, id2label=None):
    model.eval()
    all_preds = []
    dataset = TextClassificationDataset(texts, [0] * len(texts), tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    if id2label:
        return [id2label[p] for p in all_preds]
    return all_preds


# ── Data ─────────────────────────────────────
id2label = {
    0: "No_Label",
    1: "Loaded_Language",
    2: "Name_Calling-Labeling",
    3: "Doubt",
    4: "Repetition",
    5: "Appeal_to_Fear-Prejudice",
    6: "Flag_Waving",
    7: "Exaggeration-Minimisation",
}
label2id = {v: k for k, v in id2label.items()}

df = pd.read_csv(DATASET_CSV)
df = df[df["label"].isin(id2label.values())].reset_index(drop=True)
texts = df["sentence"].tolist()
labels_str = df["label"].tolist()
labels_int = [label2id[l] for l in labels_str]

texts_array = np.asarray(texts, dtype=object).reshape(-1, 1)
ros = RandomOverSampler(random_state=SEED)
texts_resampled, labels_resampled = ros.fit_resample(texts_array, labels_int)
texts = [str(t).strip() for t in np.asarray(texts_resampled, dtype=object).ravel()]
labels = np.asarray(labels_resampled).tolist()

# Train / val / test (estratificado)
tv_texts, test_texts, tv_labels, test_labels = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    tv_texts,
    tv_labels,
    test_size=VAL_SIZE_OF_REMAINING,
    random_state=SEED,
    stratify=tv_labels,
)

print(
    f"Split: train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}",
    flush=True,
)
print(train_texts[:1], flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    use_safetensors=True,
).to(DEVICE)

_pin = DEVICE.type == "cuda"
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=_pin,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=_pin,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=_pin,
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
history_train_loss = []
history_val_loss = []
best_val_loss = float("inf")
best_ckpt_path = None

for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===", flush=True)
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc, val_report = evaluate(model, val_loader)

    history_train_loss.append(train_loss)
    history_val_loss.append(val_loss)

    print(
        f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}",
        flush=True,
    )
    print(val_report, flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        slug = _loss_to_ckpt_slug(val_loss)
        best_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"ck_best_val_loss_{slug}")
        os.makedirs(best_ckpt_path, exist_ok=True)
        model.save_pretrained(best_ckpt_path)
        tokenizer.save_pretrained(best_ckpt_path)
        print(f"  ✓ Checkpoint (melhor val loss): {best_ckpt_path}", flush=True)

# Gráfico de losses
epochs_x = np.arange(1, len(history_train_loss) + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs_x, history_train_loss, label="Treino", marker="o", markersize=3)
ax.plot(epochs_x, history_val_loss, label="Validação", marker="o", markersize=3)
ax.set_xlabel("Época")
ax.set_ylabel("Loss média")
ax.set_title("Curva de loss (treino vs validação)")
ax.legend()
ax.grid(True, alpha=0.3)
loss_plot_path = os.path.join(CHECKPOINTS_DIR, "loss_train_val.png")
fig.tight_layout()
fig.savefig(loss_plot_path, dpi=150)
plt.close(fig)
print(f"Gráfico de loss salvo em: {loss_plot_path}", flush=True)

# Matriz de confusão no conjunto de TESTE (melhor checkpoint)
best_model = None
if best_ckpt_path and os.path.isfile(os.path.join(best_ckpt_path, "config.json")):
    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_ckpt_path
    ).to(DEVICE)
    _, _, _, y_pred, y_true = evaluate(best_model, test_loader, return_preds=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    cm_csv = os.path.join(CHECKPOINTS_DIR, "confusion_matrix_test.csv")
    np.savetxt(cm_csv, cm, delimiter=",", fmt="%d")
    print(f"Matriz de confusão (CSV): {cm_csv}", flush=True)

    disp_labels = [id2label[i] for i in range(NUM_LABELS)]
    fig2, ax2 = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot(ax=ax2, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax2.set_title("Matriz de confusão — conjunto de teste")
    fig2.tight_layout()
    cm_png = os.path.join(CHECKPOINTS_DIR, "confusion_matrix_test.png")
    fig2.savefig(cm_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Matriz de confusão (PNG): {cm_png}", flush=True)

    print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(NUM_LABELS)], zero_division=0), flush=True)
else:
    print("Sem checkpoint para avaliar teste.", flush=True)

print(f"\nTraining complete. Melhor val loss: {best_val_loss:.6f}", flush=True)
print(f"Melhor checkpoint: {best_ckpt_path}", flush=True)

if best_model is not None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Espelho do melhor modelo em: {SAVE_DIR}", flush=True)
    sample = [
        "Você prefere o brasil ou cuba?",
        "Caralho porra que merda de politica do caralho",
    ]
    preds = predict(sample, best_model, tokenizer, id2label=id2label)
    for text, pred in zip(sample, preds):
        print(f"  [{pred}] {text}", flush=True)
