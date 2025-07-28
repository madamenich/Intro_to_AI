import os
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
import torch
from sklearn.model_selection import train_test_split

MODEL_NAME = "distilgpt2"
DATA_FILE = "data/ai_qa_50.txt"
OUT_DIR = "./distilgpt2_ai_finetuned"

def load_custom_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        samples = [blk.strip() for blk in f.read().split("\n\n") if blk.strip()]
    return Dataset.from_dict({"text": samples})

def main():
    assert os.path.exists(DATA_FILE), f"Dataset file not found at {DATA_FILE}"

    print("Loading tokenizer & model...")
    tokenizer = GPT2LMHeadModel.from_pretrained  # just to fail early if transformers is wrong
    tokenizer = None

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.eos_token_id

    print("Loading dataset...")
    ds = load_custom_dataset(DATA_FILE)

    # Split to train/val (even if tiny) for early stopping & best model selection
    train_indices, val_indices = train_test_split(
        list(range(len(ds))), test_size=0.2, random_state=42, shuffle=True
    )
    train_ds = ds.select(train_indices)
    val_ds = ds.select(val_indices)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    print("Tokenizing...")
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=2,  # simulate larger batch
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving best model to {OUT_DIR} ...")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
