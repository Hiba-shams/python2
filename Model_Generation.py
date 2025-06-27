from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import torch

MODEL_NAME = "google/flan-t5-small"  # Try flan-t5-base if needed

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Load JSONL dataset and split into train and validation (90/10)
dataset = load_dataset("json", data_files="car_rental_dataset.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Tokenization function
def tokenize(batch):
    model_inputs = tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["response"], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize both train and validation sets
tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=["prompt", "response"])
tokenized_val = val_dataset.map(tokenize, batched=True, remove_columns=["prompt", "response"])

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./car-rental-finetuned",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",  # Evaluate every epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,  # Increase to allow early stopping to work
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    report_to="none",
    gradient_accumulation_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define trainer with early stopping callback
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Stop if no improvement for 2 evals
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("./car-rental-finetuned")
tokenizer.save_pretrained("./car-rental-finetuned")

print("✅ Fine-tuning complete with evaluation and early stopping. Model saved to ./car-rental-finetuned")
