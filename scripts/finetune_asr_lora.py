"""
ASR LoRA Fine-Tuning Script for Whisper on Persian Conversational Speech.
Uses processor.tokenizer for correct Seq2SeqTrainer WER evaluation.
"""

import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model


def setup_whisper_lora(
    model_name: str = "openai/whisper-large-v3-turbo",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    freeze_encoder_ratio: float = 0.6,
):
    processor = WhisperProcessor.from_pretrained(model_name, language="fa", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Freeze bottom 60% of encoder layers
    num_encoder_layers = len(model.model.encoder.layers)
    freeze_until_layer = int(num_encoder_layers * freeze_encoder_ratio)
    for i in range(freeze_until_layer):
        for param in model.model.encoder.layers[i].parameters():
            param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def run_training(
    train_dataset,
    eval_dataset,
    output_dir: str = "checkpoints/whisper_fa_lora",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
):
    model, processor = setup_whisper_lora()

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        warmup_steps=100,
        num_train_epochs=epochs,
        eval_strategy="steps",
        fp16=True,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        eval_steps=500,
        logging_steps=50,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Correctly pass tokenizer (not feature_extractor) to enable string decoding for WER
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    print("ASR LoRA configuration and tokenizer verified.")
