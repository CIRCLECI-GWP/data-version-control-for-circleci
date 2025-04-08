import os
import argparse
from datetime import datetime

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from data.dataloader import TextGenerationDataLoader
from model.model_loader import LoraLanguageModel

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Parameters for Text Generation Model")

    parser.add_argument("--model_name", type=str, default="distilbert/distilgpt2", help="HuggingFace Model Name")
    parser.add_argument("--dataset_name", type=str, default="karpathy/tiny_shakespeare", help="HuggingFace Dataset Name")
    parser.add_argument("--max_length", type=int, default=512, help="Max Length of Training Example")
    parser.add_argument("--data_split", type=str, default="train", help="Data Split of the Mentioned Dataset")
    parser.add_argument("--run_name", type=str, default=None, help="MLFlow Training Run Name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size for LORA Training")
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    return parser.parse_args()


def train(args):
    short_model_name = args.model_name.split("/")[-1]
    short_dataset_name = args.dataset_name.split("/")[-1]
    if not args.run_name:
        args.run_name = f"{short_model_name}-{short_dataset_name}-{int(datetime.timestamp(datetime.utcnow()))}"
    output_dir = f"./results/{args.run_name}"
    
    dataloader = TextGenerationDataLoader(args.model_name, max_length=args.max_length, data_split=args.data_split)
    dataset = dataloader.prepare_dataset(dataset_name=args.dataset_name)
    dataset_stats = dataloader.get_dataset_stats(dataset)

    model = LoraLanguageModel(model_name=args.model_name, rank=args.lora_r, lora_alpha=args.lora_alpha)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        save_strategy="no",
        run_name=args.run_name
    )

    data_collator = DataCollatorForLanguageModeling(
            tokenizer=model.get_tokenizer(),
            mlm=False  # Not using masked language modeling
        )
        
    trainer = Trainer(
        model=model.get_model(),
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    train_result = trainer.train()
    adapter_path = os.path.join("./results", 'latest_lora_adapter')
    model.get_model().save_pretrained(adapter_path)
    
    return True


if __name__ == "__main__":
    args = parse_arguments()
    train(args)