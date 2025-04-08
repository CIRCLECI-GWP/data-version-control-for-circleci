from datasets import load_dataset
from transformers import AutoTokenizer

class TextGenerationDataLoader:
    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        data_split: str = "train"
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.data_split = data_split
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
    
    def prepare_dataset(
        self,
        dataset_name: str,
    ) -> None:
        dataset = load_dataset(dataset_name, trust_remote_code=True)[self.data_split]
        text_column = self._get_text_column(dataset)

        def tokenize_func(examples):
            texts = [txt + self.tokenizer.eos_token for txt in examples[text_column]]
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            )
        tokenized_data = dataset.map(tokenize_func, batched=True)
        tokenized_data.set_format(
            type="torch",
            columns=['input_ids', 'attention_mask']
        )
        return tokenized_data

    def _get_text_column(self, dataset):
        """
        Determine the text column name based on dataset structure
        """
        possible_text_columns = ['text', 'content', 'sentence', 'document']
        
        for column in possible_text_columns:
            if column in dataset.column_names:
                return column
     
        # If none of the common names are found, use the first column
        return dataset.column_names[0]

    def get_dataset_stats(self, dataset):
        return {
            "total_samples": len(dataset),
            "inputs_ids_shape": dataset["input_ids"].shape,
            "attention_mask_shape": dataset["attention_mask"].shape
        }