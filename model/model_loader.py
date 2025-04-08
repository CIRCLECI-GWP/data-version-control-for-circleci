from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class LoraLanguageModel:
    def __init__(
        self,
        model_name: str,
        rank : int = 16,
        lora_alpha: int = 32
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params} || All params: {all_param}")
        print(f"Trainable percentage: {100 * trainable_params / all_param:.2f}%")
    