import torch
from random import randrange
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM


class LMTrainingPipeline:
    def __init__(self, model_id, dataset_path, output_dir):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def install_dependencies(self):
        # Install necessary dependencies
        #pip install "transformers==4.31.0" "datasets==2.13.0" "peft==0.4.0" "accelerate==0.21.0" "bitsandbytes==0.40.2" "trl==0.4.7" "safetensors>=0.3.1" --upgrade
        pass
    
    def load_and_preprocess_data(self):
        # Load and preprocess dataset
        dataset = load_dataset("csv", data_files=self.dataset_path)["train"]
        my_dict = dataset[:32]
        self.dataset = Dataset.from_dict(my_dict)

    def format_instruction(self, sample):
        return f"""### Instruction:
    Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{sample['title']}

### Response:
{sample['comment']}
"""

    def load_model_and_tokenizer(self):
        # Load tokenizer and model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb_config, use_cache=False, device_map="auto")
        self.model.config.pretraining_tp = 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def prepare_model_for_training(self):
        # Define LoRA configuration
        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)

    def train_model(self):
        # Define training arguments
        self.args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            fp16=True,
            tf32=False,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            disable_tqdm=True,
        )

        # Initialize the trainer
        max_seq_length = 2048
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=True,
            formatting_func=self.format_instruction,
            args=self.args,
        )

        # Train the model
        trainer.train()

        # Save the trained model
        trainer.save_model()

    def generate_instruction(self):
        # Optional: Unpatch flash attention if used
        if use_flash_attention:
            from utils.llama_patch import unplace_flash_attn_with_attn
            unplace_flash_attn_with_attn()

        # Load trained model and tokenizer
        self.args.output_dir = self.output_dir
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            self.args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.output_dir)

        # Load a sample from the dataset
        sample = self.dataset[randrange(len(self.dataset))]

        # Create a prompt for model generation
        prompt = f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{sample['title']}

### Response:
"""

        # Generate instructions using the model
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.9)

        # Display results
        print(f"Prompt:\n{sample['title']}\n")
        print(f"Generated instruction:\n{self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
        print(f"Ground truth:\n{sample['comment']}")

if __name__ == "__main__":
    # Set configuration flags
    use_flash_attention = False

    # Initialize and execute the training pipeline
    pipeline = LMTrainingPipeline(
        model_id="NousResearch/Llama-2-7b-hf",  # Update with your desired model
        dataset_path="PlantQA.csv",  # Update with your dataset path
        output_dir="llama-7-int4-plantqa",  # Update with your desired output directory
    )

    pipeline.install_dependencies()
    pipeline.load_and_preprocess_data()
    pipeline.load_model_and_tokenizer()
    pipeline.prepare_model_for_training()
    pipeline.train_model()
    pipeline.generate_instruction()
