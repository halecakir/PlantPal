import os
import warnings

import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
import json
import os
import pickle
import re
from random import randrange

import torch
import transformers
import uvicorn
from datasets import Dataset, load_dataset
from fastapi import FastAPI
from fuzzywuzzy import fuzz
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from pydantic import BaseModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from torch import bfloat16, cuda
from transformers import (
    AlbertTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TFAlbertModel,
    TrainingArguments,
)
from trl import SFTTrainer

# GLOBALS
ROOT_PATH = "data/plant-db"
KNOWLEDGE_PATH = "data/data_rqa"
BERT_MODEL = "albert-base-v2"
MAX_SEQ_LEN = 50
LABEL_ENCODER = "models/JointAlbert/checkpoints/intent_label_encoder.pkl"
SEQ_OUT_INDEX = "models/JointAlbert/checkpoints/seq_out_index_word.pkl"
CHECKPOINT_DIR = "models/JointAlbert/checkpoints/model/"
CLOSED_QA_CHECKPOINT = "models/LLaMa-FineTuned/checkpoint"

model_id = "meta-llama/Llama-2-7b-chat-hf"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

quantization_config = transformers.BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True
)

# begin initializing HF items, need auth token for these
hf_auth = "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"
model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)

llama_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=quantization_config,
    device_map={"": 3},
    use_auth_token=hf_auth,
)
llama_model.eval()
print(f"Model loaded on {device}")

tokenizer_llama_caretaking = transformers.AutoTokenizer.from_pretrained(
    model_id, use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
    model=llama_model,
    tokenizer=tokenizer_llama_caretaking,
    return_full_text=True,  # langchain expects the full text
    task="text-generation",
    # we pass model parameters here too
    temperature=0.75,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=300,  # mex number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
)
llm = HuggingFacePipeline(pipeline=generate_text)

model_name = "BAAI/bge-base-en"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity

model_norm = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs={"device": "cuda"}, encode_kwargs=encode_kwargs
)

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def get_prompt(instruction, system_prompt):
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


sys_prompt = """\
You are a helpful, plant caretaking assistant. Use the following pieces of information to answer the user's question. You can assume that the information is always about the plant in question even if the name is different.
If you don't know the answer, just say that you don't know, do under no circumstances try to make up an answer."""

instruction = """
CONTEXT:/n/n {context}/n

Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""


def get_llama_answer(question, file):
    loader = TextLoader(file)
    documents = loader.load()

    # persist_directory = 'db
    ## Here is the new embeddings being used
    embedding = model_norm
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding)

    # persist_directory=persist_directory)

    retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    prompt_template = get_prompt(instruction, sys_prompt)

    llama_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": llama_prompt}

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )

    return rag_pipeline(question)


class JointIntentAndSlotFillingModel(tf.keras.Model):
    def __init__(
        self,
        total_intent_no=None,
        total_slot_no=None,
        model_name=BERT_MODEL,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.bert = TFAlbertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(total_intent_no, activation="softmax")
        self.slot_classifier = Dense(total_slot_no, activation="softmax")

    def call(self, inputs, **kwargs):
        bert_output = self.bert(inputs)

        sequence_output = self.dropout(bert_output[0])
        slots_predicted = self.slot_classifier(sequence_output)

        pooled_output = self.dropout(bert_output[1])
        intent_predicted = self.intent_classifier(pooled_output)

        return slots_predicted, intent_predicted


with open(
    LABEL_ENCODER,
    "rb",
) as fp:
    intent_le = pickle.load(fp)
with open(
    SEQ_OUT_INDEX,
    "rb",
) as fp:
    index_to_word = pickle.load(fp)

tokenizer = AlbertTokenizer.from_pretrained(BERT_MODEL)

latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)

model = JointIntentAndSlotFillingModel(
    total_intent_no=3, total_slot_no=4, dropout_prob=0.1
)

model.load_weights(latest)


def detokenize(slots):
    data = slots

    result = {}
    current_key = ""
    current_value = None

    for key, value in data.items():
        if not key.startswith("▁") or key == "?":
            current_key += key
        else:
            if current_key:
                result[current_key[1:]] = current_value
            current_key = key
            current_value = value

    if current_key:
        result[current_key[1:]] = current_value

    return result


def show_predictions(text):
    tokenized_sent = tokenizer.encode(text)
    predicted_slots, predicted_intents = model.predict([tokenized_sent])
    intent = intent_le.inverse_transform([np.argmax(predicted_intents)])
    slots = np.argmax(predicted_slots, axis=-1)
    slots = [index_to_word[w_idx] for w_idx in slots[0]]
    slot_dict = {}
    for w, l in zip(tokenizer.tokenize(text), slots[1:-1]):
        slot_dict[w] = l
    slot_dict = detokenize(slot_dict)
    return intent, slot_dict


def get_plant_name(slots):
    plant_name = None
    for key, value in slots.items():
        if value == "PlantName":
            if plant_name == None:
                plant_name = key
            else:
                plant_name = plant_name + " " + key
    return plant_name


def get_latin2common():
    latin2common = {}

    total = 0
    missing_common_name = 0
    for file in os.listdir(ROOT_PATH):
        total += 1
        full_path = os.path.join(ROOT_PATH, file)
        all_names = []
        try:
            with open(full_path) as target:
                json_f = json.load(target)
                latin = json_f["pid"].lower()
                common_name = json_f["common_name"].lower()
                search_result = json_f["search_results"]
                search_result = [entry.lower() for entry in search_result]
                all_names = [latin, common_name] + search_result

                latin2common[latin] = all_names
        except Exception as e:
            missing_common_name += 1
            # print(f"Unknown error {e}")
    return latin2common


# GLOBALS
latin2common = get_latin2common()


def partial_match(plant_name, l2c):
    similiarity_score = 0
    best_match: None
    for key, value in l2c.items():
        for name in value:
            if plant_name in name:
                return key
            current_score = fuzz.token_sort_ratio(plant_name, name)
            if current_score >= similiarity_score:
                similiarity_score = current_score
                best_match = key
    if similiarity_score >= 90:
        return best_match
    else:
        return None


def search_plant_file(plant_name, l2c):
    file = None
    for key, value in l2c.items():
        if plant_name in value:
            file = key
            return file
    file = partial_match(plant_name, l2c)
    return file


def full_slots(slot_dict):
    full_slots_ = {}
    current_key = None
    current_value = None

    for key, value in slot_dict.items():
        if value.startswith("B-"):
            if current_key:
                full_slots_[current_key] = current_value
            current_key = key
            current_value = value[2:]  # Remove the 'B-' prefix
        elif value.startswith("I-"):
            if current_key:
                current_key += " " + key
                current_value = value[2:]  # Remove the 'I-' prefix
            else:
                full_slots_[key] = value[2:]
        else:
            if current_key:
                full_slots_[current_key] = current_value
                current_key = None
                current_value = None
            full_slots_[key] = value

    # Handle the last entry
    if current_key:
        full_slots_[current_key] = current_value

    return full_slots_


class CareTakingError(Exception):
    pass


def caretaking(question, slots):
    plant_name = get_plant_name(slots)
    if plant_name == None:
        raise CareTakingError(
            "Sorry, we could not recognize your plant name. Could you please provide either latin or common name?"
        )

    plant_file = search_plant_file(plant_name, latin2common)
    if plant_file == None:
        raise CareTakingError(
            "Unfortunately, given plant does not exist in our Knowledge Base."
        )
    file_path = f"{KNOWLEDGE_PATH}/{plant_file}.txt"

    return get_llama_answer(question, file_path)


class LMTrainingPipeline:
    def __init__(self, model_id, dataset_path, output_dir):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def install_dependencies(self):
        # pip install "transformers==4.31.0" "datasets==2.13.0" "peft==0.4.0" "accelerate==0.21.0" "bitsandbytes==0.40.2" "trl==0.4.7" "safetensors>=0.3.1" --upgrade
        pass

    def load_and_preprocess_data(self):
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            use_cache=False,
            device_map="auto",
        )
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

    def load_model(self):
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

    def generate_instruction(self, text):
        # Create a prompt for model generation
        prompt = f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{text}

### Response:
"""

        # Generate instructions using the model
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )

        # Display results
        return self.tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True
        )[0][len(prompt) :]


use_flash_attention = False

pipeline = LMTrainingPipeline(
    model_id="NousResearch/Llama-2-7b-hf",
    dataset_path="PlantQA.csv",
    output_dir=CLOSED_QA_CHECKPOINT,
)

# pipeline.install_dependencies()
# pipeline.load_and_preprocess_data()
pipeline.load_model_and_tokenizer()
pipeline.prepare_model_for_training()
pipeline.train_model()
pipeline.load_model()


def conversation(input_data):
    question = input_data.text
    prev_intent = input_data.prev_intent
    intent, slots = show_predictions(question)
    slots = full_slots(slots)
    intent = intent[0]
    if intent == "plant_caretaking_advice":
        try:
            res = caretaking(question, slots)
            if res:
                out = res["result"]
            else:
                out = "Sorry, I can't answer your question right now."
        except CareTakingError as e:
            out = str(e)
        return {
            "prediction": out,
            "prev_intent": "plant_caretaking_advice",
            "command": "DONE",
        }
    if intent == "plant_disease_advice":
        return {
            "prediction": pipeline.generate_instruction(question),
            "prev_intent": "plant_disease_advice",
            "command": "DONE",
        }
    if intent == "plant_purchasing_advice":
        out = "It looks like you are looking for a new plant to buy. Let me ask you some questions to make sure the plant will feel well in your home!\n\n"
        out += "Please estimate the average temperature in the room for your plant."
        return {
            "prediction": out,
            "prev_intent": "plant_purchasing_advice",
            "command": "CONT|TEMP",
            "info": {},
        }


def plant_advice_sent_gen(info):
    # Directory containing JSON files
    suggestions = []

    # Iterate over each JSON file in the directory
    for filename in os.listdir(ROOT_PATH):
        if len(suggestions) >= 5:
            break
        if filename.endswith(".json"):
            json_path = os.path.join(ROOT_PATH, filename)

            try:
                # Load the JSON data
                with open(json_path, "r") as json_file:
                    data = json.load(json_file)
                    suited = True

                # Extract the necessary information
                color_des = data["basic"]["color"]
                min_temp = data["parameter"]["min_temp"]
                max_temp = data["parameter"]["max_temp"]
                sunlight_des = data["maintenance"]["sunlight"]
                if info["color"] not in color_des:
                    suited = False
                elif info["temp"] < min_temp:
                    suited = False
                elif info["temp"] > max_temp:
                    suited = False
                if info["sunlight"]:
                    if "Like sunshine" not in sunlight_des:
                        suited = False
                else:
                    if "Like sunshine" in sunlight_des:
                        suited = False

                if suited == True:
                    suggestions.append(data["display_pid"])

            except Exception as e:
                print(f"Error processing {json_path}: {str(e)}")
    return (
        "The following plants seem like a good fit for you:"
        + ", ".join(suggestions)
        + " .Let me know if you want any further suggestions or advices."
    )


def plant_purchasing_advice(input_data):
    type = input_data.command.split("|")[1]
    info = input_data.info
    reply = input_data.text

    if type == "TEMP":
        temp = float(reply)
        info["temp"] = temp
        out = "Will your plant have direct sunlight"
        return {
            "prediction": out,
            "command": "CONT|LIGHT",
            "prev_intent": "plant_purchasing_advice",
            "info": info,
        }
    elif type == "LIGHT":
        info["sunlight"] = True if "yes" in reply.lower() else False
        out = "Which color do you want the flowers of your plant to have?"
        return {
            "prediction": out,
            "command": "CONT|COLOR",
            "info": info,
            "prev_intent": "plant_purchasing_advice",
        }
    elif type == "COLOR":
        color = reply
        color_names = [
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "purple",
            "pink",
            "brown",
            "gray",
            "black",
            "white",
        ]

        pattern = r"\b(?:" + "|".join(color_names) + r")\b"

        match = re.search(pattern, color, flags=re.IGNORECASE)

        # Check if a color name was found
        if match:
            color = match.group()
        else:
            color = None
        info["color"] = color
        out = plant_advice_sent_gen(info)
        return {
            "prediction": out,
            "command": "DONE",
            "prev_intent": "plant_purchasing_advice",
            "info": info,
        }
        pass
    else:
        # TODO
        pass
        return {
            "prediction": out,
            "command": "DONE",
            "prev_intent": "plant_purchasing_advice",
        }


# Define the input data model
class InputText(BaseModel):
    text: str
    command: str
    info: dict
    prev_intent: str


# Create a FastAPI app
app = FastAPI()


# Define the prediction endpoint
@app.post("/predict")
def generate_sentence(input_data: InputText):
    print(input_data)
    if input_data.command == "intent":
        out = conversation(input_data)
        return out
    elif input_data.command.startswith("CONT"):
        if input_data.prev_intent == "plant_purchasing_advice":
            return plant_purchasing_advice(input_data)
        else:
            return {
                "prediction": "Internal Error: Rule-based QA only supported in plant_purchasing_advice",
                "command": "DONE",
                "prev_intent": "",
                "info": {},
            }
    else:
        return {
            "prediction": "Internal Error: User interactions should be either intent-based or rule-based.",
            "command": "DONE",
            "prev_intent": "",
            "info": {},
        }


# Run the API server using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
