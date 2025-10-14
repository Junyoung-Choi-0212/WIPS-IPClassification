from datasets import Dataset
from dotenv import load_dotenv
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
from trl import SFTConfig, SFTTrainer

import os
import pandas as pd
import torch
import torch.nn.functional as F 

load_dotenv()

hf_token = os.getenv('HF_TOKEN')

# 양자화 세팅
bnb_config_params = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': torch.float16, 'bnb_4bit_use_double_quant': True}
bnb_config = BitsAndBytesConfig(**bnb_config_params)

# 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-0.6B", device_map='auto', quantization_config=bnb_config, num_labels=5)

# 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", token=hf_token, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# LoRA 설정
lora_config_params = {
    'lora_alpha': 128,
    'lora_dropout': 0.1,
    'r': 64,
    'bias': 'none',
    'task_type': 'SEQ_CLS',
    'target_modules': ['v_proj', 'q_proj']
}

# PEFT 설정
peft_config = LoraConfig(**lora_config_params)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

torch.cuda.empty_cache()

df = pd.read_excel("../data/test.xlsx")
dataset = Dataset.from_pandas(df)

label_list = df["사용자태그"].unique().tolist()
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

def preprocess_function(examples):
    tokenized = tokenizer(examples['전체청구항'], truncation=True, max_length=512)
    tokenized['labels'] = [label2id[l] for l in examples['사용자태그']]
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True)
split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
tokenized_train = split["train"]
tokenized_test = split["test"]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    logits_tensor = torch.tensor(pred.predictions)
    labels_tensor = torch.tensor(pred.label_ids)
    loss = F.cross_entropy(logits_tensor, labels_tensor).item()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'eval_loss': loss
    }

default_training_args = {
    'output_dir': '../adapter_test',
    'learning_rate': 2e-5,
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'gradient_accumulation_steps': 4,
    'optim': 'paged_adamw_8bit',
    'lr_scheduler_type': 'cosine',
    'num_train_epochs': 5,
    'warmup_steps': 50,
    'logging_steps': 50,
    'fp16': True,
    'gradient_checkpointing': False,
    'dataset_text_field': '전체청구항',
    'max_length': 512,
    'label_smoothing_factor': 0.00,
    'label_names': ['labels']
}

training_arguments = SFTConfig(**default_training_args)
peft_config = LoraConfig(**lora_config_params)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    args=training_arguments,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    peft_config=peft_config
)
trainer.train()
trainer.evaluate()

trainer.model.save_pretrained("../adapter")