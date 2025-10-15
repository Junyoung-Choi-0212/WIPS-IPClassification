from datasets import Dataset
from dotenv import load_dotenv
from peft import PeftConfig, PeftModel
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding

import os
import pandas as pd
import torch

load_dotenv()

hf_token = os.getenv('HF_TOKEN')

peft_config = PeftConfig.from_pretrained("../adapter")

bnb_config_params = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': torch.float16, 'bnb_4bit_use_double_quant': True}
bnb_config = BitsAndBytesConfig(**bnb_config_params)

# 테스트 데이터
df = pd.read_excel("../data/cpc_c01_merge.xlsx")
dataset = Dataset.from_pandas(df)

label_list = df["사용자태그"].unique().tolist()
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# 베이스 모델 및 토크나이저 로드 
model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16, num_labels = len(label_list)) 
tokenizer = AutoTokenizer.from_pretrained( peft_config.base_model_name_or_path )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'
peft_model = PeftModel.from_pretrained(model, "../adapter", torch_dtype=torch.bfloat16) 
inference_model = peft_model

def preprocess_function(examples):
    tokenized = tokenizer(examples['전체청구항'], truncation=True, max_length=512)
    tokenized['labels'] = [label2id[l] for l in examples['사용자태그']]
    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True)
split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
tokenized_train = split["train"]
tokenized_test = split["test"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
columns_to_use = ["input_ids", "attention_mask", "labels"]
test_loader = DataLoader(
    tokenized_test.remove_columns([c for c in tokenized_test.column_names if c not in columns_to_use]),
    batch_size=1,
    collate_fn=DataCollatorWithPadding(tokenizer, padding="longest")
)

inference_model.eval()  # 평가 모드
device = next(inference_model.parameters()).device  # 모델이 있는 device

all_preds, all_labels, all_ids = [], [], []

for i, batch in enumerate(test_loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = inference_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
    
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(batch["labels"].numpy())
    all_ids.append(tokenized_test[i]["출원번호"])
        
print(classification_report(all_labels, all_preds, target_names=label_list))