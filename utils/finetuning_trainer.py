from datasets import DatasetDict
from dotenv import load_dotenv
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding
from trl import SFTConfig, SFTTrainer 
from utils.model_downscaler import create_downscaled_model

import gc
import os
import pickle
import tempfile
import torch
import torch.nn.functional as F

load_dotenv() # .env 파일 로드

class FineTuningTrainer:
    def __init__(self, model_name="google/gemma-2-2b", hf_token=None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.labels_list = None
        self.label2id = None
        self.id2label = None

    def initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

    def setup_model(self, bnb_config_params=None, lora_config_params=None, layer_usage=100):
        gc.collect()               # Python GC 실행
        torch.cuda.empty_cache()   # PyTorch GPU 캐시 비우기
            
        if hasattr(self, "model") and self.model is not None: # 이전에 만든 모델이 있을 경우 메모리 확보를 위해 삭제
            self.model.cpu()           # GPU → CPU로 이동
            del self.model             # 객체 삭제
        
        model = create_downscaled_model(self.hf_token, self.model_name, self.tokenizer, self.labels_list, layer_usage)
        
        with tempfile.TemporaryDirectory() as temp_dir: # 임시경로를 생성한 다음 temp_dir으로 접근(블록 종료 시 자동으로 삭제)
            model.save_pretrained(temp_dir)
            self.tokenizer.save_pretrained(temp_dir)
            
            if bnb_config_params is None:
                bnb_config_params = {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'float16', 'bnb_4bit_use_double_quant': True}

            bnb_config = BitsAndBytesConfig(**bnb_config_params)

            # 임시로 저장했던 모델을 불러오면서 양자화 진행
            self.model = AutoModelForSequenceClassification.from_pretrained(temp_dir, device_map='auto', quantization_config=bnb_config)

            torch.cuda.empty_cache() # GPU 캐시 정리

        if lora_config_params is None:
            lora_config_params = {
                'lora_alpha': 128,
                'lora_dropout': 0.1,
                'r': 64,
                'bias': 'none',
                'task_type': 'SEQ_CLS',
                'target_modules': ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            }

        peft_config = LoraConfig(**lora_config_params)
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)
        
        torch.cuda.empty_cache() # GPU 캐시 정리
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        torch.cuda.empty_cache() # GPU 캐시 정리
        
        print(f"[{self.model_name}] 최종 모델 config 레이어 수: {self.model.config.num_hidden_layers}") # 레이어 축소 적용 여부 확인

        return peft_config

    def compute_metrics(self, pred):
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

    def train_model(self, tokenized_dataset, output_dir, lora_config_params=None, training_config_params=None):
        """DatasetDict을 받도록 수정"""

        # DatasetDict인지 확인
        if isinstance(tokenized_dataset, DatasetDict):
            tokenized_train = tokenized_dataset['train']
            tokenized_test = tokenized_dataset['test']
        else:
            tokenized_train, tokenized_test = tokenized_dataset # 기존 방식 호환성

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        per_device_batch = 1
        gradient_acc_step = 4
        optim = 'paged_adamw_8bit'

        if torch.cuda.get_device_properties(0).total_memory >= 8e9:  # GPU 메모리가 8GB 이상의 여유있는 PC라면 성능↑
            per_device_batch = 2
            gradient_acc_step = 2
            optim = 'paged_adamw_32bit'

        default_training_args = {
            'output_dir': output_dir,
            'learning_rate': 2e-5,
            'per_device_train_batch_size': per_device_batch,
            'per_device_eval_batch_size': per_device_batch,
            'gradient_accumulation_steps': gradient_acc_step,
            'optim': optim,
            'lr_scheduler_type': 'cosine',
            'num_train_epochs': 5,
            'warmup_steps': 50,
            'logging_steps': 50,
            'fp16': True,
            'gradient_checkpointing': True,
            'dataset_text_field': 'text',
            'max_length': 512,
            'label_smoothing_factor': 0.05,
            'label_names': ['labels']
        }

        if training_config_params:
            default_training_args.update(training_config_params)

        training_arguments = SFTConfig(**default_training_args)

        if lora_config_params is None:
            lora_config_params = {
                'lora_alpha': 128,
                'lora_dropout': 0.1,
                'r': 64,
                'bias': 'none',
                'task_type': 'SEQ_CLS',
                'target_modules': ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            }

        peft_config = LoraConfig(**lora_config_params)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            processing_class=self.tokenizer,
            args=training_arguments,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            peft_config=peft_config
        )

        self.trainer.train()

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'label_mappings.pkl'), 'wb') as f:
            pickle.dump({'labels_list': self.labels_list, 'label2id': self.label2id, 'id2label': self.id2label}, f)

        return self.trainer.evaluate()

    def save_model(self, output_dir, merge_adapter=True):
        """어댑터 병합 옵션 추가"""

        if self.trainer:
            if merge_adapter:
                merged_model = self.trainer.model.merge_and_unload() # 어댑터 병합

                # 병합된 모델 저장
                merged_output_dir = os.path.join(output_dir, "merged_model")
                os.makedirs(merged_output_dir, exist_ok=True)
                merged_model.save_pretrained(merged_output_dir)
                self.tokenizer.save_pretrained(merged_output_dir)

                print(f"병합된 모델이 {merged_output_dir}에 저장되었습니다.")
            else:
                # 기존 방식 (어댑터만 저장)
                self.trainer.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)