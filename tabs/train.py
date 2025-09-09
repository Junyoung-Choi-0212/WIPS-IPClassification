# train.py

from utils.data_proceesor import DataProcessor
from utils.finetuning_trainer import FineTuningTrainer
from utils.text_chunker import SlidingWindowChunker
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict

# .env 파일 로드
load_dotenv()

def show():
    with st.expander("**COLUMNS TO USE FOR TRAIN**", expanded=True):
        df = st.session_state.uploaded_df

        available_cols = DataProcessor.get_available_columns(df)
        selected_cols = st.multiselect(
            "SELECTED COLUMNS",
            options=available_cols,
            default=[col for col in ["발명의 명칭", "요약", "전체청구항"] if col in available_cols],
            key="train_cols"
        )

        if not selected_cols:
            st.warning("Please select at least one column.")

    with st.expander("**HYPERPARAMETER**", expanded=False):
        try:
            DataProcessor.validate_dataframe(df, ["사용자태그"])
        except ValueError as e:
            st.error(e)
            return

        with st.expander("**QUANTIZATION**", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                load_in_4bit = st.checkbox("4 - BIT QUANTIZATION", value=True)
            with col2:
                bnb_4bit_quant_type = st.selectbox("QUANTIZATION TYPE", ["nf4", "fp4"], index=0)
            with col3:
                bnb_4bit_compute_dtype = st.selectbox("COMPUTE DTYPE", ["float16", "bfloat16", "float32"], index=0)
            with col4:
                bnb_4bit_use_double_quant = st.checkbox("DOUBLE QUANTIZATION", value=True)

        with st.expander("**LoRA**", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                lora_r = st.number_input("LoRA RANK (R)", min_value=4, max_value=256, value=64)
                lora_alpha = st.number_input("LoRA ALPHA", min_value=8, max_value=512, value=128)
            with col2:
                lora_dropout = st.number_input("LoRA DROPOUT", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
                bias_setting = st.selectbox("BIAS", ["none", "all", "lora_only"], index=0)
            with col3:
                task_type = st.selectbox("TASK TYPE", ["SEQ_CLS", "CAUSAL_LM", "SEQ_2_SEQ_LM"], index=0)
                target_modules = st.multiselect(
                    "TARGET MODULES",
                    options=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj'],
                    default=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
                )

        with st.expander("**SFT**", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.number_input("EPOCHS", min_value=1, max_value=10, value=5)
                learning_rate = st.number_input("LEARNING RATE", min_value=1e-7, max_value=1e-3, value=2e-5,
                                                format="%.0e")
            with col2:
                warmup_steps = st.number_input("WARMUP STEPS", min_value=0, max_value=100, value=50)
                test_size = st.number_input("TEST SIZE", min_value=0.1, max_value=0.3, value=0.2)
            with col3:
                max_length = st.number_input("MAX LENGTH", min_value=128, max_value=1024, value=512)
                stride = st.number_input("STRIDE", min_value=10, max_value=100, value=50)

            model_name_input = st.text_input("OUTPUT DIR", value="ft_gemma_2_2b")
            output_dir = os.path.join(r"C:\company\wips", model_name_input)

    if st.button("**T R A I N**", type="primary", use_container_width=True):
        try:
            model_name = st.session_state.get('ft_model_name', 'google/gemma-2-2b')
            hf_token = st.session_state.get('ft_hf_token') or os.getenv('HF_TOKEN')

            trainer = FineTuningTrainer(model_name, hf_token)

            with st.spinner("INITIALIZING MODEL ..."):
                trainer.initialize_tokenizer()

            with st.spinner("PREPROCESSING DATA ..."):
                processed_df = DataProcessor.prepare_data(trainer, df, selected_cols=selected_cols)
                
                chunker = SlidingWindowChunker(trainer.tokenizer)
                
                df_chunked = chunker.create_chunked_dataset(processed_df, max_length, stride)

                # 수정된 부분: 반환값 2개만 받음
                tokenized_dataset, test_df = DataProcessor.create_balanced_datasetdict(
                    df_chunked, trainer.tokenizer, test_size=test_size
                )

            with st.spinner("CONFIGURING MODEL ..."):
                bnb_config_params = {
                    'load_in_4bit': load_in_4bit,
                    'bnb_4bit_quant_type': bnb_4bit_quant_type,
                    'bnb_4bit_compute_dtype': bnb_4bit_compute_dtype,
                    'bnb_4bit_use_double_quant': bnb_4bit_use_double_quant
                }

                lora_config_params = {
                    'lora_alpha': lora_alpha,
                    'lora_dropout': lora_dropout,
                    'r': lora_r,
                    'bias': bias_setting,
                    'task_type': task_type,
                    'target_modules': target_modules
                }

                trainer.setup_model(bnb_config_params, lora_config_params)

            with st.spinner("TRAINING MODEL ..."):
                training_config_params = {
                    'num_train_epochs': epochs,
                    'learning_rate': learning_rate,
                    'warmup_steps': warmup_steps,
                    'max_length': max_length
                }

                # 수정된 부분: train_model 에 tokenized_dataset 전달
                eval_results = trainer.train_model(
                    tokenized_dataset, output_dir,
                    bnb_config_params, lora_config_params, training_config_params
                )

                trainer.save_model(output_dir)
            try:
                test_save_path = os.path.join(output_dir, "test_data.csv")
                test_df.to_csv(test_save_path, index=False, encoding="utf-8-sig")
                st.info(f"Test dataset saved to {test_save_path}")
            except Exception as e:
                st.warning(f"Could not save test dataset: {e}")

            st.toast("TRAIN IS COMPLETE")

            st.subheader("TRAIN RESULT")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("**ACCURACY**", f"{eval_results['eval_accuracy']:.4f}")
            with col2:
                st.metric("**F1 SCORE**", f"{eval_results['eval_f1']:.4f}")
            with col3:
                st.metric("**PRECISION**", f"{eval_results['eval_precision']:.4f}")
            with col4:
                st.metric("**RECALL**", f"{eval_results['eval_recall']:.4f}")

            st.session_state.model_info = {
                "model_path": output_dir,
                "labels_list": trainer.labels_list,
                "label2id": trainer.label2id,
                "id2label": trainer.id2label
            }

        except Exception as e:
            st.error(e)
            st.code(str(e))