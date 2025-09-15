from components import sidebar
from dotenv import load_dotenv
from methods import finetuning, prompt_engineering
import streamlit as st
import os

load_dotenv()

st.set_page_config(page_title = "특허 문서 분류 자동화 플랫폼", layout = "wide")

st.title("특허 문서 분류 자동화 플랫폼")

st.markdown("---")

if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:1234/v1/chat/completions"
if "api_model" not in st.session_state:
    st.session_state.api_model = "qwen/qwen3-14b"
if 'classification_results' not in st.session_state:
    st.session_state.classification_results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_data' not in st.session_state:
    st.session_state.training_data = None
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'categories' not in st.session_state:
    st.session_state.categories = {
        "category" : "a description of certain category"
    }
if 'step1_prompt' not in st.session_state:
    st.session_state.step1_prompt = """다음은 특허 정보입니다.

{text}

아래의 분류 카테고리가 이 특허에 얼마나 적합한지 평가해 주세요.  

0.0은 전혀 관련 없음을 의미하고,  
5.0은 부분적으로 관련 있음을 의미하며,  
10.0은 완벽히 일치함을 의미합니다.  

분류 카테고리 : {code}
설명 : {desc}

다음 조건을 반드시 지켜야 합니다:
1. 0.0 ~ 10.0 사이 소수점 1자리 숫자 하나만 출력
2. 다른 텍스트, 태그, <think> 등 절대 출력 금지
3. 모델이 생각 과정이나 헛소리를 내보내면 안 됨
4. 출력 예시: 7.5
5. 숫자를 출력하지 못하면 모델은 반드시 다시 시도"""

if 'step2_prompt' not in st.session_state:
    st.session_state.step2_prompt = """다음은 특허 정보입니다.

{text}

아래의 분류 후보 중에서 위의 특허에 가장 적합한 **단 1개의 분류 카테고리**를 골라 주세요.

[분류 후보 및 설명]
{candidate_text}

다음 조건을 반드시 지켜야 합니다:
1. 분류 후보 중 하나만 출력
2. 다른 텍스트, 태그, <think> 등 절대 출력 금지
3. 모델이 생각 과정이나 헛소리를 내보내면 안 됨
4. 출력 예시: {example}
5. 후보를 출력하지 못하면 모델은 반드시 다시 시도"""
if 'default_save_dir' not in st.session_state:
    current_dir = os.getcwd()
    default_dir = os.path.join(current_dir, "models") 
    os.makedirs(default_dir, exist_ok=True) # 경로가 없을 경우 생성
    st.session_state.default_save_dir = default_dir

classification_method = sidebar.show()

if classification_method == "PROMPT ENGINEERING":
    prompt_engineering.show()

elif classification_method == "FINE TUNING":
    finetuning.show()

st.markdown("---")

st.markdown(
    """
    <div style = 'text-align : center; color : gray;'>
        특허 문서 분류 자동화 플랫폼 © IPickYou
    </div>
    """, 
    unsafe_allow_html = True
)