from utils import lmstudio
import streamlit as st

def show(selected_columns, df, custom_separator):

    with st.expander("**PROMPT TEMPLATE**", expanded = False):
    
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

        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.step1_prompt = st.text_area(
                "[STEP 1] RELEVANCE SCORING",
                value = st.session_state.step1_prompt,
                height = 300,
                help = "각 카테고리마다 적합도 점수를 계산합니다."
            )
        
        with col2:
            st.session_state.step2_prompt = st.text_area(
                "[STEP 2] BEST MATCH SELECTION", 
                value = st.session_state.step2_prompt,
                height = 300,
                help = "동점인 경우, 최적의 코드를 재선택합니다."
            )
        
        if st.button("INITIALIZE PROMPT"):
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
            st.rerun()

    lmstudio.settings()

    if selected_columns and len(st.session_state.categories) > 0:
        lmstudio.inference(selected_columns, df, custom_separator, 
                          st.session_state.step1_prompt, st.session_state.step2_prompt, 
                          st.session_state.categories)
    else:
        st.warning("Please select at least one column and one category.")