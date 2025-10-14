import re
import requests
import streamlit as st
import time
import traceback

def settings():
    with st.expander("**LM STUDIO SETTING**"):
        col_api1, col_api2 = st.columns(2)

        with col_api1:
            api_url_input = st.text_input("BASE URL", value = st.session_state.get("api_url", "http://localhost:1234/v1/chat/completions"))
            st.session_state.api_url = api_url_input
            
        with col_api2:
            api_model_input = st.text_input("MODEL", value = st.session_state.get("api_model", "qwen/qwen3-14b"))
            st.session_state.api_model = api_model_input

        if st.button("API CONNECTION"):
            try:
                test_response = requests.post(
                    st.session_state.api_url,
                    json = {"model": st.session_state.api_model, "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10},
                    timeout = 30
                )

                if test_response.status_code != 200:
                    st.error(f"API CONNECTION FAILED ({test_response.status_code})")
                    st.session_state.api_connection_success = False
                    return
                
                response_data = test_response.json()
                actual_model = response_data.get("model", "")

                if actual_model != st.session_state.api_model:
                    st.error(
                        f"""API CONNECTION FAILED  
                        - REQUEST MODEL : {st.session_state.api_model}  
                        - LOAD MODEL : {actual_model}"""
                    )
                    st.session_state.api_connection_success = False
                    return

                st.success("API CONNECTION SUCCESSFUL")
                st.session_state.api_connection_success = True

            except Exception as e:
                st.error("API CONNECTION FAILED")
                st.error(e)
                st.session_state.api_connection_success = False

# STEP1 : 설정한 카테고리 별 점수를 모델이 예측
def get_score_for_candidate(text, code, desc, step1_prompt, api_url, api_model, max_retry=5, retry_delay=1):
    prompt = step1_prompt.format(text = text, code = code, desc = desc)
    
    for attempt in range(max_retry):
        try:
            response = requests.post(
                api_url,
                json = {
                    "model": api_model,
                    "messages": [
                        {"role": "system", "content": "당신은 특허를 주어진 카테고리로 분류하는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 8
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                score_text = result['choices'][0]['message']['content'].strip()

                # 숫자 추출
                match = re.search(r"\b([0-9](\.[0-9])?|10(\.0)?)\b", score_text)
                if match:
                    score = float(match.group())
                    # 범위 체크
                    if score < 0.0 or score > 10.0:
                        print(f"Score out of range, retrying: {score}")
                        score = None
                    else:
                        print(f"prompt engineering > code : {code}, score : {score}")
                        return score  # 정상 숫자면 바로 반환
                else:
                    print(f"숫자를 찾지 못함, 재시도 {attempt+1}")
            else:
                print(f"API 상태 코드 {response.status_code}, 재시도 {attempt+1}")

        except Exception as e:
            print(f"예외 발생: {e}, 재시도 {attempt+1}")

        time.sleep(retry_delay) # 재시도 전 딜레이

    # 최대 재시도 후에도 숫자 못 받으면 0.0
    print("최대 재시도 후에도 숫자 없음, 0.0 반환")
    return 0.0

# STEP2 : STEP1에서 모델이 예측한 점수가 같은 카테고리 후보군 중 모델이 최종 선택을 하게 프롬프트 전달
def reselect_best_code(text, candidate_codes, candidates, example, step2_prompt, api_url, api_model, max_retry=5, retry_delay=1):
    candidate_text = "\n".join([f"{code} : {candidates[code]}" for code in candidate_codes])
    prompt = step2_prompt.format(text = text, candidate_text = candidate_text, example = example)
    
    for attempt in range(max_retry):
        try:
            response = requests.post(
                api_url,
                json={
                    "model": api_model,
                    "messages": [
                        {"role": "system", "content": "당신은 특허를 주어진 카테고리로 분류하는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0,
                    "max_tokens": 15
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                final_choice = result['choices'][0]['message']['content'].strip()

                # 후보 코드 중 하나가 포함되어 있으면 바로 반환
                for code in candidate_codes:
                    if code in final_choice:
                        print(f"prompt engineering > reselect : {code}")
                        return code

                print(f"후보 코드 없음, 재시도 {attempt+1}")
            else:
                print(f"API 상태 코드 {response.status_code}, 재시도 {attempt+1}")

        except Exception as e:
            print(f"예외 발생: {e}, 재시도 {attempt+1}")

        time.sleep(retry_delay)

    print("최대 재시도 후에도 후보 코드 선택 실패") # 최대 재시도 후에도 후보 코드가 없으면 ERROR 반환
    return "ERROR"

def inference(selected_columns, df, custom_separator, step1_prompt, step2_prompt, candidates):
    if not st.session_state.get('api_connection_success', False):
        st.warning("Please complete API CONNECTION first.")
        return

    if st.button("**C L A S S I F Y**", type = "primary", width = 'stretch'):
        if not hasattr(st.session_state, 'api_url') or not hasattr(st.session_state, 'api_model'):
            st.error("Please set up the LM STUDIO API first.")
            return
        
        with st.spinner("CLASSIFICATION IN PROGRESS ..."):
            if len(selected_columns) == 1: # 분류에 사용할 컬럼이 한 개 라면
                data_to_classify = df[selected_columns[0]].dropna().astype(str).tolist()
            else: # 분류에 사용할 컬럼이 여러 개 라면
                clean_df = df[selected_columns].dropna()
                data_to_classify = clean_df.apply(lambda row: custom_separator.join([str(row[col]) for col in selected_columns]), axis = 1).tolist()

            data_to_classify = [text for text in data_to_classify if text.strip()]
            
            if not data_to_classify:
                st.error("No data available for classification.")
                return
            
            results = []

            progress_bar = st.progress(0)
            for i, text in enumerate(data_to_classify):
                try:
                    scores = {}

                    for code, desc in candidates.items(): # 카테고리 별 점수 계산
                        score = get_score_for_candidate(text, code, desc, step1_prompt, st.session_state.api_url, st.session_state.api_model)
                        scores[code] = score
                        time.sleep(0.5)

                    print(f"prompt engineering > score : {scores}")

                    if scores:
                        max_score = max(scores.values())
                        candidates_with_max_score = [code for code, s in scores.items() if s == max_score]

                        if len(candidates_with_max_score) == 1: # 가장 높은 점수인 분류가 하나만 있다면 해당 분류로 최종 결정
                            classification = candidates_with_max_score[0]
                        else: # 가장 높은 점수인 분류가 여러 개 있다면 다시 선택하는 프롬프트 실행
                            classification = reselect_best_code(text, candidates_with_max_score, candidates, next(iter(st.session_state.categories)), step2_prompt, st.session_state.api_url, st.session_state.api_model)
                            time.sleep(0.5)
                    else: # 분류 예측 점수 값을 받아오지 못했다면 에러 return
                        classification = "ERROR"

                    results.append({
                        'index': i,
                        'text': text,
                        'classification': classification,
                        'text_preview': text[:100] + "..." if len(text) > 100 else text
                    })
                except Exception:
                    print(f"Error at index {i} with text: {text[:50]}...")
                    traceback.print_exc()
                    results.append({
                        'index': i,
                        'text': text,
                        'classification': "ERROR",
                        'text_preview': text[:100] + "..." if len(text) > 100 else text
                    })
                
                progress_bar.progress((i + 1) / len(data_to_classify)) # 프로그레스 바 업데이트로 진행상황 UI에 노출
                time.sleep(1.0)
            
            st.session_state.classification_results = results
            st.toast("CLASSIFICATION COMPLETED")