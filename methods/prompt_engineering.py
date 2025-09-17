from components.prompt_engineering import category_settings, prompt_settings
from utils import excel_download
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def show():
    if st.session_state.uploaded_df is not None:
        st.header("PROMPT ENGINEERING")

        df = st.session_state.uploaded_df

        with st.expander("**DATA PREPARATION**", expanded = True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**DATA PREVIEW**")
                st.dataframe(df.head(), use_container_width = True)
                st.metric("**TOTAL ROWS**", len(df))
                
            with col2:
                st.write("**COLUMNS TO INCLUDE IN PROMPT**")
                selected_columns = st.multiselect(
                    "SELECTED COLUMNS",
                    df.columns.tolist(),
                    help = "The contents of the selected columns are combined and sent to the prompt."
                )
                
                if len(selected_columns) > 1:
                    combine_method = st.selectbox(
                        "COLUMN MERGE METHOD",
                        ["SPACE", "LINE BREAKS", "CUSTOM DELIMITER"]
                    )
                    
                    if combine_method == "CUSTOM DELIMITER":
                        custom_separator = st.text_input("DELIMITER", value = " | ")
                    else:
                        custom_separator = " " if combine_method == "SPACE" else "\n"

                else:
                    custom_separator = ""
        
        category_settings.show()
        prompt_settings.show(selected_columns, df, custom_separator)

    else:
        st.info("⬅️ 특허 문서를 업로드해 주세요.")
    
    if st.session_state.classification_results:
        st.markdown("---")

        st.subheader("CLASSIFICATION RESULT")
        
        results = st.session_state.classification_results
        results_df = pd.DataFrame(results)

        col_overview1, col_overview2, col_overview3 = st.columns(3)

        with col_overview1:
            st.metric("**TOTAL CLASSIFICATIONS**", len(results))
        with col_overview2:
            unique_categories = results_df[results_df['classification'] != "ERROR"]['classification'].nunique()
            st.metric("**COUNT OF UNIQUE CATEGORIES**", unique_categories)
        with col_overview3:
            error_count = len([r for r in results if "ERROR" in r['classification']])
            st.metric("**CLASSIFICATION FAILED**", error_count)
        
        classification_groups = results_df.groupby('classification')
        
        for category, group in classification_groups:
            with st.expander(f"**{category} ({len(group)}건)**", expanded = True):
                display_df = group[['text_preview', 'classification']].copy()
                display_df.columns = ['TEXT PREVIEW', 'CLASSIFICATION']
                st.dataframe(display_df, use_container_width = True)
        
        st.subheader("PREDICTION DISTRIBUTION")
        classification_counts = results_df['classification'].value_counts()
        
        labels = classification_counts.index.tolist()
        sizes = classification_counts.values.tolist()

        # Plotly 파이차트
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=sizes,
            text=sizes,           # 각 조각 안에 count 표시
            textinfo='label+text', # 라벨과 count 표시
            hole=0.5,               # 도넛형 만들려면 0~1 사이 값 설정(1에 가까워질 수록 가운데 구멍 크기 증가)
            insidetextfont=dict(size=32) # 텍스트 크기 조절
        )])

        # 레이아웃: 차트 크기 조절 + 여백 최소화
        fig.update_layout(
            width=600,   # 차트 가로 크기
            height=600,  # 차트 세로 크기
            margin=dict(l=20, r=20, t=20, b=20)
        )

        # CSS로 정확히 중앙 정렬
        st.markdown(
            """
            <div style='display:flex; justify-content:center;'>
                <div style='width:500px'>
            """, 
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        excel_download.show_promptengineering(results_df, classification_groups)