from components.prompt_engineering import category_settings, prompt_settings
from utils import excel_download
from utils.result_pie_chart import PIE_CHART_COLORS, get_chart_figure

import pandas as pd
import streamlit as st

def show():
    if st.session_state.uploaded_df is not None:
        st.header("PROMPT ENGINEERING")

        df = st.session_state.uploaded_df

        with st.expander("**DATA PREPARATION**", expanded = True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**DATA PREVIEW**")
                df = df.reset_index(drop=True)
                df.index = df.index + 1    # index가 1부터 시작하도록 변경
                st.dataframe(df.head(), width = 'stretch')
                st.metric("**TOTAL ROWS**", len(df))
                
            with col2:
                st.write("**COLUMNS TO INCLUDE IN PROMPT**")
                selected_columns = st.multiselect(
                    "SELECTED COLUMNS",
                    df.columns.tolist(),
                    help = "The contents of the selected columns are combined and sent to the prompt."
                )
                
                # 프롬프트에 사용할 컬럼 선택
                if len(selected_columns) > 1:
                    combine_method = st.selectbox("COLUMN MERGE METHOD", ["SPACE", "LINE BREAKS", "CUSTOM DELIMITER"])
                    
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
        
        # 분류 결과 UI 표시
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
            with st.expander(f"**{category} ({len(group)}건)**", expanded = False):
                display_df = group[['text_preview', 'classification']].copy()
                display_df.columns = ['TEXT PREVIEW', 'CLASSIFICATION']
                display_df = display_df.reset_index(drop=True)
                display_df.index = display_df.index + 1    # index가 1부터 시작하도록 변경
                st.dataframe(display_df, width = 'stretch')
        
        st.subheader("PREDICTION DISTRIBUTION")
        
        # 분류 결과 파이 그래프로 표시
        st.session_state.prompt_fig = get_chart_figure(results_df['classification'])
        
        palette_name = st.selectbox("결과 그래프 색상 테마 선택", list(PIE_CHART_COLORS.keys()), index=0)
        st.session_state.prompt_fig.update_traces(marker=dict(colors=PIE_CHART_COLORS[palette_name]))
        
        # CSS로 정확히 중앙 정렬
        st.markdown(
            """
            <div style='display:flex; justify-content:center;'>
                <div style='width:500px'>
            """, 
            unsafe_allow_html=True
        )
        st.plotly_chart(st.session_state.prompt_fig, width='stretch')
        st.markdown("</div></div>", unsafe_allow_html=True)

        excel_download.show_promptengineering(results_df, classification_groups)