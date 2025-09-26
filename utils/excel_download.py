# excel_download.py

from io import BytesIO
import pandas as pd
import streamlit as st
import time

import plotly.express as px
from openpyxl.drawing.image import Image as XLImage

def show_promptengineering(results_df, classification_groups):
    download_df = results_df[['text', 'classification']].copy()
    download_df.columns = ['TEXT', 'CLASSIFICATION']

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine = 'openpyxl') as writer:
        download_df.to_excel(writer, sheet_name = '전체', index = False)
 
        for category, group in classification_groups:
            safe_name = category.replace('/', '_').replace(':', '_')[:31]
            category_df = group[['text', 'classification']].copy()
            category_df.columns = ['TEXT', 'CLASSIFICATION']
            category_df.to_excel(writer, sheet_name = safe_name, index = False)

        stats_df = results_df['classification'].value_counts().reset_index()
        stats_df.columns = ['CLASSIFICATION', 'COUNT']
        stats_df.to_excel(writer, sheet_name = '통계', index = False)

    excel_buffer.seek(0)

    st.download_button(
        label = "✔️ DOWNLOAD PROMPT ENGINEERING RESULT (EXCEL)",
        data = excel_buffer.getvalue(),
        file_name = f"patent_classification_prompt_{int(time.time())}.xlsx",
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_finetuning(results_df):
    if results_df is None or len(results_df) == 0:
        st.warning("다운로드할 결과가 없습니다.")
        return
        
    download_df = results_df[["출원번호", "텍스트", "예측분류", "신뢰도"]].copy()
    download_df.columns = ['PATENT ID', 'TEXT', 'CLASSIFICATION', 'CONFIDENCE']
    
    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine = 'openpyxl') as writer:
        download_df.to_excel(writer, sheet_name = '전체', index = False)

        if 'CLASSIFICATION' in download_df.columns:
            classification_groups = download_df.groupby('예측분류')
            
            for category, group in classification_groups:
                safe_name = str(category).replace('/', '_').replace(':', '_')[:31]
                group.to_excel(writer, sheet_name = safe_name, index = False)
            
            stats_df = download_df['예측분류'].value_counts().reset_index()
            stats_df.columns = ['예측분류', '개수']
            stats_df.to_excel(writer, sheet_name = '통계', index = False)
        
            if st.session_state.inference_fig is not None:
                # 파이차트 생성
                fig = st.session_state.inference_fig
                img_bytes = fig.to_image(format = "png")
                
                # Excel 워크북에 이미지 추가
                workbook = writer.book
                worksheet = workbook['통계']
                img_stream = BytesIO(img_bytes)
                xl_img = XLImage(img_stream)
                
                # 파이 차트 삽입을 위해 E ~ F 컬럼 열 너비 키우기
                worksheet.column_dimensions["E"].width = 40
                worksheet.column_dimensions["F"].width = 40
                
                worksheet.add_image(xl_img, "E2") # E2 셀에 파이 차트 이미지 삽입
        
        if 'CONFIDENCE' in download_df.columns:
            confidence_ranges = pd.cut(download_df['신뢰도'], 
                                     bins = [0, 0.5, 0.7, 0.85, 1.0], 
                                     labels = ['LOW (0-0.5)', 'MEDIUM (0.5-0.7)', 'HIGH (0.7-0.85)', 'VERY HIGH (0.85-1.0)'])
            confidence_stats = confidence_ranges.value_counts().reset_index()
            confidence_stats.columns = ['신뢰도_구간', '개수']
            confidence_stats.to_excel(writer, sheet_name = '신뢰도_분석', index = False)

    excel_buffer.seek(0)

    st.download_button(
        label = "✔️ DOWNLOAD FINE TUNING RESULT (EXCEL)",
        data = excel_buffer.getvalue(),
        file_name = f"patent_classification_finetuning_{int(time.time())}.xlsx",
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )