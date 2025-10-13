from io import BytesIO
from openpyxl.drawing.image import Image as XLImage

import pandas as pd
import streamlit as st
import time

def create_pie_chart(writer, fig):
    # 파이차트 이미지 생성
    img_bytes = fig.to_image(format = "png")
    
    # Excel 워크북에 이미지 추가
    workbook = writer.book
    worksheet = workbook['통계']
    img_stream = BytesIO(img_bytes)
    xl_img = XLImage(img_stream)
    
    # 파이 차트 삽입을 위해 D ~ E 컬럼 열 너비 키우기
    worksheet.column_dimensions["D"].width = 45
    worksheet.column_dimensions["E"].width = 45
    
    worksheet.add_image(xl_img, "D2") # D2 셀에 파이 차트 이미지 삽입

def create_download_btn(label, buffer, file_name):
    st.download_button(
        label = label,
        data = buffer.getvalue(),
        file_name = file_name,
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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

        if st.session_state.prompt_fig is not None:
            create_pie_chart(writer, st.session_state.prompt_fig)

    excel_buffer.seek(0)

    create_download_btn("✔️ DOWNLOAD PROMPT ENGINEERING RESULT (EXCEL)", excel_buffer, f"patent_classification_prompt_{int(time.time())}.xlsx")

def show_finetuning(results_df):
    if results_df is None or len(results_df) == 0:
        st.warning("다운로드할 결과가 없습니다.")
        return
        
    download_df = results_df[["patent_id", "text", "classification", "confidence"]].copy()
    download_df.columns = ['PATENT ID', 'TEXT', 'CLASSIFICATION', 'CONFIDENCE']
    
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine = 'openpyxl') as writer:
        download_df.to_excel(writer, sheet_name = '전체', index = False)

        if 'CLASSIFICATION' in download_df.columns:
            classification_groups = download_df.groupby('CLASSIFICATION')
            
            for category, group in classification_groups:
                safe_name = str(category).replace('/', '_').replace(':', '_')[:31]
                group.to_excel(writer, sheet_name = safe_name, index = False)
            
            stats_df = download_df['CLASSIFICATION'].value_counts().reset_index()
            stats_df.columns = ['예측분류', '개수']
            stats_df.to_excel(writer, sheet_name = '통계', index = False)
        
            if st.session_state.inference_fig is not None:
                create_pie_chart(writer, st.session_state.inference_fig)
        
        if 'CONFIDENCE' in download_df.columns:
            confidence_ranges = pd.cut(download_df['CONFIDENCE'], bins = [0, 0.5, 0.7, 0.85, 1.0], labels = ['LOW (0-0.5)', 'MEDIUM (0.5-0.7)', 'HIGH (0.7-0.85)', 'VERY HIGH (0.85-1.0)'])
            confidence_stats = confidence_ranges.value_counts().reset_index()
            confidence_stats.columns = ['신뢰도_구간', '개수']
            confidence_stats.to_excel(writer, sheet_name = '신뢰도_분석', index = False)

    excel_buffer.seek(0)

    create_download_btn("✔️ DOWNLOAD FINE TUNING RESULT (EXCEL)", excel_buffer, f"patent_classification_finetuning_{int(time.time())}.xlsx")