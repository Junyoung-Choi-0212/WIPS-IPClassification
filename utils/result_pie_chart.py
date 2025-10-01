import plotly.graph_objects as go
import plotly.express as px

COLORS = {
    "Set3": px.colors.qualitative.Set3,     # 12색, 파스텔톤
    "Bold": px.colors.qualitative.Bold,     # 진한 색
    "Pastel": px.colors.qualitative.Pastel, # 은은한 색
    "Dark24": px.colors.qualitative.Dark24, # 24색, 진한 색
}

def get_chart_figure(classification):
    value_counts = classification.value_counts()
    labels = value_counts.index.tolist()
    sizes = value_counts.values.tolist()

    # Plotly 파이차트
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        text=sizes,                                                                         # 각 조각 안에 count 표시
        textinfo='label+text',                                                              # 라벨과 count 표시
        hole=0.5,                                                                           # 도넛형 만들려면 0~1 사이 값 설정(1에 가까워질 수록 가운데 구멍 크기 증가)
        insidetextfont=dict(size=32),                                                       # 텍스트 크기 조절
        textposition='inside',                                                              # 조각 안쪽에 표시
        insidetextorientation='horizontal',                                                 # 텍스트 고정, 회전 안 됨
        pull=[0.1 if i == sizes.index(max(sizes)) else 0 for i in range(len(sizes))],       # 가장 분류 카운트가 높은 조각을 밖으로 분리
        hovertemplate='%{label}<br>Count: %{value}<br>Percent: %{percent}<extra></extra>',  # 마우스 호버링 시 노출되는 텍스트 템플릿
        marker=dict(colors=COLORS["Set3"])                                                  # 색상 팔레트 적용
    )])

    fig.update_layout(
        width=600,   # 차트 가로 크기
        height=600,  # 차트 세로 크기
        margin=dict(l=20, r=20, t=20, b=20), # 차트 여백 최소화
        legend=dict(font=dict(size=32), itemsizing='constant') # 범례 설정(텍스트 및 item 크기 설정)
    )

    fig.update_traces(marker=dict(line=dict(width=2))) # 범례 마커(박스) 크기 조절(테두리 두껍게)
    
    return fig