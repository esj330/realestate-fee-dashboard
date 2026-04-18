import io
import os
import zipfile
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

import plotly.express as px

import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


st.set_page_config(
    page_title='부동산 실거래가액 · 수수료 분석 Dashboard',
    page_icon='📊',
    layout='wide',
)


APP_TITLE = '부동산 실거래가액 · 수수료 분석 Dashboard'
APP_SUBTITLE = '조건별 분석, 인터랙티브 차트, 열지도 세부 설정, PDF/CSV/GPKG 다운로드 지원'

SOURCE_RENAME_MAP = {
    'addr': '행정동주소',
    'dangi_nm': '단지명',
    'trade': '거래유형',
    'm2': '면적',
    'yyyymm': '거래연월',
    'price10k': '거래가액(만원)',
    'rent': '임대료',
    'flr': '층',
    'compl_yr': '준공연도',
    'build_ty': '건물유형',
    'lat': 'lat',
    'lon': 'lon',
}

REQUIRED_STANDARD_COLUMNS = [
    '행정동주소', '단지명', '거래유형', '면적', '거래연월', '거래가액(만원)',
    '임대료', '층', '준공연도', '건물유형', 'lat', 'lon'
]

HEATMAP_GRADIENTS = {
    '터보': {
        0.10: '#30123b',
        0.25: '#4145ab',
        0.40: '#4675ed',
        0.55: '#1bcfd4',
        0.70: '#61fc6c',
        0.85: '#f3c63a',
        1.00: '#7a0403',
    },
    '기본 빨강': {
        0.20: '#ffe5e5',
        0.40: '#ffb3b3',
        0.60: '#ff6666',
        0.80: '#ff1f1f',
        1.00: '#990000',
    },
    '강한 빨강': {
        0.15: '#fff0f0',
        0.35: '#ff9999',
        0.55: '#ff4d4d',
        0.75: '#e60000',
        1.00: '#660000',
    },
    '부드러운 빨강': {
        0.20: '#fff5f5',
        0.45: '#ffd6d6',
        0.65: '#ff9e9e',
        0.85: '#ff5c5c',
        1.00: '#b30000',
    },
}

RED_SCALE = ['#ffe5e5', '#ffb3b3', '#ff6666', '#ff1f1f', '#990000']
PLOTLY_RED_SCALE = [
    [0.0, '#ffe5e5'],
    [0.25, '#ffb3b3'],
    [0.50, '#ff6666'],
    [0.75, '#ff1f1f'],
    [1.0, '#990000'],
]

PLOTLY_TURBO_SCALE = 'Turbo'


# -----------------------------
# 공통 유틸
# -----------------------------
def register_korean_font() -> str:
    candidates = [
        './NanumGothic.ttf',
        './malgun.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/malgun.ttf',
        'C:/Windows/Fonts/gulim.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font_name = 'AppKoreanFont'
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(font_name, path))
                try:
                    plt.rcParams['font.family'] = fm.FontProperties(fname=path).get_name()
                except Exception:
                    pass
                plt.rcParams['axes.unicode_minus'] = False
                return font_name
            except Exception:
                continue

    plt.rcParams['axes.unicode_minus'] = False
    return 'Helvetica'


PDF_FONT_NAME = register_korean_font()


def safe_read_csv(uploaded_file) -> pd.DataFrame:
    encodings = ['utf-8-sig', 'cp949', 'euc-kr', 'utf-8']
    last_error = None
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception as e:
            last_error = e
    raise ValueError(f'CSV를 읽지 못했습니다. 인코딩을 확인하세요. 마지막 오류: {last_error}')



def to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('원', '', regex=False)
        .str.replace('만원', '', regex=False)
        .str.replace(' ', '', regex=False)
        .replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
    )
    return pd.to_numeric(cleaned, errors='coerce')



def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    lower_cols = {str(c).strip().lower(): c for c in df.columns}

    for src, dst in SOURCE_RENAME_MAP.items():
        if src in df.columns:
            rename_map[src] = dst
        elif src.lower() in lower_cols:
            rename_map[lower_cols[src.lower()]] = dst

    alias_map = {
        '행정동주소': ['행정동주소', '주소', 'addr'],
        '단지명': ['단지명', 'dangi_nm', '아파트명'],
        '거래유형': ['거래유형', 'trade', '거래내용'],
        '면적': ['면적', 'm2', '전용면적'],
        '거래연월': ['거래연월', 'yyyymm'],
        '거래가액(만원)': ['거래가액(만원)', 'price10k', '보증금', '매매가', '전세가'],
        '임대료': ['임대료', 'rent', '월세'],
        '층': ['층', 'flr'],
        '준공연도': ['준공연도', 'compl_yr'],
        '건물유형': ['건물유형', 'build_ty'],
        'lat': ['lat', '위도', 'latitude'],
        'lon': ['lon', '경도', 'longitude'],
    }

    existing_targets = set(rename_map.values())
    for dst, aliases in alias_map.items():
        if dst in existing_targets or dst in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = dst
                break
            lowered = alias.lower()
            if lowered in lower_cols:
                rename_map[lower_cols[lowered]] = dst
                break

    out = df.rename(columns=rename_map).copy()

    for col in ['면적', '거래가액(만원)', '임대료', '층', '준공연도', 'lat', 'lon']:
        if col in out.columns:
            out[col] = to_numeric_series(out[col])

    if '거래연월' in out.columns:
        ym = out['거래연월'].astype(str).str.extract(r'(\d{6})', expand=False)
        out['거래연월'] = ym
        out['거래연월_일자'] = pd.to_datetime(out['거래연월'] + '01', format='%Y%m%d', errors='coerce')
    else:
        out['거래연월_일자'] = pd.NaT

    if '거래유형' in out.columns:
        out['거래유형'] = out['거래유형'].astype(str).str.strip()
    if '건물유형' in out.columns:
        out['건물유형'] = out['건물유형'].astype(str).str.strip()

    missing = [c for c in REQUIRED_STANDARD_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f'필수 컬럼이 없습니다: {missing}')

    return out



def calculate_trade_amount(trade_type: str, price10k: float, rent: float,
                           sale_label: str, jeonse_label: str, monthly_label: str) -> float:
    price10k = 0.0 if pd.isna(price10k) else float(price10k)
    rent = 0.0 if pd.isna(rent) else float(rent)

    if trade_type in [sale_label, jeonse_label]:
        return price10k * 10000.0
    if trade_type == monthly_label:
        return ((rent * 100.0) + price10k) * 10000.0
    return np.nan



def get_fee_rule(trade_type: str, amount: float,
                 sale_label: str, jeonse_label: str, monthly_label: str) -> Tuple[str, float, Optional[float]]:
    sale_rules = [
        (0, 50_000_000, '5000만원 미만', 0.006, 250_000),
        (50_000_000, 200_000_000, '5천만원 이상 ~ 2억원 미만', 0.005, 800_000),
        (200_000_000, 900_000_000, '2억원 이상 ~ 9억원 미만', 0.004, None),
        (900_000_000, 1_200_000_000, '9억원 이상 ~ 12억원 미만', 0.005, None),
        (1_200_000_000, 1_500_000_000, '12억원 이상 ~ 15억원 미만', 0.006, None),
        (1_500_000_000, None, '15억원 이상', 0.007, None),
    ]
    lease_rules = [
        (0, 50_000_000, '5천만원 미만', 0.005, 200_000),
        (50_000_000, 100_000_000, '5천만원 이상 ~ 1억원 미만', 0.004, 300_000),
        (100_000_000, 600_000_000, '1억원 이상 ~ 6억원 미만', 0.003, None),
        (600_000_000, 1_200_000_000, '6억원 이상 ~ 12억원 미만', 0.004, None),
        (1_200_000_000, 1_500_000_000, '12억원 이상 ~ 15억원 미만', 0.005, None),
        (1_500_000_000, None, '15억원 이상', 0.006, None),
    ]

    if pd.isna(amount):
        return '', 0.0, None

    if trade_type == sale_label:
        rules = sale_rules
    elif trade_type in [jeonse_label, monthly_label]:
        rules = lease_rules
    else:
        return '', 0.0, None

    for min_amt, max_amt, label, rate, cap in rules:
        if max_amt is None:
            if amount >= min_amt:
                return label, rate, cap
        else:
            if min_amt <= amount < max_amt:
                return label, rate, cap
    return '', 0.0, None



def add_derived_columns(df: pd.DataFrame,
                        sale_label: str = '매매',
                        jeonse_label: str = '전세',
                        monthly_label: str = '월세') -> pd.DataFrame:
    out = df.copy()

    out['거래가액'] = out.apply(
        lambda r: calculate_trade_amount(
            str(r['거래유형']).strip(), r['거래가액(만원)'], r['임대료'],
            sale_label, jeonse_label, monthly_label
        ),
        axis=1,
    )

    fee_results = out.apply(
        lambda r: get_fee_rule(
            str(r['거래유형']).strip(), r['거래가액'],
            sale_label, jeonse_label, monthly_label
        ),
        axis=1,
        result_type='expand',
    )
    fee_results.columns = ['거래금액구간', '상한요율', '한도액']
    out = pd.concat([out, fee_results], axis=1)

    out['수수료'] = out['거래가액'] * out['상한요율']
    cap_mask = out['한도액'].notna() & (out['수수료'] > out['한도액'])
    out.loc[cap_mask, '수수료'] = out.loc[cap_mask, '한도액']

    out['거래가액(억원)'] = out['거래가액'] / 100_000_000
    out['수수료(만원)'] = out['수수료'] / 10_000

    return out



def format_won(x: float) -> str:
    if pd.isna(x):
        return '-'
    return f'{x:,.0f}원'



def format_eok(x: float) -> str:
    if pd.isna(x):
        return '-'
    return f'{x:,.2f}억원'



def format_won_to_eok(x: float) -> str:
    if pd.isna(x):
        return '-'
    return f'{(x / 100_000_000):,.2f}억원'



def apply_all_filter(df: pd.DataFrame, column: str, selections: List[str]) -> pd.DataFrame:
    if selections and 'all' not in selections:
        return df[df[column].isin(selections)]
    return df



def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header('분석 조건')

    trade_options = sorted([x for x in df['거래유형'].dropna().unique().tolist() if str(x) != 'nan'])
    build_options = sorted([x for x in df['건물유형'].dropna().unique().tolist() if str(x) != 'nan'])

    trade_selector = ['all'] + trade_options
    build_selector = ['all'] + build_options

    selected_trade = st.sidebar.multiselect('거래유형', trade_selector, default=['all'])
    selected_build = st.sidebar.multiselect('건물유형', build_selector, default=['all'])

    ym_values = sorted(df['거래연월'].dropna().astype(str).unique().tolist())
    if ym_values:
        ym_start, ym_end = st.sidebar.select_slider('거래연월 범위', options=ym_values, value=(ym_values[0], ym_values[-1]))
    else:
        ym_start, ym_end = None, None

    area_min = float(np.nanmin(df['면적'])) if df['면적'].notna().any() else 0.0
    area_max = float(np.nanmax(df['면적'])) if df['면적'].notna().any() else 0.0
    selected_area = st.sidebar.slider('면적 범위(m²)', min_value=float(area_min), max_value=float(area_max), value=(float(area_min), float(area_max)))

    year_series = df['준공연도'].dropna()
    if len(year_series) > 0:
        year_min = int(year_series.min())
        year_max = int(year_series.max())
        selected_year = st.sidebar.slider('준공연도 범위', min_value=year_min, max_value=year_max, value=(year_min, year_max))
    else:
        selected_year = (0, 9999)

    floor_series = df['층'].dropna()
    if len(floor_series) > 0:
        floor_min = int(floor_series.min())
        floor_max = int(floor_series.max())
        selected_floor = st.sidebar.slider('층 범위', min_value=floor_min, max_value=floor_max, value=(floor_min, floor_max))
    else:
        selected_floor = (-999, 999)

    amount_min = float(np.nanmin(df['거래가액'])) if df['거래가액'].notna().any() else 0.0
    amount_max = float(np.nanmax(df['거래가액'])) if df['거래가액'].notna().any() else 0.0
    selected_amount = st.sidebar.slider(
        '거래가액 범위(원)',
        min_value=float(amount_min),
        max_value=float(amount_max),
        value=(float(amount_min), float(amount_max)),
        format='%.0f'
    )

    keyword_addr = st.sidebar.text_input('주소 키워드')
    keyword_dangi = st.sidebar.text_input('단지명 키워드')

    filtered = df.copy()
    filtered = apply_all_filter(filtered, '거래유형', selected_trade)
    filtered = apply_all_filter(filtered, '건물유형', selected_build)

    filtered = filtered[filtered['면적'].between(selected_area[0], selected_area[1], inclusive='both')]
    filtered = filtered[filtered['준공연도'].between(selected_year[0], selected_year[1], inclusive='both') | filtered['준공연도'].isna()]
    filtered = filtered[filtered['층'].between(selected_floor[0], selected_floor[1], inclusive='both') | filtered['층'].isna()]
    filtered = filtered[filtered['거래가액'].between(selected_amount[0], selected_amount[1], inclusive='both') | filtered['거래가액'].isna()]

    if ym_start and ym_end:
        filtered = filtered[(filtered['거래연월'] >= ym_start) & (filtered['거래연월'] <= ym_end)]
    if keyword_addr:
        filtered = filtered[filtered['행정동주소'].astype(str).str.contains(keyword_addr, case=False, na=False)]
    if keyword_dangi:
        filtered = filtered[filtered['단지명'].astype(str).str.contains(keyword_dangi, case=False, na=False)]

    return filtered



def compute_summary(df: pd.DataFrame) -> Dict[str, float]:
    return {
        '건수': int(len(df)),
        '총거래가액': float(df['거래가액'].sum(skipna=True)),
        '평균거래가액': float(df['거래가액'].mean(skipna=True)) if len(df) > 0 else np.nan,
        '최소거래가액': float(df['거래가액'].min(skipna=True)) if len(df) > 0 else np.nan,
        '최대거래가액': float(df['거래가액'].max(skipna=True)) if len(df) > 0 else np.nan,
        '총수수료': float(df['수수료'].sum(skipna=True)),
        '평균수수료': float(df['수수료'].mean(skipna=True)) if len(df) > 0 else np.nan,
        '평균면적': float(df['면적'].mean(skipna=True)) if len(df) > 0 else np.nan,
    }



def build_grouped_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if not group_cols:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            거래건수=('거래유형', 'size'),
            총거래가액=('거래가액', 'sum'),
            평균거래가액=('거래가액', 'mean'),
            최소거래가액=('거래가액', 'min'),
            최대거래가액=('거래가액', 'max'),
            총수수료=('수수료', 'sum'),
            평균수수료=('수수료', 'mean'),
            평균면적=('면적', 'mean'),
        )
        .reset_index()
        .sort_values(['총거래가액', '총수수료'], ascending=False)
    )
    return grouped



def create_chart_filtered_df(df: pd.DataFrame, trade_selections: List[str], build_selections: List[str]) -> pd.DataFrame:
    out = df.copy()
    out = apply_all_filter(out, '거래유형', trade_selections)
    out = apply_all_filter(out, '건물유형', build_selections)
    return out


# -----------------------------
# 인터랙티브 차트
# -----------------------------
def create_plotly_monthly_chart(df: pd.DataFrame):
    monthly = (
        df.groupby('거래연월_일자', dropna=False)
        .agg(거래건수=('거래유형', 'size'))
        .reset_index()
        .dropna(subset=['거래연월_일자'])
        .sort_values('거래연월_일자')
    )
    if monthly.empty:
        return None

    fig = px.line(
        monthly,
        x='거래연월_일자',
        y='거래건수',
        markers=True,
        title='월별 거래건수 추이',
    )
    fig.update_layout(template='plotly_white', xaxis_title='거래연월', yaxis_title='거래건수')
    return fig



def create_plotly_trade_type_chart(df: pd.DataFrame):
    summary = (
        df.groupby('거래유형', dropna=False)
        .agg(평균거래가액=('거래가액', 'mean'))
        .reset_index()
        .sort_values('평균거래가액', ascending=False)
    )
    if summary.empty:
        return None

    summary['평균거래가액(억원)'] = summary['평균거래가액'] / 100_000_000
    fig = px.bar(
        summary,
        x='거래유형',
        y='평균거래가액(억원)',
        title='거래유형별 평균 거래가액(억원)',
        text_auto='.2f',
    )
    fig.update_layout(template='plotly_white', xaxis_title='거래유형', yaxis_title='평균 거래가액(억원)')
    return fig



def create_plotly_build_type_chart(df: pd.DataFrame):
    summary = (
        df.groupby('건물유형', dropna=False)
        .agg(총수수료=('수수료', 'sum'))
        .reset_index()
        .sort_values('총수수료', ascending=False)
        .head(15)
    )
    if summary.empty:
        return None

    summary['총수수료(억원)'] = summary['총수수료'] / 100_000_000
    fig = px.bar(
        summary,
        x='건물유형',
        y='총수수료(억원)',
        title='건물유형별 총수수료(억원)',
        text_auto='.2f',
    )
    fig.update_layout(template='plotly_white', xaxis_title='건물유형', yaxis_title='총수수료(억원)')
    fig.update_xaxes(tickangle=35)
    return fig



def create_plotly_trade_amount_distribution(df: pd.DataFrame, bins: int):
    dist_df = df[df['거래가액(억원)'].notna()].copy()
    if dist_df.empty:
        return None

    fig = px.histogram(
        dist_df,
        x='거래가액(억원)',
        nbins=bins,
        marginal='box',
        title='거래가액 분포도(억원)',
    )
    fig.update_layout(template='plotly_white', xaxis_title='거래가액(억원)', yaxis_title='건수')
    return fig



def create_plotly_fee_distribution(df: pd.DataFrame, bins: int):
    dist_df = df[df['수수료(만원)'].notna()].copy()
    if dist_df.empty:
        return None

    fig = px.histogram(
        dist_df,
        x='수수료(만원)',
        nbins=bins,
        marginal='box',
        title='수수료 분포도(만원)',
    )
    fig.update_layout(template='plotly_white', xaxis_title='수수료(만원)', yaxis_title='건수')
    return fig



def create_plotly_spatial_scatter(df: pd.DataFrame, value_col: str):
    geo = df.dropna(subset=['lat', 'lon', value_col]).copy()
    if geo.empty:
        return None

    display_col = f'{value_col}_표시값'
    if value_col == '거래가액':
        geo[display_col] = geo[value_col] / 100_000_000
        color_label = '거래가액(억원)'
        geo['표시값라벨'] = geo[display_col].map(lambda x: f'{x:,.2f}억원')
    else:
        geo[display_col] = geo[value_col] / 10_000
        color_label = '수수료(만원)'
        geo['표시값라벨'] = geo[display_col].map(lambda x: f'{x:,.2f}만원')

    geo = geo.sort_values(display_col, ascending=True).copy()
    geo['마커크기'] = build_detail_marker_sizes(geo[display_col], min_size=5.0, max_size=18.0)

    fig = px.scatter(
        geo,
        x='lon',
        y='lat',
        color=display_col,
        size='마커크기',
        size_max=18,
        color_continuous_scale=PLOTLY_TURBO_SCALE,
        render_mode='webgl',
        hover_data={
            '단지명': True,
            '거래유형': True,
            '건물유형': True,
            '거래연월': True,
            '표시값라벨': True,
            'lat': ':.6f',
            'lon': ':.6f',
            display_col: False,
            '마커크기': False,
        },
        title=f'공간분포 산점도 ({color_label})',
    )
    fig.update_traces(
        marker=dict(
            opacity=0.82,
            line=dict(width=0.35, color='rgba(40, 40, 40, 0.35)'),
        ),
        selector=dict(mode='markers'),
    )
    fig.update_layout(
        template='plotly_white',
        xaxis_title='경도',
        yaxis_title='위도',
        coloraxis_colorbar_title=color_label,
    )
    return fig


# -----------------------------
# PDF용 정적 차트
# -----------------------------
def _save_matplotlib_figure(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=260, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf



def make_monthly_figure(df: pd.DataFrame) -> io.BytesIO:
    monthly = (
        df.groupby('거래연월_일자', dropna=False)
        .agg(거래건수=('거래유형', 'size'))
        .reset_index()
        .dropna(subset=['거래연월_일자'])
        .sort_values('거래연월_일자')
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly['거래연월_일자'], monthly['거래건수'], marker='o')
    ax.set_title('월별 거래건수 추이')
    ax.set_xlabel('거래연월')
    ax.set_ylabel('거래건수')
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _save_matplotlib_figure(fig)



def make_trade_type_figure(df: pd.DataFrame) -> io.BytesIO:
    summary = (
        df.groupby('거래유형', dropna=False)
        .agg(평균거래가액=('거래가액', 'mean'))
        .reset_index()
        .sort_values('평균거래가액', ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(summary['거래유형'].astype(str), summary['평균거래가액'] / 100_000_000, color=RED_SCALE[2])
    ax.set_title('거래유형별 평균 거래가액(억원)')
    ax.set_xlabel('거래유형')
    ax.set_ylabel('평균 거래가액(억원)')
    ax.grid(axis='y', alpha=0.25)
    return _save_matplotlib_figure(fig)



def make_build_type_figure(df: pd.DataFrame) -> io.BytesIO:
    summary = (
        df.groupby('건물유형', dropna=False)
        .agg(총수수료=('수수료', 'sum'))
        .reset_index()
        .sort_values('총수수료', ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(summary['건물유형'].astype(str), summary['총수수료'] / 100_000_000, color=RED_SCALE[3])
    ax.set_title('건물유형별 총수수료(억원)')
    ax.set_xlabel('건물유형')
    ax.set_ylabel('총수수료(억원)')
    ax.tick_params(axis='x', rotation=35)
    ax.grid(axis='y', alpha=0.25)
    return _save_matplotlib_figure(fig)



def make_trade_amount_distribution_figure(df: pd.DataFrame) -> io.BytesIO:
    plot_df = df[df['거래가액(억원)'].notna()].copy()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.hist(plot_df['거래가액(억원)'], bins=30, color=RED_SCALE[2], edgecolor='white')
    ax.set_title('거래가액 분포도(억원)')
    ax.set_xlabel('거래가액(억원)')
    ax.set_ylabel('건수')
    ax.grid(axis='y', alpha=0.25)
    return _save_matplotlib_figure(fig)



def make_fee_distribution_figure(df: pd.DataFrame) -> io.BytesIO:
    plot_df = df[df['수수료(만원)'].notna()].copy()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.hist(plot_df['수수료(만원)'], bins=30, color=RED_SCALE[3], edgecolor='white')
    ax.set_title('수수료 분포도(만원)')
    ax.set_xlabel('수수료(만원)')
    ax.set_ylabel('건수')
    ax.grid(axis='y', alpha=0.25)
    return _save_matplotlib_figure(fig)



def make_spatial_density_figure(df: pd.DataFrame, value_col: str) -> io.BytesIO:
    plot_df = df.dropna(subset=['lat', 'lon']).copy()
    fig, ax = plt.subplots(figsize=(7.5, 7))

    if len(plot_df) >= 3:
        hb = ax.hexbin(
            plot_df['lon'],
            plot_df['lat'],
            C=plot_df[value_col],
            gridsize=75,
            reduce_C_function=np.mean,
            cmap='turbo',
            mincnt=1,
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label(f'{value_col} 평균')
    else:
        ax.scatter(plot_df['lon'], plot_df['lat'], c=RED_SCALE[3])

    ax.set_title(f'공간분포(가중값: {value_col})')
    ax.set_xlabel('경도')
    ax.set_ylabel('위도')
    ax.grid(alpha=0.2)
    return _save_matplotlib_figure(fig)



def normalize_weights(series: pd.Series, power: float = 1.0) -> np.ndarray:
    s = pd.to_numeric(series, errors='coerce').fillna(0)
    if s.max() == s.min():
        base = np.ones(len(s))
    else:
        base = ((s - s.min()) / (s.max() - s.min()) + 0.01).to_numpy()
    return np.power(base, power)


def build_detail_marker_sizes(series: pd.Series, min_size: float = 5.0, max_size: float = 18.0) -> np.ndarray:
    s = pd.to_numeric(series, errors='coerce')
    if s.isna().all():
        return np.full(len(series), (min_size + max_size) / 2.0)

    ranked = s.fillna(s.median()).rank(method='average', pct=True)
    scaled = min_size + (max_size - min_size) * np.power(ranked.to_numpy(), 0.85)
    return scaled



def build_map(
    df: pd.DataFrame,
    value_col: str = '수수료',
    radius: int = 20,
    blur: int = 14,
    min_opacity: float = 0.30,
    max_zoom: int = 18,
    weight_power: float = 1.2,
    gradient: Optional[Dict[float, str]] = None,
) -> folium.Map:
    geo = df.dropna(subset=['lat', 'lon']).copy()
    if len(geo) == 0:
        raise ValueError('지도 시각화용 lat/lon 좌표가 없습니다.')

    center_lat = float(geo['lat'].mean())
    center_lon = float(geo['lon'].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True, tiles='CartoDB positron')

    weights = normalize_weights(geo[value_col], power=weight_power)
    heat_data = geo[['lat', 'lon']].copy()
    heat_data['weight'] = weights

    HeatMap(
        heat_data[['lat', 'lon', 'weight']].values.tolist(),
        radius=radius,
        blur=blur,
        min_opacity=min_opacity,
        max_zoom=max_zoom,
        gradient=gradient or HEATMAP_GRADIENTS['기본 빨강'],
        name='열지도',
    ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap



def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')



def create_gpkg_bytes(df: pd.DataFrame) -> bytes:
    geo = df.dropna(subset=['lat', 'lon']).copy()
    if len(geo) == 0:
        raise ValueError('GPKG 생성을 위한 좌표(lat/lon)가 없습니다.')

    gdf = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo['lon'], geo['lat']),
        crs='EPSG:4326',
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        gpkg_path = os.path.join(tmpdir, 'analysis_result.gpkg')
        gdf.to_file(gpkg_path, layer='transactions', driver='GPKG')

        with open(gpkg_path, 'rb') as f:
            return f.read()



def add_table(story, df: pd.DataFrame, max_rows: int = 15):
    if df.empty:
        story.append(Paragraph('표시할 데이터가 없습니다.', get_pdf_styles()['Body']))
        return

    show = df.head(max_rows).copy()
    for c in show.columns:
        if pd.api.types.is_numeric_dtype(show[c]):
            if c in ['총거래가액', '평균거래가액', '최소거래가액', '최대거래가액', '총수수료', '평균수수료']:
                show[c] = show[c].apply(lambda x: f'{x:,.0f}')
            else:
                show[c] = show[c].apply(lambda x: f'{x:,.2f}' if pd.notna(x) else '')
        else:
            show[c] = show[c].astype(str)

    data = [show.columns.tolist()] + show.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), PDF_FONT_NAME),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4e79')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor('#edf2f7')]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ]))
    story.append(table)



def get_pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='KTitle',
        fontName=PDF_FONT_NAME,
        fontSize=18,
        leading=24,
        textColor=colors.HexColor('#12355b'),
        alignment=TA_LEFT,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='KHeading',
        fontName=PDF_FONT_NAME,
        fontSize=12,
        leading=18,
        textColor=colors.HexColor('#1f4e79'),
        spaceBefore=8,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name='Body',
        fontName=PDF_FONT_NAME,
        fontSize=9,
        leading=14,
    ))
    return styles



def generate_pdf_report(filtered_df: pd.DataFrame,
                        summary_df: pd.DataFrame,
                        summary_stats: Dict[str, float],
                        filter_texts: List[str],
                        value_col_for_map: str,
                        overall_total_fee: float) -> bytes:
    styles = get_pdf_styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    story = []
    story.append(Paragraph('부동산 실거래가액 · 수수료 분석 Dashboard 보고서', styles['KTitle']))
    story.append(Paragraph(datetime.now().strftime('생성일시: %Y-%m-%d %H:%M'), styles['Body']))
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph('1. 분석 개요', styles['KHeading']))
    overview = (
        f"본 보고서는 선택 조건에 해당하는 거래 {summary_stats['건수']:,}건을 기준으로 거래가액과 중개보수(수수료)를 분석한 결과입니다. "
        f"선택 조건 평균 거래가액은 {format_won(summary_stats['평균거래가액'])}, 최소 거래가액은 {format_won(summary_stats['최소거래가액'])}, "
        f"최대 거래가액은 {format_won(summary_stats['최대거래가액'])}입니다. "
        f"전체 총수수료는 {format_won_to_eok(overall_total_fee)}, 선택 조건 총수수료는 {format_won_to_eok(summary_stats['총수수료'])}입니다."
    )
    story.append(Paragraph(overview, styles['Body']))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph('2. 적용 조건', styles['KHeading']))
    for txt in filter_texts:
        story.append(Paragraph(f'• {txt}', styles['Body']))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph('3. 핵심 KPI', styles['KHeading']))
    kpi_table = pd.DataFrame([
        ['거래건수', f"{summary_stats['건수']:,}건"],
        ['평균거래가액', format_won(summary_stats['평균거래가액'])],
        ['최소거래가액', format_won(summary_stats['최소거래가액'])],
        ['최대거래가액', format_won(summary_stats['최대거래가액'])],
        ['전체 총수수료', format_won_to_eok(overall_total_fee)],
        ['선택조건 총수수료', format_won_to_eok(summary_stats['총수수료'])],
        ['평균면적', f"{summary_stats['평균면적']:.2f}㎡" if pd.notna(summary_stats['평균면적']) else '-'],
    ], columns=['지표', '값'])
    add_table(story, kpi_table, max_rows=20)

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph('4. 그룹별 요약', styles['KHeading']))
    add_table(story, summary_df, max_rows=18)

    story.append(PageBreak())
    story.append(Paragraph('5. 차트 분석', styles['KHeading']))
    chart_buffers = [
        make_monthly_figure(filtered_df),
        make_trade_type_figure(filtered_df),
        make_build_type_figure(filtered_df),
        make_trade_amount_distribution_figure(filtered_df),
        make_fee_distribution_figure(filtered_df),
        make_spatial_density_figure(filtered_df, value_col_for_map),
    ]
    for chart_buf in chart_buffers:
        story.append(Image(chart_buf, width=180 * mm, height=80 * mm))
        story.append(Spacer(1, 4 * mm))

    story.append(Paragraph('6. 전문가 코멘트', styles['KHeading']))
    comment = (
        '거래가액과 수수료는 거래유형, 금액구간, 면적, 준공연도, 건물유형 등에 따라 분포가 달라집니다. '
        '실무 적용 시에는 단일 평균값보다 최소·최대·분포를 함께 확인하는 것이 적절합니다. '
        '열지도는 거래가액 또는 수수료가 상대적으로 집중되는 구역을 빠르게 식별하는 데 유용합니다.'
    )
    story.append(Paragraph(comment, styles['Body']))

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data



def build_filter_texts(df: pd.DataFrame) -> List[str]:
    if len(df) == 0:
        return ['현재 조건에 해당하는 데이터가 없습니다.']
    trade_text = ', '.join(sorted(df['거래유형'].dropna().astype(str).unique().tolist()))
    build_values = sorted(df['건물유형'].dropna().astype(str).unique().tolist())
    build_text = ', '.join(build_values[:10]) + (' ...' if len(build_values) > 10 else '')
    texts = [
        f"거래유형: {trade_text}",
        f"건물유형: {build_text}",
        f"거래연월 범위: {df['거래연월'].dropna().min()} ~ {df['거래연월'].dropna().max()}",
        f"면적 범위: {df['면적'].min():.2f}㎡ ~ {df['면적'].max():.2f}㎡",
        f"준공연도 범위: {int(df['준공연도'].min()) if df['준공연도'].notna().any() else '-'} ~ {int(df['준공연도'].max()) if df['준공연도'].notna().any() else '-'}",
    ]
    return texts



def build_download_bundle(csv_bytes: bytes, summary_csv_bytes: bytes, pdf_bytes: bytes, gpkg_bytes: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('analysis_result.csv', csv_bytes)
        zf.writestr('summary_result.csv', summary_csv_bytes)
        zf.writestr('analysis_report.pdf', pdf_bytes)
        zf.writestr('analysis_result.gpkg', gpkg_bytes)
    return buf.getvalue()


# -----------------------------
# 화면
# -----------------------------
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.expander('분석 기준 및 계산식', expanded=False):
    st.markdown(
        '''
        - **거래가액 계산식**
          - 매매 / 전세: `거래가액 = 거래가액(만원) × 10,000`
          - 월세: `거래가액 = ((임대료 × 100) + 거래가액(만원)) × 10,000`
        - **수수료 계산식**
          - 거래가액 × 상한요율
          - 단, 한도액이 있으면 `min(거래가액 × 상한요율, 한도액)` 적용
        - **차트 분석**
          - 거래유형 / 건물유형 조건별 인터랙티브 조회
          - 거래가액 분포도 / 수수료 분포도 추가
        - **지도 분석**
          - 열지도(Heatmap) 반경, 블러, 투명도, 강조강도, 색상강도 설정 가능
        '''
    )

uploaded_file = st.file_uploader('CSV 파일 업로드', type=['csv'])

if uploaded_file is None:
    st.info('`실거래가 CSV` 파일을 업로드하면 분석이 시작됩니다.')
    st.stop()

try:
    raw_df = safe_read_csv(uploaded_file)
    std_df = standardize_columns(raw_df)
except Exception as e:
    st.error(f'파일 처리 중 오류가 발생했습니다: {e}')
    st.stop()

with st.sidebar:
    st.header('거래유형 값 설정')
    sale_label = st.text_input('매매 값', value='매매')
    jeonse_label = st.text_input('전세 값', value='전세')
    monthly_label = st.text_input('월세 값', value='월세')

analysis_df = add_derived_columns(std_df, sale_label, jeonse_label, monthly_label)
filtered_df = build_filters(analysis_df)

st.success(f'원본 {len(analysis_df):,}건 중 현재 조건에 해당하는 거래는 {len(filtered_df):,}건입니다.')

if len(filtered_df) == 0:
    st.warning('현재 조건에는 해당 데이터가 없습니다. 필터를 완화하세요.')
    st.stop()

summary = compute_summary(filtered_df)
overall_summary = compute_summary(analysis_df)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric('거래건수', f"{summary['건수']:,}건")
kpi2.metric('평균거래가액', format_won_to_eok(summary['평균거래가액']))
kpi3.metric('최소거래가액', format_won_to_eok(summary['최소거래가액']))
kpi4.metric('최대거래가액', format_won_to_eok(summary['최대거래가액']))

kpi5, kpi6 = st.columns(2)
kpi5.metric('전체 총수수료(억원)', format_won_to_eok(overall_summary['총수수료']))
kpi6.metric('선택조건 총수수료(억원)', format_won_to_eok(summary['총수수료']))

st.markdown('---')

left, right = st.columns([1.15, 0.85])
with left:
    st.subheader('거래가액 · 수수료 계산 결과 데이터')
    display_cols = [
        '행정동주소', '단지명', '거래유형', '면적', '거래연월', '거래가액(만원)', '임대료',
        '층', '준공연도', '건물유형', '거래가액', '거래금액구간', '상한요율', '한도액', '수수료', 'lat', 'lon'
    ]
    existing_display_cols = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[existing_display_cols], use_container_width=True, height=420)

with right:
    st.subheader('그룹별 요약 분석')
    group_options = {
        '거래유형': '거래유형',
        '건물유형': '건물유형',
        '거래연월': '거래연월',
        '준공연도': '준공연도',
        '거래금액구간': '거래금액구간',
    }
    selected_groups = st.multiselect('집계 기준 선택', list(group_options.keys()), default=['거래유형', '거래연월'])
    group_cols = [group_options[k] for k in selected_groups]
    grouped_summary = build_grouped_summary(filtered_df, group_cols)
    st.dataframe(grouped_summary, use_container_width=True, height=420)

st.markdown('---')
st.subheader('차트 분석')

chart_filter_col1, chart_filter_col2, chart_filter_col3, chart_filter_col4 = st.columns(4)
chart_trade_options = ['all'] + sorted(filtered_df['거래유형'].dropna().astype(str).unique().tolist())
chart_build_options = ['all'] + sorted(filtered_df['건물유형'].dropna().astype(str).unique().tolist())

with chart_filter_col1:
    chart_trade_selection = st.multiselect('차트용 거래유형', chart_trade_options, default=['all'], key='chart_trade_selection')
with chart_filter_col2:
    chart_build_selection = st.multiselect('차트용 건물유형', chart_build_options, default=['all'], key='chart_build_selection')
with chart_filter_col3:
    chart_bins = st.slider('분포도 구간 수', min_value=10, max_value=80, value=30, step=5)
with chart_filter_col4:
    chart_metric_for_space = st.selectbox('공간분포 가중값', ['수수료', '거래가액'], index=0, key='chart_metric_for_space')

chart_filtered_df = create_chart_filtered_df(filtered_df, chart_trade_selection, chart_build_selection)

if len(chart_filtered_df) == 0:
    st.warning('차트용 거래유형/건물유형 선택 조건에 해당하는 데이터가 없습니다.')
else:
    chart_left, chart_right = st.columns(2)
    with chart_left:
        monthly_fig = create_plotly_monthly_chart(chart_filtered_df)
        if monthly_fig is not None:
            st.plotly_chart(monthly_fig, use_container_width=True)

        build_type_fig = create_plotly_build_type_chart(chart_filtered_df)
        if build_type_fig is not None:
            st.plotly_chart(build_type_fig, use_container_width=True)

        fee_dist_fig = create_plotly_fee_distribution(chart_filtered_df, chart_bins)
        if fee_dist_fig is not None:
            st.plotly_chart(fee_dist_fig, use_container_width=True)

    with chart_right:
        trade_type_fig = create_plotly_trade_type_chart(chart_filtered_df)
        if trade_type_fig is not None:
            st.plotly_chart(trade_type_fig, use_container_width=True)

        trade_amount_dist_fig = create_plotly_trade_amount_distribution(chart_filtered_df, chart_bins)
        if trade_amount_dist_fig is not None:
            st.plotly_chart(trade_amount_dist_fig, use_container_width=True)

        spatial_scatter_fig = create_plotly_spatial_scatter(chart_filtered_df, chart_metric_for_space)
        if spatial_scatter_fig is not None:
            st.plotly_chart(spatial_scatter_fig, use_container_width=True)

st.markdown('---')
st.subheader('지도 분석: 열지도')

map_cfg_col1, map_cfg_col2, map_cfg_col3 = st.columns(3)
with map_cfg_col1:
    map_metric = st.selectbox('열지도 가중값', ['수수료', '거래가액'], index=0, key='map_metric')
    heat_radius = st.slider('열지도 반경', min_value=8, max_value=40, value=20, step=2)
with map_cfg_col2:
    heat_blur = st.slider('블러 강도', min_value=5, max_value=35, value=14, step=1)
    heat_min_opacity = st.slider('최소 투명도', min_value=0.05, max_value=1.00, value=0.30, step=0.05)
with map_cfg_col3:
    heat_max_zoom = st.slider('최대 확대 줌', min_value=10, max_value=22, value=18, step=1)
    heat_weight_power = st.slider('강조 강도', min_value=0.5, max_value=3.0, value=1.2, step=0.1)

heat_gradient_options = list(HEATMAP_GRADIENTS.keys())
heat_gradient_name = st.selectbox(
    '열지도 색상 강도',
    heat_gradient_options,
    index=heat_gradient_options.index('터보'),
)

try:
    fmap = build_map(
        filtered_df,
        value_col=map_metric,
        radius=heat_radius,
        blur=heat_blur,
        min_opacity=heat_min_opacity,
        max_zoom=heat_max_zoom,
        weight_power=heat_weight_power,
        gradient=HEATMAP_GRADIENTS[heat_gradient_name],
    )
    st_folium(fmap, width='100%', height=720)
except Exception as e:
    st.warning(f'지도 생성에 실패했습니다: {e}')

st.markdown('---')
st.subheader('다운로드')

filter_texts = build_filter_texts(filtered_df)
summary_for_pdf = grouped_summary if len(grouped_summary) > 0 else build_grouped_summary(filtered_df, ['거래유형'])

try:
    pdf_bytes = generate_pdf_report(
        filtered_df=filtered_df,
        summary_df=summary_for_pdf,
        summary_stats=summary,
        filter_texts=filter_texts,
        value_col_for_map=map_metric,
        overall_total_fee=overall_summary['총수수료'],
    )
except Exception as e:
    pdf_bytes = None
    st.warning(f'PDF 생성에 실패했습니다: {e}')

csv_bytes = dataframe_to_csv_bytes(filtered_df)
summary_csv_bytes = dataframe_to_csv_bytes(summary_for_pdf)

try:
    gpkg_bytes = create_gpkg_bytes(filtered_df)
except Exception as e:
    gpkg_bytes = None
    st.warning(f'GPKG 생성에 실패했습니다: {e}')

btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
with btn_col1:
    st.download_button(
        '결과 CSV 다운로드',
        data=csv_bytes,
        file_name='부동산_실거래가_분석결과.csv',
        mime='text/csv',
        use_container_width=True,
    )
with btn_col2:
    st.download_button(
        '요약 CSV 다운로드',
        data=summary_csv_bytes,
        file_name='부동산_실거래가_요약분석.csv',
        mime='text/csv',
        use_container_width=True,
    )
with btn_col3:
    st.download_button(
        'PDF 다운로드',
        data=pdf_bytes if pdf_bytes is not None else b'',
        file_name='부동산_실거래가_분석보고서.pdf',
        mime='application/pdf',
        disabled=pdf_bytes is None,
        use_container_width=True,
    )
with btn_col4:
    st.download_button(
        'GPKG 다운로드',
        data=gpkg_bytes if gpkg_bytes is not None else b'',
        file_name='부동산_실거래가_분석결과.gpkg',
        mime='application/octet-stream',
        disabled=gpkg_bytes is None,
        use_container_width=True,
    )

if pdf_bytes is not None and gpkg_bytes is not None:
    bundle_bytes = build_download_bundle(csv_bytes, summary_csv_bytes, pdf_bytes, gpkg_bytes)
    st.download_button(
        '전체 결과 ZIP 다운로드',
        data=bundle_bytes,
        file_name='부동산_실거래가_분석결과_패키지.zip',
        mime='application/zip',
        use_container_width=True,
    )

with st.expander('Streamlit 배포용 requirements.txt 예시', expanded=False):
    st.code(
        '\n'.join([
            'streamlit',
            'pandas',
            'numpy',
            'matplotlib',
            'plotly',
            'folium',
            'streamlit-folium',
            'geopandas',
            'shapely',
            'pyogrio',
            'reportlab',
        ]),
        language='text'
    )
