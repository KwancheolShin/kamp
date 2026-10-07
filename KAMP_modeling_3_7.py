# -*- coding: utf-8 -*-
"""
KAMP 전력사용량 예측 모델링 — Spyder 실행용
(KAMP_modeling_3_7 (1).ipynb 를 .py 로 변환)

- Baseline XGBoost / Feature4 XGBoost / Persistence 비교
- TimeSeriesSplit 검증
- 월별 최대전력 피크 · 고전력 위험구간 분석
- 고전력 경보 F1, FP/FN 분석, TreeSHAP 해석

실행 방법 (Spyder)
  1) 이 파일을 Power15MinClean_NoDatetime.csv 와 같은 폴더에 둔다.
  2) F5(파일 전체 실행) 또는 Ctrl+Enter(# %% 셀 단위 실행).
  3) 그림은 Plots 창에 표시되고, 결과 CSV 는 ./outputs 폴더에 저장된다.

필요 패키지: pandas, numpy, matplotlib, scikit-learn, xgboost
  (xgboost 가 없으면 Anaconda Prompt 에서  pip install xgboost)

원본 노트북과 달라진 점(실행 환경 차이만 반영, 모델·분할·파라미터는 동일)
  - Colab 전용 코드 제거: !apt-get, google.colab.files.upload/download
  - 한글 폰트: NanumGothic -> Windows 기본 'Malgun Gothic'
  - display() -> print()
  - 결과 파일 저장 위치: ./outputs
"""

# %% #0. 환경 설정 / 라이브러리 / 한글 폰트
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

from xgboost import XGBRegressor, DMatrix
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import TimeSeriesSplit

# 데이터 폴더: 스크립트가 있는 폴더 (셀 단위 실행 시에는 현재 작업 폴더)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_FILE = os.path.join(BASE_DIR, 'Power15MinClean_NoDatetime.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한글 폰트 적용 (Windows: 맑은 고딕, 없으면 나눔고딕 시도)
available_fonts = {f.name for f in fm.fontManager.ttflist}
for font_name in ['Malgun Gothic', 'NanumGothic', 'AppleGothic']:
    if font_name in available_fonts:
        plt.rcParams['font.family'] = font_name
        break
else:
    print("경고: 한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False

# 콘솔에서 표가 잘리지 않도록
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.unicode.east_asian_width', True)


# ------------------------------------------------------------
# 공통 함수
# ------------------------------------------------------------

def make_xgb_model():
    # 현재 프로젝트에서 사용한 XGBoost 설정을 그대로 반환
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )


def regression_metrics(y_true, y_pred):
    # MAE, RMSE, R² 계산
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }


def save_csv(df, file_name):
    path = os.path.join(OUTPUT_DIR, file_name)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print('-', file_name, df.shape)


print("환경 설정 완료")
print("적용 폰트:", plt.rcParams['font.family'])
print("데이터 파일:", DATA_FILE)
print("결과 저장 폴더:", OUTPUT_DIR)


# %% #1. 데이터 불러오기
df_raw = pd.read_csv(DATA_FILE)

print("데이터 크기:", df_raw.shape)
print("\n컬럼:")
print(df_raw.columns.tolist())
print(df_raw.head())


# %% #2. Baseline 데이터 준비
# Baseline 에서는 현재 전력으로 계산된 '평균'(시간 평균 = 누수),
# 사후 생성 가능성이 있는 '이상구간여부' 를 입력에서 제외한다.
df_base = df_raw.copy()

df_base['날짜_dt'] = pd.to_datetime(
    df_base['날짜'].astype(str),
    format='%Y%m%d'
)

df_base = (
    df_base
    .sort_values(['날짜_dt', '시간', '분'])
    .reset_index(drop=True)
)

baseline_features = [
    '시간', '분', '생산량', '기온', '풍속', '습도', '강수량',
    '전기요금(계절)', 'day', 'd', 'm', '공장인원', '인건비'
]

target = '전력사용량'

print("Baseline Feature 수:", len(baseline_features))
print(baseline_features)


# %% #3. Baseline 시간순 Train / Test 8:2 분할 및 학습
split_idx_base = int(len(df_base) * 0.8)

train_base = df_base.iloc[:split_idx_base].copy()
test_base = df_base.iloc[split_idx_base:].copy()

X_train_base0 = train_base[baseline_features]
X_test_base0 = test_base[baseline_features]
y_train_base0 = train_base[target]
y_test_base0 = test_base[target]

baseline_model_initial = make_xgb_model()
baseline_model_initial.fit(X_train_base0, y_train_base0)

baseline_train_pred0 = baseline_model_initial.predict(X_train_base0)
baseline_test_pred0 = baseline_model_initial.predict(X_test_base0)

baseline_train_metrics0 = regression_metrics(y_train_base0, baseline_train_pred0)
baseline_test_metrics0 = regression_metrics(y_test_base0, baseline_test_pred0)

print("Train 데이터:", X_train_base0.shape)
print("Test 데이터 :", X_test_base0.shape)
print("\nTrain 기간:", train_base['날짜_dt'].min(), "~", train_base['날짜_dt'].max())
print("Test 기간 :", test_base['날짜_dt'].min(), "~", test_base['날짜_dt'].max())

for name, m in [('Train', baseline_train_metrics0), ('Test', baseline_test_metrics0)]:
    print(f"\n========== Baseline {name} 성능 ==========")
    print(f"MAE  : {m['MAE']:.4f}")
    print(f"RMSE : {m['RMSE']:.4f}")
    print(f"R²   : {m['R2']:.4f}")

print("\n========== Train / Test 차이 ==========")
print(f"MAE 차이  : {baseline_test_metrics0['MAE'] - baseline_train_metrics0['MAE']:.4f}")
print(f"RMSE 차이 : {baseline_test_metrics0['RMSE'] - baseline_train_metrics0['RMSE']:.4f}")
print(f"R² 차이   : {baseline_train_metrics0['R2'] - baseline_test_metrics0['R2']:.4f}")


# %% #4. Baseline 실제값 vs 예측값
baseline_test_time = (
    test_base['날짜_dt'].dt.normalize()
    + pd.to_timedelta(test_base['시간'], unit='h')
    + pd.to_timedelta(test_base['분'], unit='m')
)

plt.figure(figsize=(20, 7))
plt.plot(baseline_test_time, y_test_base0.values, label='실제 전력사용량', linewidth=1)
plt.plot(baseline_test_time, baseline_test_pred0, label='Baseline 예측 전력사용량', linewidth=1)
plt.xlabel('실제 시간')
plt.ylabel('전력사용량')
plt.title('Baseline 실제 전력사용량 vs 예측 전력사용량')
plt.legend()
plt.grid(alpha=0.3)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% #5. Baseline 예측결과 및 대규모 오차 구간
baseline_result = test_base.copy()
baseline_result['실제전력'] = y_test_base0.values
baseline_result['예측전력'] = baseline_test_pred0
baseline_result['오차'] = baseline_result['실제전력'] - baseline_result['예측전력']
baseline_result['절대오차'] = np.abs(baseline_result['오차'])

baseline_error_threshold = baseline_result['절대오차'].quantile(0.95)

baseline_large_error = baseline_result[
    baseline_result['절대오차'] >= baseline_error_threshold
].copy()

print(f"절대오차 상위 5% 기준값: {baseline_error_threshold:.4f}")
print(f"전체 Test 데이터 수: {len(baseline_result)}")
print(f"대규모 오차 데이터 수: {len(baseline_large_error)}")

print(baseline_large_error[
    ['날짜_dt', '시간', '분', '생산량', '공장인원', '전력사용량', '예측전력', '절대오차']
].head(20))


# %% #6. 대규모 오차 구간 주요 조건 비교
compare_variables = ['생산량', '시간', 'day', 'm', '공장인원', '인건비', '기온']

baseline_error_comparison = pd.DataFrame({
    '전체 Test 평균': baseline_result[compare_variables].mean(),
    '대규모 오차 평균': baseline_large_error[compare_variables].mean()
})
baseline_error_comparison['차이'] = (
    baseline_error_comparison['대규모 오차 평균']
    - baseline_error_comparison['전체 Test 평균']
)
print(baseline_error_comparison)

overall_abnormal_rate = baseline_result['이상구간여부'].mean() * 100
large_error_abnormal_rate = baseline_large_error['이상구간여부'].mean() * 100
print(f"\n전체 Test 이상구간 비율: {overall_abnormal_rate:.2f}%")
print(f"대규모 오차 구간 이상구간 비율: {large_error_abnormal_rate:.2f}%")


# %% #7. Baseline 오차 패턴 시각화
for col, xlabel, title, figsize in [
    ('시간', '시간', '시간대별 평균 예측오차', (12, 5)),
    ('day', '요일 번호', '요일별 평균 예측오차', (9, 5)),
    ('m', '월', '월별 평균 예측오차', (9, 5)),
]:
    err = baseline_result.groupby(col)['절대오차'].mean().reset_index()
    plt.figure(figsize=figsize)
    plt.bar(err[col], err['절대오차'])
    plt.xlabel(xlabel)
    plt.ylabel('평균 절대오차')
    plt.title(title)
    if col == '시간':
        plt.xticks(range(24))
    plt.grid(axis='y', alpha=0.3)
    plt.show()

plt.figure(figsize=(9, 6))
plt.scatter(baseline_result['생산량'], baseline_result['절대오차'], alpha=0.4)
plt.xlabel('생산량')
plt.ylabel('절대오차')
plt.title('생산량과 예측오차 관계')
plt.grid(alpha=0.3)
plt.show()


# %% #8. 정확히 4개 Feature 추가
# 원본 CSV 기준으로 새 데이터프레임 생성 → Baseline 임시 컬럼이 섞이지 않도록 함
df_feature4 = df_raw.copy()
original_columns = df_feature4.columns.tolist()

df_feature4 = (
    df_feature4
    .sort_values(['날짜', '시간', '분'])
    .reset_index(drop=True)
)

# ① 15분 전 전력
df_feature4['lag_1_power'] = df_feature4['전력사용량'].shift(1)

# ② 1시간 전 전력 (15분 간격이므로 4행 전)
df_feature4['lag_4_power'] = df_feature4['전력사용량'].shift(4)

# 이전 생산량은 계산용 Series 로만 사용 (컬럼으로 저장하지 않음)
prev_production = df_feature4['생산량'].shift(1)

# ③ 생산량 변화량
df_feature4['production_change'] = df_feature4['생산량'] - prev_production

# ④ 생산 전환 상태
conditions = [
    (prev_production == 0) & (df_feature4['생산량'] == 0),
    (prev_production == 0) & (df_feature4['생산량'] > 0),
    (prev_production > 0) & (df_feature4['생산량'] == 0),
    (prev_production > 0) & (df_feature4['생산량'] > 0)
]
choices = ['비생산지속', '생산시작', '생산종료전환', '생산지속']

df_feature4['operation_state'] = np.select(conditions, choices, default=None)

new_features = ['lag_1_power', 'lag_4_power', 'production_change', 'operation_state']

added_columns = [c for c in df_feature4.columns if c not in original_columns]

# 정확히 4개만 추가되었는지 검증
assert added_columns == new_features, f"추가 컬럼이 예상과 다릅니다: {added_columns}"
assert len(df_feature4.columns) == len(original_columns) + 4

print("원래 컬럼 수:", len(original_columns))
print("추가된 Feature:", added_columns)
print("추가 Feature 개수:", len(added_columns))
print("최종 컬럼 수:", len(df_feature4.columns))
print(df_feature4[
    ['날짜', '시간', '분', '전력사용량', '생산량',
     'lag_1_power', 'lag_4_power', 'production_change', 'operation_state']
].head(12))


# %% #9. Feature4 모델 입력 데이터 생성
# Lag 때문에 생기는 초반 결측행 제거
df_exp = df_feature4.dropna(subset=new_features).reset_index(drop=True)

new_numeric_features = ['lag_1_power', 'lag_4_power', 'production_change']

# operation_state 는 모델 입력에서만 임시 One-Hot Encoding
state_encoded = pd.get_dummies(
    df_exp['operation_state'],
    prefix='operation_state',
    dtype=int
)

X_feature4 = pd.concat(
    [df_exp[baseline_features + new_numeric_features], state_encoded],
    axis=1
)
y_feature4 = df_exp[target]

print("저장용 Feature4 데이터:", df_feature4.shape)
print("모델 학습용 데이터:", df_exp.shape)
print("모델 입력 X:", X_feature4.shape)
print("\n모델 입력 컬럼:")
print(X_feature4.columns.tolist())


# %% #10. 동일 데이터에서 시간순 Train / Test 8:2 분할
# Feature4 는 lag 로 앞 4행이 제거되므로 Baseline 도 같은 행으로 재학습해 공정 비교
split_idx = int(len(df_exp) * 0.8)

train_df = df_exp.iloc[:split_idx].copy()
test_df = df_exp.iloc[split_idx:].copy()

X_train_feature4 = X_feature4.iloc[:split_idx]
X_test_feature4 = X_feature4.iloc[split_idx:]

X_baseline_fair = df_exp[baseline_features]
X_train_baseline = X_baseline_fair.iloc[:split_idx]
X_test_baseline = X_baseline_fair.iloc[split_idx:]

y_train = y_feature4.iloc[:split_idx]
y_test = y_feature4.iloc[split_idx:]

print("Train 데이터:", X_train_feature4.shape)
print("Test 데이터 :", X_test_feature4.shape)
print(f"Train 비율: {len(X_train_feature4) / len(df_exp):.2%}")
print(f"Test 비율 : {len(X_test_feature4) / len(df_exp):.2%}")


# %% #11. Baseline / Feature4 XGBoost 학습
baseline_model = make_xgb_model()
feature4_model = make_xgb_model()

baseline_model.fit(X_train_baseline, y_train)
feature4_model.fit(X_train_feature4, y_train)

baseline_train_pred = baseline_model.predict(X_train_baseline)
baseline_test_pred = baseline_model.predict(X_test_baseline)
feature4_train_pred = feature4_model.predict(X_train_feature4)
feature4_test_pred = feature4_model.predict(X_test_feature4)

print("Baseline / Feature4 모델 학습 완료")


# %% #12. Baseline vs Feature4 성능 비교
baseline_train_metrics = regression_metrics(y_train, baseline_train_pred)
baseline_test_metrics = regression_metrics(y_test, baseline_test_pred)
feature4_train_metrics = regression_metrics(y_train, feature4_train_pred)
feature4_test_metrics = regression_metrics(y_test, feature4_test_pred)

comparison_model = pd.DataFrame({
    '모델': ['Baseline', 'Feature4'],
    'Train MAE': [baseline_train_metrics['MAE'], feature4_train_metrics['MAE']],
    'Test MAE': [baseline_test_metrics['MAE'], feature4_test_metrics['MAE']],
    'Train RMSE': [baseline_train_metrics['RMSE'], feature4_train_metrics['RMSE']],
    'Test RMSE': [baseline_test_metrics['RMSE'], feature4_test_metrics['RMSE']],
    'Train R2': [baseline_train_metrics['R2'], feature4_train_metrics['R2']],
    'Test R2': [baseline_test_metrics['R2'], feature4_test_metrics['R2']]
})
comparison_model['R2 Gap'] = comparison_model['Train R2'] - comparison_model['Test R2']

print(comparison_model.round(4))

print("\n========== Feature 추가 효과 ==========")
print(f"Test MAE 변화 : {baseline_test_metrics['MAE']:.4f} → {feature4_test_metrics['MAE']:.4f}")
print(f"Test RMSE 변화: {baseline_test_metrics['RMSE']:.4f} → {feature4_test_metrics['RMSE']:.4f}")
print(f"Test R² 변화  : {baseline_test_metrics['R2']:.4f} → {feature4_test_metrics['R2']:.4f}")
print(f"\nR² 개선량 : {feature4_test_metrics['R2'] - baseline_test_metrics['R2']:+.4f}")
print(f"RMSE 감소량: {baseline_test_metrics['RMSE'] - feature4_test_metrics['RMSE']:+.4f}")
print(f"MAE 감소량 : {baseline_test_metrics['MAE'] - feature4_test_metrics['MAE']:+.4f}")


# %% #13. 실제값 / Baseline / Feature4 예측 비교
test_time = (
    pd.to_datetime(test_df['날짜'].astype(str), format='%Y%m%d')
    + pd.to_timedelta(test_df['시간'], unit='h')
    + pd.to_timedelta(test_df['분'], unit='m')
)

plt.figure(figsize=(20, 7))
plt.plot(test_time, y_test.values, label='실제 전력', linewidth=1.2)
plt.plot(test_time, baseline_test_pred, label='Baseline 예측', alpha=0.7)
plt.plot(test_time, feature4_test_pred, label='Feature4 예측', alpha=0.8)
plt.xlabel('실제 시간')
plt.ylabel('전력사용량')
plt.title('Baseline vs Feature4 전력사용량 예측 비교')
plt.legend()
plt.grid(alpha=0.3)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% #14. Feature4 모델 대규모 오차 분석
feature4_result = test_df.copy()
feature4_result['실제전력'] = y_test.values
feature4_result['예측전력'] = feature4_test_pred
feature4_result['오차'] = feature4_result['실제전력'] - feature4_result['예측전력']
feature4_result['절대오차'] = np.abs(feature4_result['오차'])

feature4_error_threshold = feature4_result['절대오차'].quantile(0.95)
feature4_large_error = feature4_result[
    feature4_result['절대오차'] >= feature4_error_threshold
].copy()

print("Feature4 절대오차 상위 5% 기준:", round(feature4_error_threshold, 4))
print("대규모 오차 데이터 수:", len(feature4_large_error))
print(feature4_large_error[
    ['날짜', '시간', '분', '생산량', 'lag_1_power', 'lag_4_power',
     'production_change', 'operation_state', '실제전력', '예측전력', '절대오차']
].head(30))


# %% #15. Feature4 모델 변수 중요도
feature_importance = pd.DataFrame({
    '변수': X_feature4.columns,
    '중요도': feature4_model.feature_importances_
}).sort_values('중요도', ascending=False).reset_index(drop=True)

print(feature_importance)

plot_data = feature_importance.sort_values('중요도', ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(plot_data['변수'], plot_data['중요도'])
plt.xlabel('Feature Importance')
plt.ylabel('변수')
plt.title('Feature 4개 추가 후 변수 중요도')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# %% #16. Persistence Baseline vs Feature4
# Persistence: 다음 값을 직전 실제 전력값으로 예측 (P̂_t = P_{t-1})
persistence_pred = df_exp['lag_1_power'].iloc[split_idx:].values
persistence_metrics = regression_metrics(y_test, persistence_pred)

persistence_comparison = pd.DataFrame({
    '모델': ['Persistence', 'Feature4 XGBoost'],
    'MAE': [persistence_metrics['MAE'], feature4_test_metrics['MAE']],
    'RMSE': [persistence_metrics['RMSE'], feature4_test_metrics['RMSE']],
    'R2': [persistence_metrics['R2'], feature4_test_metrics['R2']]
})
print(persistence_comparison.round(4))

print("\n========== XGBoost 개선량 ==========")
print(f"R² 개선 : {feature4_test_metrics['R2'] - persistence_metrics['R2']:+.4f}")
print(f"MAE 감소 : {persistence_metrics['MAE'] - feature4_test_metrics['MAE']:+.4f}")
print(f"RMSE 감소 : {persistence_metrics['RMSE'] - feature4_test_metrics['RMSE']:+.4f}")


# %% #17. TimeSeriesSplit 5-Fold 검증 (15분 ahead rolling prediction)
tscv = TimeSeriesSplit(n_splits=5)
cv_results = []

for fold, (train_idx, valid_idx) in enumerate(tscv.split(X_feature4), start=1):
    X_train_cv = X_feature4.iloc[train_idx]
    X_valid_cv = X_feature4.iloc[valid_idx]
    y_train_cv = y_feature4.iloc[train_idx]
    y_valid_cv = y_feature4.iloc[valid_idx]

    model_cv = make_xgb_model()
    model_cv.fit(X_train_cv, y_train_cv)
    xgb_metrics_cv = regression_metrics(y_valid_cv, model_cv.predict(X_valid_cv))

    # 동일 Validation 구간 Persistence
    persistence_pred_cv = df_exp['lag_1_power'].iloc[valid_idx].values
    persistence_metrics_cv = regression_metrics(y_valid_cv, persistence_pred_cv)

    valid_dates = df_exp.iloc[valid_idx]['날짜']
    cv_results.append({
        'Fold': fold,
        '검증 시작': valid_dates.iloc[0],
        '검증 종료': valid_dates.iloc[-1],
        'XGB MAE': xgb_metrics_cv['MAE'],
        'XGB RMSE': xgb_metrics_cv['RMSE'],
        'XGB R2': xgb_metrics_cv['R2'],
        'Persistence MAE': persistence_metrics_cv['MAE'],
        'Persistence RMSE': persistence_metrics_cv['RMSE'],
        'Persistence R2': persistence_metrics_cv['R2']
    })
    print(f"Fold {fold} 완료")

cv_results = pd.DataFrame(cv_results)
print(cv_results.round(4))


# %% #18. TimeSeriesSplit 평균 성능 및 Fold별 R²
for label, prefix in [('Feature4 XGBoost', 'XGB'), ('Persistence', 'Persistence')]:
    print(f"========== {label} ==========")
    print(f"평균 MAE  : {cv_results[f'{prefix} MAE'].mean():.4f}")
    print(f"평균 RMSE : {cv_results[f'{prefix} RMSE'].mean():.4f}")
    print(f"평균 R²   : {cv_results[f'{prefix} R2'].mean():.4f}")
    print(f"R² 표준편차: {cv_results[f'{prefix} R2'].std():.4f}\n")

plt.figure(figsize=(10, 6))
plt.plot(cv_results['Fold'], cv_results['XGB R2'], marker='o', label='Feature4 XGBoost')
plt.plot(cv_results['Fold'], cv_results['Persistence R2'], marker='o', label='Persistence')
plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('TimeSeriesSplit Fold별 R² 비교')
plt.xticks(cv_results['Fold'])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% #19. Feature4 데이터 / 예측결과 저장
print("저장 완료:")
save_csv(df_feature4, 'Power15MinClean_Feature4.csv')      # 기존 17개 + 정확히 4개 Feature
save_csv(feature4_result, 'feature4_prediction_result.csv')  # Test 예측 결과


# %% #20. 최종 Feature4 모델 대규모 오차 정의
error_threshold = feature4_result['절대오차'].quantile(0.95)
feature4_result['대규모오차'] = feature4_result['절대오차'] >= error_threshold

print(f"절대오차 상위 5% 기준값: {error_threshold:.4f}")
print(f"대규모 오차 데이터 수: {feature4_result['대규모오차'].sum()}")
print(f"전체 Test 데이터 수: {len(feature4_result)}")


# %% #21. 생산량 조건별 대규모 오차율
result_analysis = feature4_result.copy()

# 생산량 0은 별도 구분, 생산량 > 0 구간은 4분위수로 구분
positive_prod = result_analysis.loc[result_analysis['생산량'] > 0, '생산량']
prod_quartile = pd.qcut(positive_prod, q=4, duplicates='drop')

result_analysis['생산량구간'] = '생산량=0'
result_analysis.loc[result_analysis['생산량'] > 0, '생산량구간'] = prod_quartile.astype(str)

prod_error_analysis = (
    result_analysis
    .groupby('생산량구간')
    .agg(
        데이터수=('절대오차', 'size'),
        평균생산량=('생산량', 'mean'),
        평균절대오차=('절대오차', 'mean'),
        RMSE=('오차', lambda x: np.sqrt(np.mean(x ** 2))),
        대규모오차수=('대규모오차', 'sum'),
        대규모오차율=('대규모오차', 'mean')
    )
    .reset_index()
)
prod_error_analysis['대규모오차율'] *= 100
print(prod_error_analysis.round(4))


# %% #22. 생산량 변화 방향별 오차 분석
result_analysis['생산량변화상태'] = np.select(
    [result_analysis['production_change'] < 0,
     result_analysis['production_change'] == 0,
     result_analysis['production_change'] > 0],
    ['생산량 감소', '생산량 변화없음', '생산량 증가'],
    default='기타'
)

production_change_analysis = (
    result_analysis
    .groupby('생산량변화상태')
    .agg(
        데이터수=('절대오차', 'size'),
        평균생산량변화=('production_change', 'mean'),
        평균절대오차=('절대오차', 'mean'),
        대규모오차수=('대규모오차', 'sum'),
        대규모오차율=('대규모오차', 'mean')
    )
    .reset_index()
)
production_change_analysis['대규모오차율'] *= 100
print(production_change_analysis.round(4))


# %% #23. 생산 전환상태별 오차 분석
operation_error_analysis = (
    result_analysis
    .groupby('operation_state')
    .agg(
        데이터수=('절대오차', 'size'),
        평균절대오차=('절대오차', 'mean'),
        RMSE=('오차', lambda x: np.sqrt(np.mean(x ** 2))),
        대규모오차수=('대규모오차', 'sum'),
        대규모오차율=('대규모오차', 'mean')
    )
    .reset_index()
)
operation_error_analysis['대규모오차율'] *= 100
print(operation_error_analysis.round(4))


# %% #24. 공장인원 조건별 오차 분석
result_analysis['인원상태'] = np.where(result_analysis['공장인원'] == 0, '공장인원=0', '공장인원>0')

worker_error_analysis = (
    result_analysis
    .groupby('인원상태')
    .agg(
        데이터수=('절대오차', 'size'),
        평균절대오차=('절대오차', 'mean'),
        대규모오차수=('대규모오차', 'sum'),
        대규모오차율=('대규모오차', 'mean')
    )
    .reset_index()
)
worker_error_analysis['대규모오차율'] *= 100
print(worker_error_analysis.round(4))


# %% #25. 생산조건 조합별 대규모 오차율
combined_error_analysis = (
    result_analysis
    .groupby(['operation_state', '생산량변화상태'])
    .agg(
        데이터수=('절대오차', 'size'),
        평균절대오차=('절대오차', 'mean'),
        대규모오차수=('대규모오차', 'sum'),
        대규모오차율=('대규모오차', 'mean')
    )
    .reset_index()
)
combined_error_analysis['대규모오차율'] *= 100

# 표본이 너무 적은 조합은 제외
combined_error_analysis = (
    combined_error_analysis[combined_error_analysis['데이터수'] >= 30]
    .sort_values('대규모오차율', ascending=False)
)
print(combined_error_analysis.round(4))


# %% #26. 대규모 오차의 과소예측 / 과대예측 분석
large_error_final = result_analysis[result_analysis['대규모오차']].copy()
large_error_final['오차방향'] = np.where(large_error_final['오차'] > 0, '과소예측', '과대예측')

error_direction_analysis = (
    large_error_final
    .groupby('오차방향')
    .agg(
        데이터수=('절대오차', 'size'),
        평균절대오차=('절대오차', 'mean'),
        평균실제전력=('실제전력', 'mean'),
        평균예측전력=('예측전력', 'mean')
    )
    .reset_index()
)
print(error_direction_analysis.round(4))


# %% #27. 월별 최대전력 피크 분석 — 분석용 시각 및 연월 생성
# 월별 최대전력 피크 = 각 월의 15분 관측값 중 최댓값 (동률이면 모두 포함)
# 보조 기준: 각 월 상위 5% = 고전력 위험구간
peak_df = df_feature4.copy()

# 시간=8, 분=60 같은 값도 09:00 으로 정상 변환됨
peak_df['기준일자'] = pd.to_datetime(peak_df['날짜'].astype(str), format='%Y%m%d')
peak_df['timestamp'] = (
    peak_df['기준일자']
    + pd.to_timedelta(peak_df['시간'], unit='h')
    + pd.to_timedelta(peak_df['분'], unit='m')
)
peak_df = peak_df.sort_values('timestamp').reset_index(drop=True)

# 월 구분은 원자료 날짜 기준 (예: 1월 31일 23시 60분은 1월 관측)
peak_df['연월'] = peak_df['기준일자'].dt.to_period('M').astype(str)
peak_df['실제시각'] = peak_df['timestamp'].dt.hour

print('분석 기간:', peak_df['timestamp'].min(), '~', peak_df['timestamp'].max())
print('전체 관측 수:', len(peak_df))
print('분석 월 수:', peak_df['연월'].nunique())
print('\n월별 관측 수:')
print(peak_df.groupby('연월').size().rename('관측수').reset_index())


# %% #28. 월별 최대전력 피크 정의
monthly_max_power = peak_df.groupby('연월')['전력사용량'].transform('max')
peak_df['월별최대피크여부'] = peak_df['전력사용량'] == monthly_max_power

monthly_peak = (
    peak_df.loc[peak_df['월별최대피크여부']]
    .copy()
    .sort_values(['연월', 'timestamp'])
    .reset_index(drop=True)
)

monthly_peak_summary = (
    monthly_peak
    .groupby('연월')
    .agg(
        월별최대전력=('전력사용량', 'max'),
        최대값발생횟수=('전력사용량', 'size'),
        최초발생시각=('timestamp', 'min'),
        최종발생시각=('timestamp', 'max')
    )
    .reset_index()
)

print('월 수:', peak_df['연월'].nunique())
print('월별 최대피크 발생 시점 수(동률 포함):', len(monthly_peak))
print('동률 피크가 존재하는 월 수:', (monthly_peak_summary['최대값발생횟수'] > 1).sum())
print(monthly_peak_summary)


# %% #29. 월별 최대전력 피크 발생 시점 상세
peak_detail_columns = [
    '연월', 'timestamp', '전력사용량', '생산량', 'production_change', 'operation_state',
    '공장인원', '인건비', '기온', '습도', 'lag_1_power', 'lag_4_power'
]
print(monthly_peak[peak_detail_columns])


# %% #30. 전체 전력 추이와 월별 최대전력 피크
plt.figure(figsize=(20, 7))
plt.plot(peak_df['timestamp'], peak_df['전력사용량'], linewidth=0.7, alpha=0.75, label='15분 전력사용량')
plt.scatter(monthly_peak['timestamp'], monthly_peak['전력사용량'],
            color='red', s=55, zorder=3, label='월별 최대전력 피크')
plt.xlabel('시간')
plt.ylabel('전력사용량')
plt.title('전체 전력사용량과 월별 최대전력 피크')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% #31. 월별 최대전력 변화
monthly_peak_value = (
    monthly_peak
    .groupby('연월', as_index=False)['전력사용량']
    .max()
    .rename(columns={'전력사용량': '월별최대전력'})
)

plt.figure(figsize=(12, 5))
plt.plot(monthly_peak_value['연월'], monthly_peak_value['월별최대전력'], marker='o', linewidth=2)
plt.xlabel('연월')
plt.ylabel('월별 최대전력')
plt.title('월별 최대전력 피크 변화')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(monthly_peak_value)


# %% #32. 피크 조건 분석을 위한 파생 범주 생성
def make_change_state(series):
    return np.select(
        [series < 0, series == 0, series > 0],
        ['생산량 감소', '생산량 변화없음', '생산량 증가'],
        default='기타'
    )


peak_df['생산량변화상태'] = make_change_state(peak_df['production_change'])
monthly_peak['생산량변화상태'] = make_change_state(monthly_peak['production_change'])

peak_df['인원상태'] = np.where(peak_df['공장인원'] > 0, '공장인원>0', '공장인원=0')
monthly_peak['인원상태'] = np.where(monthly_peak['공장인원'] > 0, '공장인원>0', '공장인원=0')

# 생산량=0은 별도 구분, 양수 생산량은 전체 데이터 기준 4분위 구간화
positive_production = peak_df.loc[peak_df['생산량'] > 0, '생산량']
_, production_bins = pd.qcut(positive_production, q=4, retbins=True, duplicates='drop')


def add_production_group(data):
    result = pd.Series('생산량=0', index=data.index, dtype='object')
    positive_mask = data['생산량'] > 0
    result.loc[positive_mask] = pd.cut(
        data.loc[positive_mask, '생산량'],
        bins=production_bins,
        include_lowest=True
    ).astype(str)
    return result


peak_df['생산량구간'] = add_production_group(peak_df)
monthly_peak['생산량구간'] = add_production_group(monthly_peak)

print('피크 조건 분석용 범주 생성 완료')


# %% #33. 범주별 월별 최대피크 집중도 계산 함수
def peak_condition_table(all_data, peak_data, column):
    overall_count = all_data[column].value_counts(dropna=False)
    peak_count = peak_data[column].value_counts(dropna=False)

    result = pd.DataFrame({
        '전체관측수': overall_count,
        '월별최대피크발생수': peak_count
    }).fillna(0)

    result['전체데이터비율(%)'] = result['전체관측수'] / len(all_data) * 100
    result['피크내비율(%)'] = result['월별최대피크발생수'] / len(peak_data) * 100
    result['피크집중도'] = result['피크내비율(%)'] / result['전체데이터비율(%)']
    result['조건별피크발생률(%)'] = result['월별최대피크발생수'] / result['전체관측수'] * 100

    return (
        result
        .reset_index(names=column)
        .sort_values('피크집중도', ascending=False)
    )


# %% #34. 월별 최대피크 발생 시간대 분석
peak_hour_analysis = peak_condition_table(peak_df, monthly_peak, '실제시각')
print(peak_hour_analysis.round(3))

plt.figure(figsize=(12, 5))
hour_count = (
    monthly_peak['실제시각']
    .value_counts()
    .reindex(range(24), fill_value=0)
    .sort_index()
)
plt.bar(hour_count.index, hour_count.values)
plt.xlabel('정규화된 실제 시각')
plt.ylabel('월별 최대피크 발생 횟수')
plt.title('시간대별 월별 최대전력 피크 발생 횟수')
plt.xticks(range(24))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# %% #35~#38. 생산량 구간 / 생산량 변화 / 생산상태 / 공장인원별 월별 최대피크 조건
monthly_peak_production = peak_condition_table(peak_df, monthly_peak, '생산량구간')
monthly_peak_change = peak_condition_table(peak_df, monthly_peak, '생산량변화상태')
monthly_peak_operation = peak_condition_table(peak_df, monthly_peak, 'operation_state')
monthly_peak_worker = peak_condition_table(peak_df, monthly_peak, '인원상태')

for title, table in [
    ('#35. 생산량 구간별', monthly_peak_production),
    ('#36. 생산량 변화별', monthly_peak_change),
    ('#37. 생산상태별', monthly_peak_operation),
    ('#38. 공장인원 조건별', monthly_peak_worker),
]:
    print(f'========== {title} 월별 최대피크 조건 ==========')
    print(table.round(3), '\n')


# %% #39. 전체 구간과 월별 최대피크의 주요 수치 비교
peak_numeric_features = [
    '전력사용량', '생산량', 'production_change', '공장인원', '인건비',
    '기온', '습도', 'lag_1_power', 'lag_4_power'
]

monthly_peak_numeric_summary = pd.DataFrame({
    '전체 평균': peak_df[peak_numeric_features].mean(),
    '월별 최대피크 평균': monthly_peak[peak_numeric_features].mean(),
    '전체 중앙값': peak_df[peak_numeric_features].median(),
    '월별 최대피크 중앙값': monthly_peak[peak_numeric_features].median()
})
monthly_peak_numeric_summary['피크/전체 평균비'] = (
    monthly_peak_numeric_summary['월별 최대피크 평균']
    / monthly_peak_numeric_summary['전체 평균']
)
print(monthly_peak_numeric_summary.round(4))


# %% #40. 생산상태와 생산량 변화 조합 분석
peak_df['운영조건조합'] = (
    peak_df['operation_state'].astype(str) + ' + ' + peak_df['생산량변화상태'].astype(str)
)
monthly_peak['운영조건조합'] = (
    monthly_peak['operation_state'].astype(str) + ' + ' + monthly_peak['생산량변화상태'].astype(str)
)

monthly_peak_combination = peak_condition_table(peak_df, monthly_peak, '운영조건조합')
print(monthly_peak_combination.round(3))


# %% #41. 월별 상위 5% 고전력 위험구간 정의 (보조 분석)
peak_df['월별95%기준'] = (
    peak_df.groupby('연월')['전력사용량'].transform(lambda x: x.quantile(0.95))
)
peak_df['고전력위험구간'] = peak_df['전력사용량'] >= peak_df['월별95%기준']

high_power_df = peak_df.loc[peak_df['고전력위험구간']].copy().reset_index(drop=True)

print('전체 관측 수:', len(peak_df))
print('고전력 위험구간 수:', len(high_power_df))
print('고전력 위험구간 비율:', f"{len(high_power_df) / len(peak_df) * 100:.2f}%")

print(
    peak_df
    .groupby('연월')
    .agg(
        월별95퍼센트기준=('월별95%기준', 'first'),
        월별최대전력=('전력사용량', 'max'),
        고전력관측수=('고전력위험구간', 'sum')
    )
    .reset_index()
    .round(3)
)


# %% #42. 고전력 위험구간의 조건별 집중도
high_power_hour = peak_condition_table(peak_df, high_power_df, '실제시각')
high_power_production = peak_condition_table(peak_df, high_power_df, '생산량구간')
high_power_change = peak_condition_table(peak_df, high_power_df, '생산량변화상태')
high_power_operation = peak_condition_table(peak_df, high_power_df, 'operation_state')
high_power_worker = peak_condition_table(peak_df, high_power_df, '인원상태')
high_power_combination = peak_condition_table(peak_df, high_power_df, '운영조건조합')

for title, table in [
    ('시간대별', high_power_hour),
    ('생산량 구간별', high_power_production),
    ('생산량 변화별', high_power_change),
    ('생산상태별', high_power_operation),
    ('공장인원별', high_power_worker),
    ('운영조건 조합별', high_power_combination),
]:
    print(f'========== {title} 고전력 위험조건 ==========')
    print(table.round(3), '\n')


# %% #43. 월별 최대피크 직전 1시간 흐름 분석
peak_windows = []

for peak_number, peak_row in monthly_peak.iterrows():
    start_time = peak_row['timestamp'] - pd.Timedelta(hours=1)
    end_time = peak_row['timestamp']

    window = peak_df.loc[
        peak_df['timestamp'].between(start_time, end_time),
        ['timestamp', '전력사용량', '생산량', 'production_change', 'operation_state']
    ].copy()

    window['피크연월'] = peak_row['연월']
    window['피크번호'] = peak_number + 1
    window['피크까지분'] = (
        (window['timestamp'] - end_time).dt.total_seconds().div(60).astype(int)
    )
    peak_windows.append(window)

monthly_peak_windows = pd.concat(peak_windows, ignore_index=True)

pre_peak_profile = (
    monthly_peak_windows
    .groupby('피크까지분')
    .agg(
        평균전력=('전력사용량', 'mean'),
        중앙전력=('전력사용량', 'median'),
        평균생산량=('생산량', 'mean'),
        관측수=('전력사용량', 'size')
    )
    .reset_index()
    .sort_values('피크까지분')
)
print(pre_peak_profile.round(3))

plt.figure(figsize=(10, 5))
plt.plot(pre_peak_profile['피크까지분'], pre_peak_profile['평균전력'], marker='o', label='평균 전력')
plt.axvline(0, color='red', linestyle='--', label='월별 최대피크')
plt.xlabel('피크 시점까지 남은 시간(분)')
plt.ylabel('평균 전력사용량')
plt.title('월별 최대전력 피크 발생 전 1시간의 평균 전력 흐름')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% #44. 핵심 피크 위험조건 자동 요약
def show_top_condition(table, condition_name, min_count=30):
    eligible = table.loc[table['전체관측수'] >= min_count]
    if eligible.empty:
        return f'{condition_name}: 표본 부족'

    top = eligible.sort_values('피크집중도', ascending=False).iloc[0]
    return (
        f"{condition_name}: {top.iloc[0]} | "
        f"고전력구간 내 비율 {top['피크내비율(%)']:.2f}% | "
        f"집중도 {top['피크집중도']:.2f}배 | "
        f"조건별 고전력 발생률 {top['조건별피크발생률(%)']:.2f}%"
    )


print('========== 고전력 위험조건 요약 ==========', '\n')
print(show_top_condition(high_power_hour, '시간대'))
print(show_top_condition(high_power_production, '생산량 구간'))
print(show_top_condition(high_power_change, '생산량 변화'))
print(show_top_condition(high_power_operation, '생산상태'))
print(show_top_condition(high_power_worker, '공장인원'))
print(show_top_condition(high_power_combination, '운영조건 조합'))

print('\n※ 집중도 1보다 크면 해당 조건이 전체 데이터에서 차지하는 비율보다')
print('   고전력 위험구간에서 차지하는 비율이 더 높다는 뜻입니다.')
print('※ 위 결과는 조건 간 연관성을 나타내며 직접적인 인과관계를 의미하지 않습니다.')


# %% #45. 월별 피크 분석 결과 저장
print('저장 완료:')
save_csv(monthly_peak, 'monthly_max_peak_result.csv')
save_csv(high_power_df, 'monthly_high_power_risk_result.csv')
save_csv(monthly_peak_windows, 'monthly_peak_previous_1hour.csv')


# %% #46. 고전력 경보 — 학습 구간에서 고전력 기준 고정
# 회귀모델은 MAE·RMSE·R², 고전력 경보는 Precision·Recall·F1 로 평가
# 기준은 Train 전력의 95% 분위수 (Test 정보로 임계값을 정하지 않음)
RISK_QUANTILE = 0.95
# 실제 관리 한계가 제공되면 수치를 입력. 없으면 Train 분위수 사용.
OPERATING_LIMIT = None
alarm_limit = (float(OPERATING_LIMIT) if OPERATING_LIMIT is not None
               else float(y_train.quantile(RISK_QUANTILE)))

alarm_result = test_df.copy().reset_index(drop=True)
alarm_result['timestamp'] = (
    pd.to_datetime(alarm_result['날짜'].astype(str), format='%Y%m%d')
    + pd.to_timedelta(alarm_result['시간'], unit='h')
    + pd.to_timedelta(alarm_result['분'], unit='m'))
alarm_result['실제시각'] = alarm_result['timestamp'].dt.hour
alarm_result['실제전력'] = np.asarray(y_test)
alarm_result['예측전력'] = np.asarray(feature4_test_pred)
alarm_result['실제고전력'] = (alarm_result['실제전력'] >= alarm_limit).astype(int)
alarm_result['예측고전력'] = (alarm_result['예측전력'] >= alarm_limit).astype(int)
alarm_result['오차'] = alarm_result['실제전력'] - alarm_result['예측전력']
alarm_result['절대오차'] = alarm_result['오차'].abs()
truth = alarm_result['실제고전력'].to_numpy()

print(f'고전력 기준: {alarm_limit:.4f} (Train에서 고정)')
print('Test 음성/양성 수:', np.bincount(truth, minlength=2))
print('동률 때문에 양성비율이 정확히 5%일 필요는 없습니다.')


# %% #47. 동일한 정답·임계값으로 3가지 방법 비교
def alarm_metrics(y_true, predictions, limit):
    labels = (np.asarray(predictions) >= limit).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    return {'Precision': precision_score(y_true, labels, zero_division=0),
            'Recall': recall_score(y_true, labels, zero_division=0),
            'F1': f1_score(y_true, labels, zero_division=0),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


alarm_model_comparison = pd.DataFrame([
    {'방법': name, **alarm_metrics(truth, pred, alarm_limit),
     **regression_metrics(y_test, pred)}
    for name, pred in [('Baseline XGBoost', baseline_test_pred),
                       ('Persistence', persistence_pred),
                       ('Feature4 XGBoost', feature4_test_pred)]])
print(alarm_model_comparison.round(4))

ConfusionMatrixDisplay.from_predictions(
    truth, alarm_result['예측고전력'], labels=[0, 1],
    display_labels=['일반구간', '고전력'], cmap='Blues', colorbar=False)
plt.title('Feature4 고전력 경보 혼동행렬 (행: 실제 / 열: 예측)')
plt.tight_layout()
plt.show()

if truth.sum() == 0:
    print('이 Test에는 고전력 양성이 없어 F1=0을 모델 실패로 해석하면 안 됩니다.')


# %% #48. FP/FN 행 추출 및 Train 기준 생산량 구간화
actual_positive = alarm_result['실제고전력'].eq(1)
predicted_positive = alarm_result['예측고전력'].eq(1)
alarm_result['판정'] = np.select(
    [actual_positive & predicted_positive,
     ~actual_positive & predicted_positive,
     actual_positive & ~predicted_positive], ['TP', 'FP', 'FN'], default='TN')
alarm_result['FP'] = alarm_result['판정'].eq('FP').astype(int)
alarm_result['FN'] = alarm_result['판정'].eq('FN').astype(int)
alarm_result['생산량변화상태'] = np.select(
    [alarm_result['production_change'] < 0,
     alarm_result['production_change'] == 0,
     alarm_result['production_change'] > 0],
    ['감소', '유지', '증가'], default='결측')
alarm_result['인원상태'] = np.where(alarm_result['공장인원'] > 0, '인원>0', '인원=0')

train_positive_production = train_df.loc[train_df['생산량'] > 0, '생산량']
if len(train_positive_production):
    internal_edges = np.unique(train_positive_production.quantile([.25, .5, .75]).to_numpy())
else:
    internal_edges = np.array([])
alarm_production_bins = np.r_[-np.inf, internal_edges, np.inf]
alarm_result['생산량구간'] = '생산량=0'
positive_prod_mask = alarm_result['생산량'] > 0
alarm_result.loc[positive_prod_mask, '생산량구간'] = pd.cut(
    alarm_result.loc[positive_prod_mask, '생산량'],
    bins=alarm_production_bins).astype(str)

print('생산량 구간 내부 경계 (Train에서 산출):', internal_edges)
print('전체 판정:', alarm_result['판정'].value_counts().to_dict())

error_detail_columns = ['timestamp', '실제전력', '예측전력', '오차', '판정',
                        '생산량', 'production_change', 'operation_state',
                        '공장인원', 'lag_1_power', 'lag_4_power']
for error_type in ['FN', 'FP']:
    print(f'\n{error_type} 사례 (절대오차 큰 순서, 최대 20건)')
    print(alarm_result.loc[alarm_result['판정'].eq(error_type)]
          .sort_values('절대오차', ascending=False)[error_detail_columns].head(20))


# %% #49. 공정조건별 오류 수 + 적절한 분모의 오류율
# FN율 = FN / 실제고전력수, FP율 = FP / 실제일반수
MIN_CONDITION_ROWS = 30
MIN_CLASS_ROWS = 10


def error_condition_table(data, columns):
    keys = [columns] if isinstance(columns, str) else list(columns)
    rows = []
    base_fn_rate = data['FN'].sum() / data['실제고전력'].sum() if data['실제고전력'].sum() else np.nan
    base_fp_rate = data['FP'].sum() / data['실제고전력'].eq(0).sum() if data['실제고전력'].eq(0).sum() else np.nan
    for key, group in data.groupby(keys, dropna=False, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        n = len(group)
        pos = int(group['실제고전력'].sum())
        neg = n - pos
        fn = int(group['FN'].sum())
        fp = int(group['FP'].sum())
        fn_rate = fn / pos if pos else np.nan
        fp_rate = fp / neg if neg else np.nan
        rows.append({**dict(zip(keys, key)), '관측수': n, '실제고전력수': pos,
                     '실제일반수': neg, 'FN수': fn, 'FP수': fp,
                     'FN율(%)': fn_rate * 100, 'FP율(%)': fp_rate * 100,
                     'FN전체대비율': fn_rate / base_fn_rate if base_fn_rate > 0 else np.nan,
                     'FP전체대비율': fp_rate / base_fp_rate if base_fp_rate > 0 else np.nan,
                     'FN해석표본충족': n >= MIN_CONDITION_ROWS and pos >= MIN_CLASS_ROWS,
                     'FP해석표본충족': n >= MIN_CONDITION_ROWS and neg >= MIN_CLASS_ROWS,
                     'MAE': group['절대오차'].mean()})
    return pd.DataFrame(rows)


condition_error_tables = {}
for column in ['실제시각', '생산량구간', '생산량변화상태', 'operation_state', '인원상태']:
    table = error_condition_table(alarm_result, column)
    condition_error_tables[column] = table
    print(f'\n조건: {column}')
    print(table.sort_values(['FN율(%)', 'FP율(%)'], ascending=False).round(4))
print('\nNaN 오류율: 해당 조건에 실제 양성/음성이 없어 분모가 0인 경우')
print('표본충족은 간단한 해석 가이드이며 통계적 유의성 검정이 아닙니다.')


# %% #50. 조건 조합별 FP/FN: 상호작용의 운영상 해석
combination_error_tables = {}
for columns in [('실제시각', '생산량구간'),
                ('operation_state', '생산량변화상태'),
                ('실제시각', 'operation_state')]:
    name = ' + '.join(columns)
    table = error_condition_table(alarm_result, columns)
    combination_error_tables[name] = table
    print(f'\n========== {name} ==========')
    print(table.sort_values('FN수', ascending=False).round(4).head(20))
    for flag, metric in [('FN해석표본충족', 'FN율(%)'), ('FP해석표본충족', 'FP율(%)')]:
        print(f'\n표본조건 충족 그룹의 {metric} 상위 5개')
        print(table.loc[table[flag]].sort_values(metric, ascending=False).head(5).round(4))
print('\n조합별 오류율은 연관성 분석입니다. 설비별 원인 또는 인과관계를 입증하지 않습니다.')


# %% #51. XGBoost 내장 TreeSHAP: 주요 영향변수와 방향
# 별도의 shap 패키지가 필요하지 않음. 예측 기여도 해석이며 인과가 아님.
SHAP_SAMPLE_SIZE = 500
sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_feature4))
sample_positions = np.sort(np.random.default_rng(42).choice(
    len(X_test_feature4), size=sample_size, replace=False))
shap_X = X_test_feature4.iloc[sample_positions].copy()
booster = feature4_model.get_booster()
shap_matrix = booster.predict(DMatrix(shap_X), pred_contribs=True)
shap_values = shap_matrix[:, :-1]
shap_bias = shap_matrix[:, -1]
sample_predictions = feature4_model.predict(shap_X)
assert np.allclose(shap_values.sum(axis=1) + shap_bias,
                   sample_predictions, atol=1e-3, rtol=1e-4)

shap_importance = pd.DataFrame({
    '변수': shap_X.columns,
    '평균절대SHAP': np.abs(shap_values).mean(axis=0),
    '평균SHAP': shap_values.mean(axis=0)
}).sort_values('평균절대SHAP', ascending=False)
print(shap_importance.round(4))

plt.figure(figsize=(10, 7))
plot_imp = shap_importance.head(15).sort_values('평균절대SHAP')
plt.barh(plot_imp['변수'], plot_imp['평균절대SHAP'])
plt.xlabel('평균 |SHAP| (전력 예측값 단위)')
plt.title(f'Test 무작위 {sample_size}건의 주요 영향변수')
plt.tight_layout()
plt.show()

top_names = shap_importance['변수'].head(3).tolist()
fig, axes = plt.subplots(1, len(top_names), figsize=(6 * len(top_names), 4))
for ax, name in zip(np.atleast_1d(axes), top_names):
    j = shap_X.columns.get_loc(name)
    ax.scatter(shap_X[name], shap_values[:, j], alpha=.4, s=15)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xlabel(name)
    ax.set_ylabel('SHAP 기여도')
plt.tight_layout()
plt.show()

print('양의 SHAP: 기준 예측에서 전력을 높임 / 음의 SHAP: 낮춤')
print('평균SHAP은 양·음이 상쇄됩니다. 중요도 순위는 평균절대SHAP을 봅니다.')


# %% #52. TreeSHAP 상호작용: 변수 A의 영향이 B에 따라 달라지는 조합
INTERACTION_SAMPLE_SIZE = 100  # 계산시간이 길면 50으로 줄일 수 있음
interaction_X = shap_X.sample(n=min(INTERACTION_SAMPLE_SIZE, len(shap_X)), random_state=42)
interaction_values = booster.predict(DMatrix(interaction_X), pred_interactions=True)
mean_abs_interaction = np.abs(interaction_values[:, :-1, :-1]).mean(axis=0)

pair_rows = []
names = interaction_X.columns.tolist()
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        # 대칭인 두 비대각항을 합친 쌍 기여도 규모
        pair_rows.append({'변수A': names[i], '변수B': names[j],
                          '평균절대쌍상호작용': 2 * mean_abs_interaction[i, j]})
shap_interaction_table = pd.DataFrame(pair_rows).sort_values(
    '평균절대쌍상호작용', ascending=False).reset_index(drop=True)
print(shap_interaction_table.head(15).round(4))

best_pair = shap_interaction_table.iloc[0]
a, b = best_pair['변수A'], best_pair['변수B']
ia, ib = names.index(a), names.index(b)
fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(interaction_X[a], 2 * interaction_values[:, ia, ib],
                c=interaction_X[b], cmap='viridis', s=25, alpha=.7)
fig.colorbar(sc, ax=ax, label=b)
ax.axhline(0, color='black', linewidth=.8)
ax.set_xlabel(a)
ax.set_ylabel('쌍 SHAP 상호작용 기여도')
ax.set_title(f'{a} × {b}: 모델이 학습한 비가산적 기여')
plt.tight_layout()
plt.show()
print('대각항(주효과)은 제외. 상호작용 크기는 상관계수나 인과효과가 아닙니다.')


# %% #53. FP/FN 사례에서 모델이 어떤 변수를 활용했는지
error_shap_profiles = []
for kind in ['TP', 'FN', 'FP', 'TN']:
    positions = np.flatnonzero(alarm_result['판정'].eq(kind).to_numpy())
    if not len(positions):
        print(kind, '사례가 없습니다.')
        continue
    # 계산량 제한을 위해 최대 200건 무작위 추출
    positions = np.sort(np.random.default_rng(42).choice(
        positions, size=min(200, len(positions)), replace=False))
    values = booster.predict(DMatrix(X_test_feature4.iloc[positions]),
                             pred_contribs=True)[:, :-1]
    profile = pd.DataFrame({'판정': kind, '변수': X_test_feature4.columns,
                            '분석표본수': len(positions),
                            '평균SHAP': values.mean(axis=0),
                            '평균절대SHAP': np.abs(values).mean(axis=0)})
    error_shap_profiles.append(profile)
    print(f'\n{kind} 예측 기여도 상위 변수')
    print(profile.sort_values('평균절대SHAP', ascending=False).head(8).round(4))

error_shap_profile = (pd.concat(error_shap_profiles, ignore_index=True)
                      if error_shap_profiles else pd.DataFrame())
print('\nFN/FP SHAP는 예측 형성을 설명합니다. 실제 오차의 원인을 확정하지 않습니다.')


# %% #54. 공식 월별 최대피크의 경보 누락 여부 (고전력 F1과 별개)
peak_reference = df_feature4.copy()
peak_reference['timestamp'] = (
    pd.to_datetime(peak_reference['날짜'].astype(str), format='%Y%m%d')
    + pd.to_timedelta(peak_reference['시간'], unit='h')
    + pd.to_timedelta(peak_reference['분'], unit='m'))
peak_reference['연월'] = pd.to_datetime(
    peak_reference['날짜'].astype(str), format='%Y%m%d').dt.to_period('M').astype(str)
peak_reference['월별최대피크여부'] = peak_reference['전력사용량'].eq(
    peak_reference.groupby('연월')['전력사용량'].transform('max'))
if peak_reference['timestamp'].duplicated().any():
    raise ValueError('timestamp 중복: 조인 전 중복 관측을 확인하세요.')

monthly_peak_alarm = alarm_result.merge(
    peak_reference.loc[peak_reference['월별최대피크여부'], ['timestamp', '연월']],
    on='timestamp', how='inner', validate='one_to_one')

print('Test에 포함된 공식 월별 최대피크의 경보 결과')
print(monthly_peak_alarm[['연월'] + error_detail_columns])
print('고전력 기준이 월별 최대값보다 높으면 공식 피크라도 실제고전력=0일 수 있습니다.')
print('관측이 불완전한 월의 최대값은 제공된 관측 범위의 최대값입니다.')


# %% #55. 분석 결과 저장 (./outputs)
analysis_exports = {
    'peak_alarm_model_comparison.csv': alarm_model_comparison,
    'peak_alarm_test_results.csv': alarm_result,
    'peak_alarm_FN_cases.csv': alarm_result.loc[alarm_result['판정'].eq('FN')],
    'peak_alarm_FP_cases.csv': alarm_result.loc[alarm_result['판정'].eq('FP')],
    'peak_alarm_condition_errors.csv': pd.concat(
        [table.assign(분석조건=name) for name, table in condition_error_tables.items()],
        ignore_index=True),
    'peak_alarm_combination_errors.csv': pd.concat(
        [table.assign(분석조건=name) for name, table in combination_error_tables.items()],
        ignore_index=True),
    'power_SHAP_importance.csv': shap_importance,
    'power_SHAP_interactions.csv': shap_interaction_table,
    'peak_alarm_error_SHAP.csv': error_shap_profile,
    'monthly_max_peak_alarm_results.csv': monthly_peak_alarm,
}
print('저장 완료:')
for filename, table in analysis_exports.items():
    save_csv(table, filename)

print('\n모든 단계 실행 완료')
