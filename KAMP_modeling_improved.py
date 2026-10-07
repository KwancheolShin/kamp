# -*- coding: utf-8 -*-
"""
KAMP 전력사용량 예측 — 개선 버전 (Spyder 실행용)

원본(KAMP_modeling_3_7.py)의 4가지 한계를 보완한다.

  1) 예측 구간이 15분으로 고정  → 15분 / 1시간 / 24시간 앞 예측을 각각 학습·평가
                                   (예측 시점에 알 수 있는 lag 만 사용, Persistence 와 비교)
  2) 시간 내 미래 정보 누수      → 두 가지 시나리오로 분리
       - 계획기반: 생산량·공장인원 = 사전 생산계획, 날씨 = 예보로 간주 (가정을 명시)
       - 실적기반: 예측 시점에 이미 끝난 시간의 생산 실적과 관측 날씨만 사용 (누수 없음)
  3) 피크 과소예측               → 고전력 구간 학습 가중치 + 검증 구간에서 경보 임계값 선택
                                   (Test 는 임계값 선택에 쓰지 않음)
  4) 평일 휴무 미반영            → 가동일 달력 변수 추가 + 생산기록 누락일 보정

실행: Power15MinClean_NoDatetime.csv 와 같은 폴더에 두고 F5.
결과: ./outputs/improved_*.csv, 그림은 Plots 창.
필요 패키지: pandas, numpy, matplotlib, scikit-learn, xgboost
"""

# %% #0. 환경 설정
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score
)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DATA_FILE = os.path.join(BASE_DIR, 'Power15MinClean_NoDatetime.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

available_fonts = {f.name for f in fm.fontManager.ttflist}
for font_name in ['Malgun Gothic', 'NanumGothic', 'AppleGothic']:
    if font_name in available_fonts:
        plt.rcParams['font.family'] = font_name
        break
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)
pd.set_option('display.unicode.east_asian_width', True)

# ------------------------------------------------------------
# 실험 설정
# ------------------------------------------------------------
HORIZONS = {1: '15분 앞', 4: '1시간 앞', 96: '24시간 앞'}   # 15분 간격 기준 칸 수
SCENARIOS = ['실적기반', '계획기반']

RISK_QUANTILE = 0.95      # 고전력 기준: Train 전력의 95% 분위수 (원본과 동일)
PEAK_WEIGHT = 3.0         # 고전력 구간 학습 가중치 (1.0 이면 가중치 없음)
VALID_RATIO = 0.2         # Train 뒤쪽 20% 를 경보 임계값 선택용 검증 구간으로 사용
MIN_HISTORY = 800         # lag_672(1주) + 여유: 앞쪽 800행은 lag 가 비어 학습에서 제외

# 생산 기록 누락으로 판단한 날짜 (#2 데이터 점검 참고)
PROD_MISSING_DAYS = [20210713, 20210715]

# 가동일 달력: 해당일 평균 전력이 이 값 미만이면 휴무일로 간주
# ※ 실제 운영에서는 '사전에 확정된 휴무 계획(여름휴가, 공휴일 등)'으로 대체해야 함
OPERATING_DAY_THRESHOLD = 30


def make_xgb_model():
    # 원본과 동일한 하이퍼파라미터
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
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }


def alarm_metrics(y_true, y_pred, limit, threshold):
    # 정답: 실제 >= limit, 경보: 예측 >= threshold
    truth = np.asarray(y_true) >= limit
    alarm = np.asarray(y_pred) >= threshold
    return {
        'Precision': precision_score(truth, alarm, zero_division=0),
        'Recall': recall_score(truth, alarm, zero_division=0),
        'F1': f1_score(truth, alarm, zero_division=0)
    }


def save_csv(df, file_name):
    df.to_csv(os.path.join(OUTPUT_DIR, file_name), index=False, encoding='utf-8-sig')
    print('-', file_name, df.shape)


print("환경 설정 완료 / 폰트:", plt.rcParams['font.family'])


# %% #1. 데이터 불러오기 및 정렬
df = pd.read_csv(DATA_FILE)
df = df.sort_values(['날짜', '시간', '분']).reset_index(drop=True)

# 분=60 은 다음 정각 (예: 시간 8, 분 60 → 09:00)
df['timestamp'] = (
    pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d')
    + pd.to_timedelta(df['시간'], unit='h')
    + pd.to_timedelta(df['분'], unit='m')
)

power = df['전력사용량']
print("데이터 크기:", df.shape)
print("기간:", df['timestamp'].min(), "~", df['timestamp'].max())


# %% #2. 데이터 점검: 생산 기록 누락일
# 생산량이 하루 종일 0인데 전력은 평소 가동일 수준인 날을 찾는다.
daily = df.groupby('날짜').agg(
    일평균전력=('전력사용량', 'mean'),
    일생산량=('생산량', 'sum'),
    일공장인원=('공장인원', 'sum'),
    요일=('day', 'first')
)
suspect = daily[(daily['일생산량'] == 0) & (daily['일평균전력'] >= 100)]
print("생산량 0 인데 전력이 높은 날 (생산 기록 누락 또는 무생산 가동 의심):")
print(suspect.round(1))

print("\n이 중 PROD_MISSING_DAYS 로 보정할 날짜:", PROD_MISSING_DAYS)
print("  → okm_augumented_2021.csv 에서도 이 두 날짜의 '시간' 값이 깨져 있어 기록 오류로 판단")
print("  → 해당일 생산량·공장인원을 결측(NaN)으로 바꾸고 XGBoost 의 결측 처리에 맡긴다.")

missing_mask = df['날짜'].isin(PROD_MISSING_DAYS)
df.loc[missing_mask, ['생산량', '공장인원']] = np.nan


# %% #3. 공통 파생변수
# (1) 생산 관련 (계획기반 시나리오에서 '같은 시각' 값으로 사용)
prev_production = df['생산량'].shift(1)
df['production_change'] = df['생산량'] - prev_production

# 생산상태: 0 비생산지속 / 1 생산시작 / 2 생산종료전환 / 3 생산지속 / NaN 판단불가
state = np.select(
    [(prev_production == 0) & (df['생산량'] == 0),
     (prev_production == 0) & (df['생산량'] > 0),
     (prev_production > 0) & (df['생산량'] == 0),
     (prev_production > 0) & (df['생산량'] > 0)],
    [0, 1, 2, 3],
    default=-1
).astype(float)
state[state < 0] = np.nan
df['생산상태'] = state

# 일 생산계획: 하루 총생산량 (누락일은 NaN)
df['일생산계획'] = df.groupby('날짜')['생산량'].transform('sum')
df.loc[missing_mask, '일생산계획'] = np.nan

# (2) 가동일 달력 (사전 휴무 계획의 대리 변수)
day_mean_power = df.groupby('날짜')['전력사용량'].transform('mean')
df['가동일'] = (day_mean_power >= OPERATING_DAY_THRESHOLD).astype(int)

print("가동일 / 휴무일 일수:", df.groupby('날짜')['가동일'].first().value_counts().to_dict())
weekday_off = daily[(df.groupby('날짜')['가동일'].first() == 0) & (daily['요일'] <= 5)]
print("평일 휴무일:", weekday_off.index.tolist())

# (3) 실적기반 시나리오용: '예측 시점에 이미 끝난 시간'의 생산 실적
# 시간 H 의 생산량은 분=60 행(H+1 시 정각)에 확정된다.
# 그 다음 3행(15/30/45분)까지는 직전 완료 시간의 값을 그대로 사용한다.
hour_done = df['분'].eq(60)
df['완료생산실적'] = df['생산량'].where(hour_done).ffill(limit=3)
df['완료공장인원'] = df['공장인원'].where(hour_done).ffill(limit=3)

CALENDAR_FEATURES = ['시간', '분', 'day', 'd', 'm', '전기요금(계절)', '인건비', '가동일']
WEATHER = ['기온', '습도', '풍속', '강수량']


# %% #4. 예측 구간·시나리오별 입력 생성 함수
def build_features(h, scenario, use_calendar=True):
    """
    h        : 몇 칸(15분 단위) 앞을 예측하는가
    scenario : '실적기반' 또는 '계획기반'

    모든 lag 는 h 이상 떨어진 과거 값만 사용한다
    → 예측 시점(t-h)에 실제로 알 수 있는 전력 정보만 입력된다.
    """
    cal = CALENDAR_FEATURES if use_calendar else [c for c in CALENDAR_FEATURES if c != '가동일']
    X = df[cal].copy()

    # 전력 lag: 직전 값, 그 직전, 1시간 전, 하루 전·1주 전 같은 시각
    lags = [h, h + 1, h + 4, 672]
    if h < 96:
        lags.append(96)
    for lag in sorted(set(lags)):
        X[f'lag_{lag}'] = power.shift(lag)

    # 예측 시점까지의 최근 1시간 / 24시간 평균 전력
    X['최근1시간평균'] = power.shift(h).rolling(4).mean()
    X['최근24시간평균'] = power.shift(h).rolling(96).mean()

    if scenario == '계획기반':
        # 가정: 생산량·공장인원은 사전 생산계획, 날씨는 예보값으로 미리 알 수 있다
        for c in ['생산량', 'production_change', '생산상태', '공장인원', '일생산계획'] + WEATHER:
            X[c] = df[c]
    elif scenario == '실적기반':
        # 예측 시점(t-h)에 이미 확정된 실적·관측값만 사용 (누수 없음)
        X['완료생산실적'] = df['완료생산실적'].shift(h)
        X['완료공장인원'] = df['완료공장인원'].shift(h)
        for c in WEATHER:
            X[c] = df[c].shift(h)
    else:
        raise ValueError(scenario)

    return X


# %% #5. Train / Validation / Test 분할
# Test 구간은 원본 노트북과 동일 (원본 df_exp 기준 8:2 분할 시점)
TEST_START = df['timestamp'].iloc[int((len(df) - 4) * 0.8) + 4]

row_no = np.arange(len(df))
is_test = (df['timestamp'] >= TEST_START).to_numpy()
is_train_all = (~is_test) & (row_no >= MIN_HISTORY)

# Train 뒤쪽 VALID_RATIO 를 검증 구간으로 (경보 임계값 선택에만 사용)
train_rows = np.flatnonzero(is_train_all)
valid_start = train_rows[int(len(train_rows) * (1 - VALID_RATIO))]
is_train_inner = is_train_all & (row_no < valid_start)
is_valid = is_train_all & (row_no >= valid_start)

alarm_limit = float(power[is_train_all].quantile(RISK_QUANTILE))
y_test = power[is_test]

print("Train 전체 :", is_train_all.sum(), "행",
      df['timestamp'][is_train_all].min(), "~", df['timestamp'][is_train_all].max())
print("  └ 학습용 :", is_train_inner.sum(), "행")
print("  └ 검증용 :", is_valid.sum(), "행",
      df['timestamp'][is_valid].min(), "~", df['timestamp'][is_valid].max())
print("Test       :", is_test.sum(), "행", TEST_START, "~", df['timestamp'].max())
print(f"고전력 기준(Train 95% 분위수): {alarm_limit:.1f}")


# %% #6. 학습·평가 함수
def sample_weights(y):
    return np.where(np.asarray(y) >= alarm_limit, PEAK_WEIGHT, 1.0)


def choose_alarm_threshold(X):
    """
    검증 구간에서 경보 임계값을 고른다 (Test 는 보지 않음).
    학습용 구간으로 학습 → 검증 구간 예측 → F1 이 최대가 되는 임계값.
    """
    model = make_xgb_model()
    model.fit(X[is_train_inner], power[is_train_inner],
              sample_weight=sample_weights(power[is_train_inner]))
    pred_valid = model.predict(X[is_valid])
    candidates = np.arange(alarm_limit - 30, alarm_limit + 5.5, 1.0)
    scores = [alarm_metrics(power[is_valid], pred_valid, alarm_limit, c)['F1'] for c in candidates]
    return float(candidates[int(np.argmax(scores))])


def evaluate(name, X, h, scenario):
    threshold = choose_alarm_threshold(X)

    model = make_xgb_model()
    model.fit(X[is_train_all], power[is_train_all],
              sample_weight=sample_weights(power[is_train_all]))
    pred = model.predict(X[is_test])

    reg = regression_metrics(y_test, pred)
    pers = regression_metrics(y_test, power.shift(h)[is_test])
    fixed = alarm_metrics(y_test, pred, alarm_limit, alarm_limit)
    tuned = alarm_metrics(y_test, pred, alarm_limit, threshold)

    row = {
        '모델': name, '예측구간': HORIZONS.get(h, f'{h}칸 앞'), 'h': h, '시나리오': scenario,
        'MAE': reg['MAE'], 'RMSE': reg['RMSE'], 'R2': reg['R2'],
        'Persistence MAE': pers['MAE'], 'Persistence R2': pers['R2'],
        'MAE 개선(vs Persistence)': pers['MAE'] - reg['MAE'],
        '경보(178고정) Precision': fixed['Precision'],
        '경보(178고정) Recall': fixed['Recall'],
        '경보(178고정) F1': fixed['F1'],
        '검증선택 임계값': threshold,
        '경보(검증선택) Precision': tuned['Precision'],
        '경보(검증선택) Recall': tuned['Recall'],
        '경보(검증선택) F1': tuned['F1'],
    }
    return row, model, pred


# %% #7. 원본 Feature4 재현 (같은 Train/Test 행에서 비교 기준으로 사용)
original_features = ['시간', '분', '생산량', '기온', '풍속', '습도', '강수량',
                     '전기요금(계절)', 'day', 'd', 'm', '공장인원', '인건비']
raw = pd.read_csv(DATA_FILE).sort_values(['날짜', '시간', '분']).reset_index(drop=True)
raw_prev = raw['생산량'].shift(1)
X_original = raw[original_features].copy()
X_original['lag_1_power'] = raw['전력사용량'].shift(1)
X_original['lag_4_power'] = raw['전력사용량'].shift(4)
X_original['production_change'] = raw['생산량'] - raw_prev
X_original['생산상태'] = np.select(
    [(raw_prev == 0) & (raw['생산량'] == 0), (raw_prev == 0) & (raw['생산량'] > 0),
     (raw_prev > 0) & (raw['생산량'] == 0), (raw_prev > 0) & (raw['생산량'] > 0)],
    [0, 1, 2, 3], default=-1)

original_model = make_xgb_model()
original_model.fit(X_original[is_train_all], power[is_train_all])
original_pred = original_model.predict(X_original[is_test])
original_row = {
    '모델': '원본 Feature4', '예측구간': '15분 앞', 'h': 1, '시나리오': '원본(누수 포함)',
    **regression_metrics(y_test, original_pred),
    'Persistence MAE': regression_metrics(y_test, power.shift(1)[is_test])['MAE'],
    'Persistence R2': regression_metrics(y_test, power.shift(1)[is_test])['R2'],
}
original_row['MAE 개선(vs Persistence)'] = original_row['Persistence MAE'] - original_row['MAE']
fixed = alarm_metrics(y_test, original_pred, alarm_limit, alarm_limit)
original_row.update({f'경보(178고정) {k}': v for k, v in fixed.items()})
print("원본 Feature4 재현 완료:", {k: round(original_row[k], 4) for k in ['MAE', 'R2']})


# %% #8. 예측 구간 × 시나리오 전체 학습 (수 분 소요)
results = [original_row]
models = {}
predictions = {}

for h in HORIZONS:
    for scenario in SCENARIOS:
        name = f'개선 {HORIZONS[h]} ({scenario})'
        X = build_features(h, scenario)
        row, model, pred = evaluate(name, X, h, scenario)
        results.append(row)
        models[(h, scenario)] = (model, X.columns.tolist())
        predictions[(h, scenario)] = pred
        print(f"{name:28s} MAE {row['MAE']:6.2f}  R2 {row['R2']:.4f}  "
              f"| 경보 F1 고정 {row['경보(178고정) F1']:.3f} → 검증선택 {row['경보(검증선택) F1']:.3f}")

results_df = pd.DataFrame(results)


# %% #9. 결과표
show_cols = ['모델', 'MAE', 'R2', 'Persistence MAE', 'Persistence R2', 'MAE 개선(vs Persistence)']
print("========== 회귀 성능 (Test) ==========")
print(results_df[show_cols].round(4).to_string(index=False))

alarm_cols = ['모델', '경보(178고정) Precision', '경보(178고정) Recall', '경보(178고정) F1',
              '검증선택 임계값', '경보(검증선택) Precision', '경보(검증선택) Recall', '경보(검증선택) F1']
print("\n========== 고전력 경보 성능 (Test, 정답 기준 = 실제 >= {:.0f}) ==========".format(alarm_limit))
print(results_df[alarm_cols].round(3).to_string(index=False))

print("""
읽는 법
- 실적기반: 예측 시점에 이미 끝난 정보만 사용 → 실제 운영에서 그대로 가능한 성능
- 계획기반: 생산계획·날씨예보를 안다는 가정 → 계획 정보의 가치를 보여줌
- 두 시나리오의 차이 = '생산계획을 미리 알면 얼마나 더 잘 맞히는가'
- 검증선택 임계값: Train 뒤쪽 검증 구간에서만 고른 값. Test 로 고르지 않았다.
""")


# %% #10. 개선 요소별 기여 확인 (Ablation) — 24시간 앞 / 계획기반
# 가동일 달력과 고전력 가중치를 하나씩 빼고 다시 학습해 각 요소의 효과를 본다.
ablation_rows = []
h_ab = 96

X_full = build_features(h_ab, '계획기반')
X_no_cal = build_features(h_ab, '계획기반', use_calendar=False)

for label, X_ab, weight in [
    ('전체 (가동일 + 가중치)', X_full, PEAK_WEIGHT),
    ('가동일 달력 제외', X_no_cal, PEAK_WEIGHT),
    ('고전력 가중치 제외', X_full, 1.0),
]:
    saved_weight = PEAK_WEIGHT
    PEAK_WEIGHT = weight
    row, _, _ = evaluate(label, X_ab, h_ab, '계획기반')
    PEAK_WEIGHT = saved_weight
    ablation_rows.append(row)

ablation_df = pd.DataFrame(ablation_rows)
print("========== 24시간 앞(계획기반) 개선 요소별 효과 ==========")
print(ablation_df[['모델', 'MAE', 'R2', '경보(178고정) Recall', '경보(178고정) F1',
                   '경보(검증선택) F1']].round(4).to_string(index=False))


# %% #11. 그림 1 — 예측 구간별 R²: Persistence vs 개선 모델
fig, ax = plt.subplots(figsize=(10, 5))
labels = list(HORIZONS.values())
x = np.arange(len(labels))
width = 0.26

pers_r2 = [results_df.query('h == @h and 시나리오 == "실적기반"')['Persistence R2'].iloc[0] for h in HORIZONS]
strict_r2 = [results_df.query('h == @h and 시나리오 == "실적기반"')['R2'].iloc[0] for h in HORIZONS]
plan_r2 = [results_df.query('h == @h and 시나리오 == "계획기반"')['R2'].iloc[0] for h in HORIZONS]

for offset, values, label, color in [
    (-width, pers_r2, 'Persistence (과거 값 복사)', '#b4b2a9'),
    (0, strict_r2, '개선 모델 – 실적기반', '#8fb8e8'),
    (width, plan_r2, '개선 모델 – 계획기반', '#2a78d6'),
]:
    bars = ax.bar(x + offset, values, width, label=label, color=color)
    ax.bar_label(bars, fmt='%.2f', fontsize=9)

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Test R²')
ax.set_title('예측 구간별 성능: 과거 값 복사 vs 개선 모델')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# %% #12. 그림 2 — 휴무 주간 전후: 24시간 앞 예측 (가동일 달력 효과)
window = (df['timestamp'] >= '2021-07-26') & (df['timestamp'] < '2021-08-16')
test_ts = df['timestamp'][is_test]
in_window = window[is_test].to_numpy()

model_nocal = make_xgb_model()
model_nocal.fit(X_no_cal[is_train_all], power[is_train_all],
                sample_weight=sample_weights(power[is_train_all]))
pred_nocal = model_nocal.predict(X_no_cal[is_test])

plt.figure(figsize=(20, 6))
plt.plot(test_ts[in_window], y_test[in_window], label='실제 전력', linewidth=1)
plt.plot(test_ts[in_window], pred_nocal[in_window], label='24시간 앞 예측 (가동일 달력 없음)',
         linewidth=1, alpha=0.7, linestyle='--')
plt.plot(test_ts[in_window], predictions[(96, '계획기반')][in_window],
         label='24시간 앞 예측 (가동일 달력 포함)', linewidth=1, alpha=0.9)
plt.axvspan(pd.Timestamp('2021-08-02'), pd.Timestamp('2021-08-07'), color='gray', alpha=0.15,
            label='평일 휴무 (8/2~8/6)')
plt.ylabel('전력사용량')
plt.title('24시간 앞 예측: 평일 휴무 주간 전후')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% #13. 그림 3 — 고전력 경보: 고정 기준 vs 검증 선택 임계값
alarm_plot = results_df[results_df['모델'] != '원본 Feature4'].copy()
alarm_plot = pd.concat([results_df[results_df['모델'] == '원본 Feature4'], alarm_plot])

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, metric in zip(axes, ['Recall', 'F1']):
    y_pos = np.arange(len(alarm_plot))
    ax.barh(y_pos - 0.2, alarm_plot[f'경보(178고정) {metric}'], 0.4,
            label='178 고정', color='#b4b2a9')
    ax.barh(y_pos + 0.2, alarm_plot[f'경보(검증선택) {metric}'].fillna(0), 0.4,
            label='검증 구간에서 선택한 임계값', color='#2a78d6')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(alarm_plot['모델'])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(f'고전력 경보 {metric}')
    ax.grid(axis='x', alpha=0.3)
axes[0].legend(loc='lower right')
plt.tight_layout()
plt.show()


# %% #14. 그림 4 — 24시간 앞(계획기반) 모델의 변수 중요도
model_96, cols_96 = models[(96, '계획기반')]
importance_96 = (
    pd.DataFrame({'변수': cols_96, '중요도': model_96.feature_importances_})
    .sort_values('중요도', ascending=False)
    .reset_index(drop=True)
)
print(importance_96.round(4))

plot_imp = importance_96.head(15).sort_values('중요도')
plt.figure(figsize=(10, 7))
plt.barh(plot_imp['변수'], plot_imp['중요도'], color='#2a78d6')
plt.xlabel('Feature Importance (gain 기준)')
plt.title('24시간 앞 예측(계획기반) 변수 중요도 상위 15')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# %% #15. 결과 저장
print('저장 완료:')
save_csv(results_df, 'improved_model_comparison.csv')
save_csv(ablation_df, 'improved_ablation_24h_plan.csv')
save_csv(importance_96, 'improved_importance_24h_plan.csv')

pred_out = df.loc[is_test, ['timestamp', '날짜', '시간', '분', '전력사용량', '생산량', '가동일']].copy()
pred_out['원본Feature4_예측'] = original_pred
for (h, scenario), pred in predictions.items():
    pred_out[f'{HORIZONS[h]}_{scenario}_예측'] = pred
save_csv(pred_out, 'improved_test_predictions.csv')

print('\n모든 단계 실행 완료')
