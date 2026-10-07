# -*- coding: utf-8 -*-
"""보고서용 그림 생성 (report/figures/*.pdf). 개선 스크립트를 실행해 결과를 재사용한다."""
import os
import runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
    'pdf.fonttype': 42, 'font.size': 10, 'axes.titlesize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})
BLUE, LBLUE, GRAY, ORANGE, RED = '#2a78d6', '#8fb8e8', '#b4b2a9', '#eb6834', '#e34948'


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), bbox_inches='tight')
    plt.close()
    print('saved', name)


# ------------------------------------------------------------------
# 개선 스크립트 실행 → 변수 재사용
# ------------------------------------------------------------------
os.chdir(ROOT)
g = runpy.run_path(os.path.join(ROOT, 'KAMP_modeling_improved.py'), run_name='report')
df, power, results_df = g['df'], g['power'], g['results_df']
is_test, is_train_all = g['is_test'], g['is_train_all']
predictions, pred_nocal = g['predictions'], g['pred_nocal']
importance_96 = g['importance_96']
TEST_START = g['TEST_START']

raw = pd.read_csv(os.path.join(ROOT, 'Power15MinClean_NoDatetime.csv')).sort_values(['날짜', '시간', '분']).reset_index(drop=True)
raw['ts'] = pd.to_datetime(raw['날짜'].astype(str)) + pd.to_timedelta(raw['시간'], unit='h') + pd.to_timedelta(raw['분'], unit='m')
P = raw['전력사용량']

# 1. 일평균 전력 추이 --------------------------------------------------
daily = raw.groupby('날짜').agg(p=('전력사용량', 'mean'), day=('day', 'first'))
daily.index = pd.to_datetime(daily.index.astype(str))
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(daily.index, daily['p'], color=GRAY, lw=0.8, zorder=1)
wk = daily['day'] <= 5
ax.scatter(daily.index[wk], daily['p'][wk], s=8, color=BLUE, label='평일', zorder=2)
ax.scatter(daily.index[~wk], daily['p'][~wk], s=8, color=ORANGE, label='주말', zorder=2)
ax.axvspan(pd.Timestamp('2021-08-02'), pd.Timestamp('2021-08-07'), color=RED, alpha=0.12, label='평일 휴무(8/2~8/6)')
ax.axvline(TEST_START, color='k', ls='--', lw=0.8)
ax.text(TEST_START, ax.get_ylim()[1] * 0.97, ' Test 시작', fontsize=9, va='top')
ax.set_ylabel('일평균 전력사용량')
ax.legend(loc='lower left', ncol=3, fontsize=8, frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m월'))
save('fig01_daily_power.pdf')

# 2. 요일 × 시간 평균 전력 --------------------------------------------
hm = raw.assign(hour=raw['ts'].dt.hour).pivot_table(index='day', columns='hour', values='전력사용량', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10, 2.9))
im = ax.imshow(hm.values, aspect='auto', cmap='Blues')
ax.set_yticks(range(7))
ax.set_yticklabels(['월', '화', '수', '목', '금', '토', '일'])
ax.set_xticks(range(24))
ax.set_xlabel('시각')
plt.colorbar(im, ax=ax, label='평균 전력', fraction=0.025)
save('fig02_heatmap.pdf')

# 3. lag_1 과 생산량 계단 (8/11) --------------------------------------
day = raw[(raw['날짜'] == 20210811) & (raw['시간'] >= 6) & (raw['시간'] < 13)].copy()
day['lag1'] = P.shift(1)[day.index]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
ax = axes[0]
ax.plot(day['ts'], day['전력사용량'], color=BLUE, lw=1.6, marker='o', ms=2.5, label='실제 전력')
ax.plot(day['ts'], day['lag1'], color=ORANGE, lw=1.6, ls='--', label='lag_1_power (15분 전)')
ax.set_title('(a) 실제 전력과 15분 전 전력')
ax.legend(fontsize=8, frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax = axes[1]
colors = [BLUE if (r['시간'] == 10) else LBLUE for _, r in day.iterrows()]
ax.bar(day['ts'], day['생산량'], width=0.009, color=colors)
ax.set_ylabel('생산량')
ax2 = ax.twinx()
ax2.plot(day['ts'], day['전력사용량'], color=ORANGE, lw=1.4)
ax2.set_ylabel('전력', color=ORANGE)
ax2.spines['right'].set_visible(True)
ax.set_title('(b) 생산량은 1시간 단위 계단 (진한 막대 = 10시대)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
save('fig03_lag_and_leak.pdf')

# 4. Persistence 예측구간별 R² ----------------------------------------
from sklearn.metrics import r2_score
labels, vals = [], []
for h, n in [(1, '15분'), (4, '1시간'), (8, '2시간'), (16, '4시간'), (96, '전날\n같은 시각'), (672, '지난주\n같은 시각')]:
    m = P.shift(h).notna()
    labels.append(n)
    vals.append(r2_score(P[m], P.shift(h)[m]))
fig, ax = plt.subplots(figsize=(7, 3))
bars = ax.bar(labels, vals, color=[BLUE] + [LBLUE] * 3 + [RED, LBLUE], width=0.6)
ax.bar_label(bars, fmt='%.2f', fontsize=9)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('R² (과거 값 그대로 사용)')
save('fig04_persistence_decay.pdf')

# 5. 원본 모델 비교 (원본 노트북 결과) -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
names = ['Baseline\nXGBoost', 'Persistence', 'Feature4\nXGBoost']
for ax, vals, lab, fmt in [(axes[0], [17.54, 7.63, 5.71], 'Test MAE (낮을수록 좋음)', '%.2f'),
                           (axes[1], [0.779, 0.953, 0.983], 'Test R²', '%.3f')]:
    b = ax.bar(names, vals, color=[GRAY, LBLUE, BLUE], width=0.55)
    ax.bar_label(b, fmt=fmt, fontsize=9)
    ax.set_title(lab)
axes[1].set_ylim(0.7, 1.0)
save('fig05_original_comparison.pdf')

# 6. 피크 과소예측 (9/10) --------------------------------------------
r = pd.read_csv(os.path.join(ROOT, 'outputs', 'peak_alarm_test_results.csv'), parse_dates=['timestamp'])
d = r[(r['timestamp'].dt.strftime('%Y-%m-%d') == '2021-09-10') & (r['timestamp'].dt.hour.between(7, 17))]
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(d['timestamp'], d['실제전력'], color=BLUE, lw=1.6, label='실제 전력')
ax.plot(d['timestamp'], d['예측전력'], color=ORANGE, lw=1.6, ls='--', label='Feature4 예측')
ax.axhline(178, color=RED, ls=':', lw=1.2, label='고전력 기준 178')
fn = d[(d['실제전력'] >= 178) & (d['예측전력'] < 178)]
ax.scatter(fn['timestamp'], fn['실제전력'], color=RED, s=22, zorder=3, label='놓친 고전력(FN)')
ax.legend(fontsize=8, ncol=4, frameon=False, loc='lower center')
ax.set_ylabel('전력사용량')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
save('fig06_peak_underprediction.pdf')

# 7. Baseline 휴무 주간 오판 -----------------------------------------
f = ['시간', '분', '생산량', '기온', '풍속', '습도', '강수량', '전기요금(계절)', 'day', 'd', 'm', '공장인원', '인건비']
s = int(len(raw) * 0.8)
bm = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, random_state=42).fit(raw[f][:s], P[:s])
t = raw[s:].copy()
t['b'] = bm.predict(t[f])
gd = t.groupby('날짜').agg(a=('전력사용량', 'mean'), b=('b', 'mean')).loc[20210726:20210815]
x = pd.to_datetime(gd.index.astype(str))
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(x, gd['a'], color=BLUE, marker='o', ms=3.5, label='실제 일평균')
ax.plot(x, gd['b'], color=ORANGE, marker='D', ms=3.5, ls='--', label='Baseline 예측 일평균')
ax.axvspan(pd.Timestamp('2021-08-01 12:00'), pd.Timestamp('2021-08-06 12:00'), color=GRAY, alpha=0.3, label='평일 휴무')
ax.legend(fontsize=8, frameon=False)
ax.set_ylabel('일평균 전력')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
save('fig07_shutdown_baseline.pdf')

# 8. 피크 조건: 시간대별 고전력 집중도 + 피크 직전 1시간 ---------------
pk = raw.copy()
pk['연월'] = pk['날짜'].astype(str).str[:6]
pk['q95'] = pk.groupby('연월')['전력사용량'].transform(lambda v: v.quantile(0.95))
pk['high'] = pk['전력사용량'] >= pk['q95']
pk['hour'] = pk['ts'].dt.hour
rate = pk.groupby('hour')['high'].mean() * 100
pre = pd.read_csv(os.path.join(ROOT, 'outputs', 'monthly_peak_previous_1hour.csv')).groupby('피크까지분')['전력사용량'].mean()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), gridspec_kw={'width_ratios': [1.6, 1]})
ax = axes[0]
ax.bar(rate.index, rate.values, color=[BLUE if 8 <= h <= 11 else (ORANGE if h == 12 else LBLUE) for h in rate.index])
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel('시각')
ax.set_ylabel('고전력 구간 발생률 (%)')
ax.set_title('(a) 시간대별 월 상위 5% 고전력 발생률')
ax = axes[1]
ax.plot(pre.index, pre.values, marker='o', color=BLUE)
for xv, yv in zip(pre.index, pre.values):
    ax.text(xv, yv + 3, f'{yv:.0f}', ha='center', fontsize=8)
ax.axvline(0, color=RED, ls='--', lw=1)
ax.set_xlabel('월별 최대피크까지 남은 시간(분)')
ax.set_ylabel('평균 전력')
ax.set_title('(b) 월별 최대피크 직전 1시간 (9건 평균)')
save('fig08_peak_conditions.pdf')

# 9. 개선 모델: 예측구간별 R² ----------------------------------------
H = {1: '15분 앞', 4: '1시간 앞', 96: '24시간 앞'}
fig, ax = plt.subplots(figsize=(8, 3.3))
xx = np.arange(3)
w = 0.26
sel = lambda h, sc, col: results_df.query('h == @h and 시나리오 == @sc')[col].iloc[0]
for off, vals, lab, c in [
    (-w, [sel(h, '실적기반', 'Persistence R2') for h in H], 'Persistence', GRAY),
    (0, [sel(h, '실적기반', 'R2') for h in H], '개선 – 실적기반', LBLUE),
    (w, [sel(h, '계획기반', 'R2') for h in H], '개선 – 계획기반', BLUE)]:
    b = ax.bar(xx + off, vals, w, label=lab, color=c)
    ax.bar_label(b, fmt='%.2f', fontsize=8)
ax.set_xticks(xx)
ax.set_xticklabels(H.values())
ax.set_ylabel('Test R²')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
save('fig09_improved_r2.pdf')

# 10. 24시간 앞 예측: 가동일 달력 효과 --------------------------------
ts = df['timestamp'][is_test]
win = ((ts >= '2021-07-28') & (ts < '2021-08-12')).to_numpy()
yt = power[is_test].to_numpy()
fig, ax = plt.subplots(figsize=(10, 3.3))
ax.plot(ts[win], yt[win], color=BLUE, lw=0.9, label='실제 전력')
ax.plot(ts[win], pred_nocal[win], color=GRAY, lw=0.9, ls='--', label='24시간 앞 예측 (가동일 달력 없음)')
ax.plot(ts[win], predictions[(96, '계획기반')][win], color=ORANGE, lw=0.9, label='24시간 앞 예측 (가동일 달력 포함)')
ax.axvspan(pd.Timestamp('2021-08-02'), pd.Timestamp('2021-08-07'), color=GRAY, alpha=0.15)
ax.legend(fontsize=8, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
ax.set_ylabel('전력사용량')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
save('fig10_calendar_effect.pdf')

# 11. 경보 성능: 고정 vs 검증선택 -------------------------------------
rr = results_df.copy()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
yy = np.arange(len(rr))
for ax, met in zip(axes, ['Recall', 'F1']):
    ax.barh(yy - 0.2, rr[f'경보(178고정) {met}'], 0.4, color=GRAY, label='기준 178 고정')
    ax.barh(yy + 0.2, rr[f'경보(검증선택) {met}'].fillna(0), 0.4, color=BLUE, label='검증 구간 선택 임계값')
    ax.set_yticks(yy)
    ax.set_yticklabels(rr['모델'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(f'고전력 경보 {met}')
axes[0].legend(fontsize=8, frameon=False, loc='lower right')
save('fig11_alarm.pdf')

# 12. 24시간 앞 변수 중요도 -------------------------------------------
im = importance_96.head(12).sort_values('중요도')
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.barh(im['변수'], im['중요도'], color=BLUE)
ax.set_xlabel('Feature importance (gain)')
save('fig12_importance_24h.pdf')

print('done')
