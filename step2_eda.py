# %% [markdown]
# # Step 2. 변수별 분포와 이상치
# - 기온/풍속/습도/강수량 분포 (히스토그램, 박스플롯)
# - 강수량: 시간당 값인지 일 누적값인지 검증 (리셋 위치, 단조 증가 구간, 정체 구간)
# - 풍속·강수량의 0 비율
# - 기온 급변 구간
# '시간' 컬럼은 쓰지 않는다. 하루 안의 순서는 행 순번(0~23)으로 대신한다.
# (Step 2 확인: 모든 날짜가 24행이고 날짜순 정렬 → 행 순번 = 하루 안의 순서)

# %% 0. 환경 설정
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = ROOT / "data" / "okm_augumented_2021.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def set_korean_font():
    preferred = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system())
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in [preferred, "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
        if name and name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.family"]


print("사용 폰트:", set_korean_font())
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# %% 1. 로드 (원본 읽기 전용) + 행 순번
raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
WEATHER = ["기온", "풍속", "습도", "강수량"]
df = raw[["날짜"] + WEATHER].copy()
df["date"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
df["month"] = df["date"].dt.month
SEASON_MAP = {1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름", 9: "가을"}
SEASON_ORDER = ["겨울", "봄", "여름", "가을"]
df["season"] = pd.Categorical(df["month"].map(SEASON_MAP), categories=SEASON_ORDER, ordered=True)

rows_per_day = df.groupby("date").size()
assert (rows_per_day == 24).all(), "하루 24행이 아닌 날짜가 있음 → 행 순번 사용 불가"
assert df["date"].is_monotonic_increasing, "날짜순 정렬이 아님"
df["slot"] = df.groupby("date").cumcount()  # 하루 안의 순번 0~23
print("모든 날짜 24행, 날짜순 정렬 확인. 행수:", len(df))

# %% 2. 분포 그래프: 히스토그램 (강수량은 0과 양수를 나눠서)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, c, bins in zip(axes.flat[:3], WEATHER[:3], [50, np.arange(-0.05, 7.75, 0.1), np.arange(7.5, 99.5, 1)]):
    s = df[c].dropna()
    ax.hist(s, bins=bins, color="tab:blue", alpha=0.8)
    ax.set_title(f"{c} (n={len(s)})")
    ax.grid(alpha=0.3)
r = df["강수량"].dropna()
ax = axes.flat[3]
ax.hist(r[r > 0], bins=np.logspace(-1, np.log10(r.max()), 40), color="tab:blue", alpha=0.8)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))  # 수식체 눈금(마이너스 깨짐) 대신 일반 숫자
ax.set_title(f"강수량 > 0만 (n={(r > 0).sum()}, 0값 {(r == 0).sum()}행 제외, 로그축)")
ax.grid(alpha=0.3)
fig.suptitle("변수별 분포 (시간 단위 관측치)")
fig.tight_layout()
fig.savefig(OUT_DIR / "step2_hist.png", dpi=150)
plt.show()

# %% 3. 분포 그래프: 박스플롯 (전체 / 계절별)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for j, c in enumerate(WEATHER):
    s = df[c].dropna()
    axes[0, j].boxplot(s, flierprops={"markersize": 2, "alpha": 0.4})
    axes[0, j].set_title(f"{c} 전체 (n={len(s)})")
    data = [df.loc[df["season"] == k, c].dropna() for k in SEASON_ORDER]
    axes[1, j].boxplot(data, tick_labels=SEASON_ORDER, flierprops={"markersize": 2, "alpha": 0.4})
    axes[1, j].set_title(f"{c} 계절별")
    for ax in axes[:, j]:
        ax.grid(axis="y", alpha=0.3)
fig.suptitle("박스플롯 (수염 = 1.5×IQR, 강수량은 원본값 그대로 = 누적값일 가능성 있음)")
fig.tight_layout()
fig.savefig(OUT_DIR / "step2_boxplot.png", dpi=150)
plt.show()

# %% 4. 이상치 점검: 물리 범위 + IQR 규칙 (전체 기준 / 계절 내 기준)
PHYS = {"기온": (-30, 45), "풍속": (0, 40), "습도": (0, 100), "강수량": (0, None)}


def iqr_flags(s, k):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


out_rows = []
for c in WEATHER:
    s = df[c]
    lo, hi = PHYS[c]
    phys = (s < lo) | ((s > hi) if hi is not None else False)
    seasonal = df.groupby("season", observed=True)[c].transform(lambda x: iqr_flags(x, 1.5))
    out_rows.append({
        "변수": c, "n": int(s.notna().sum()),
        "물리범위 밖": int(phys.sum()),
        "IQR1.5 전체": int(iqr_flags(s, 1.5).sum()),
        "IQR3 전체": int(iqr_flags(s, 3).sum()),
        "IQR1.5 계절 내": int(seasonal.sum()),
        "최솟값": s.min(), "최댓값": s.max(),
    })
outlier_tbl = pd.DataFrame(out_rows)
print(outlier_tbl.to_string(index=False))
outlier_tbl.to_csv(OUT_DIR / "step2_outlier_counts.csv", index=False, encoding="utf-8-sig")

# 풍속 상위값 위치 (계절 내 IQR 기준 이상치 중 상위 5)
print("\n풍속 상위 5:")
print(df.nlargest(5, "풍속")[["date", "slot", "풍속", "season"]].to_string(index=False))
print("\n습도 상위값 분포:", df["습도"].value_counts().sort_index().tail(4).to_dict(),
      f"/ 98 비율 {(df['습도'] == 98).mean():.1%}")

# %% 5. 강수량 검증 ① 값이 줄어드는 위치 (리셋은 하루 경계에서만 일어나야 함)
r = df["강수량"]
prev = r.shift()
drop = (r < prev)
drops = df[drop].assign(직전값=prev[drop])
print("직전 행보다 값이 줄어든 횟수:", len(drops))
print("하루 순번별 감소 횟수:", drops["slot"].value_counts().sort_index().to_dict())
for sl in [0, 1]:
    dd = drops[drops["slot"] == sl]
    print(f"  순번 {sl} 감소: {len(dd)}회, 기간 {dd['date'].min().date()} ~ {dd['date'].max().date()}")
    print("    감소 후 값이 0인 비율:", round((dd["강수량"] == 0).mean(), 3))
odd = drops[~drops["slot"].isin([0, 1])]
print("\n하루 경계(순번 0·1)로 설명되지 않는 감소:")
print(odd[["date", "slot", "직전값", "강수량"]].to_string(index=False))

# 리셋 순번이 바뀌는 시점: 월별로 순번 0/1 감소 횟수
sw = drops[drops["slot"].isin([0, 1])].groupby([drops["month"], "slot"]).size().unstack(fill_value=0)
print("\n월별 리셋 순번(0/1) 횟수:\n", sw.to_string())
drops.to_csv(OUT_DIR / "step2_rain_drops.csv", index=False, encoding="utf-8-sig")

# %% 6. 강수량 검증 ② 하루 창 안에서 단조 증가(감소 없음)인가
# 창 A: 순번 0~23 (달력 날짜)  /  창 B: 순번 1 ~ 다음날 순번 0 (리셋이 순번 1인 경우)
df["winB"] = df["date"] - pd.to_timedelta((df["slot"] == 0).astype(int), unit="D")


def monotone_days(key):
    g = df.dropna(subset=["강수량"]).groupby(key)["강수량"]
    rainy = g.max() > 0
    mono = g.apply(lambda s: bool((s.diff().dropna() >= 0).all()))
    return int(rainy.sum()), int((mono & rainy).sum())


for name, key in [("창 A (0~23)", "date"), ("창 B (1~다음날 0)", "winB")]:
    n_rainy, n_mono = monotone_days(key)
    print(f"{name}: 비 온 창 {n_rainy}개 중 감소 없는(단조 비감소) 창 {n_mono}개 ({n_mono / n_rainy:.1%})")

# 기간을 나눠서: 리셋 순번이 1이던 기간(~5/21)은 창 B, 이후는 창 A가 맞는지
cut = pd.Timestamp("2021-05-22")
for name, key, mask in [("~5/21 창 A", "date", df["date"] < cut), ("~5/21 창 B", "winB", df["winB"] < cut),
                        ("5/22~ 창 A", "date", df["date"] >= cut), ("5/22~ 창 B", "winB", df["winB"] >= cut)]:
    g = df[mask].dropna(subset=["강수량"]).groupby(key)["강수량"]
    rainy = g.max() > 0
    mono = g.apply(lambda s: bool((s.diff().dropna() >= 0).all()))
    print(f"  {name}: 비 온 창 {int(rainy.sum())}개 중 단조 비감소 {int((mono & rainy).sum())}개")

# %% 7. 강수량 검증 ③ 정체 구간: 양수 값이 바뀌지 않고 이어지는 길이
# 시간당 값이라면 같은 양수 값이 여러 시간 연속되는 일은 드물어야 하고,
# 누적값이라면 비가 그친 뒤 다음 리셋까지 값이 그대로 유지되어야 한다.
pos = r > 0
same = pos & (r == prev)
print(f"양수 행 {int(pos.sum())}개 중 직전과 같은 값 {int(same.sum())}개 ({same.sum() / pos.sum():.1%})")
run_id = (r != prev).cumsum()
runs = df[pos].groupby(run_id[pos]).agg(값=("강수량", "first"), 길이=("강수량", "size"), 시작=("date", "first"))
print("양수 동일값 연속 길이 분포 (길이: 구간 수):", runs["길이"].value_counts().sort_index().to_dict())

# 단조 증가 구간: 연속으로 값이 늘어나는 구간의 길이
inc = (r > prev).astype(int)
inc_run = inc.groupby((inc == 0).cumsum()).sum()
print("연속 증가 구간 길이 분포 (길이: 구간 수):", inc_run[inc_run > 0].value_counts().sort_index().to_dict())

# 시간당 해석이면 일 합계, 누적 해석이면 창 최댓값이 일 강수량
daily_sum = df.groupby("date")["강수량"].sum()
daily_max = df.groupby("date")["강수량"].max()
print("\n[해석별 일 강수량 비교]")
print(f"  시간당 해석(일 합계)   최댓값 {daily_sum.max():.1f}mm, 300mm 초과 {int((daily_sum > 300).sum())}일")
print(f"  누적 해석(일 최댓값)   최댓값 {daily_max.max():.1f}mm, 300mm 초과 {int((daily_max > 300).sum())}일")
print("  일 합계 상위 5:\n", daily_sum.nlargest(5).round(1).to_string())

# %% 8. 강수량 검증 그래프: 비 많이 온 날 전후의 시간별 값
example_days = ["2021-03-01", "2021-07-06", "2021-08-21", "2021-08-24"]
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for ax, d0 in zip(axes.flat, example_days):
    d0 = pd.Timestamp(d0)
    w = df[(df["date"] >= d0 - pd.Timedelta(days=1)) & (df["date"] <= d0 + pd.Timedelta(days=1))].reset_index()
    ax.plot(range(len(w)), w["강수량"], marker="o", ms=3)
    for i in w.index[w["slot"] == 0]:
        ax.axvline(i, color="gray", ls="--", lw=0.8)
    ax.set_xticks(w.index[w["slot"] % 6 == 0])
    ax.set_xticklabels([f"{a:%m/%d}\n{b}" for a, b in zip(w["date"][w["slot"] % 6 == 0], w["slot"][w["slot"] % 6 == 0])],
                       fontsize=8)
    ax.set_title(f"{d0:%m/%d} 전후 3일 강수량 원본값 (점선: 날짜 경계, 아래 숫자: 하루 순번)")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "step2_rain_examples.png", dpi=150)
plt.show()

# %% 9. 0 비율: 풍속, 강수량 (전체 / 계절별 / 월별)
zero = pd.DataFrame({
    "풍속=0 비율": df.groupby("season", observed=True)["풍속"].apply(lambda s: (s.dropna() == 0).mean()),
    "풍속=0 행": df.groupby("season", observed=True)["풍속"].apply(lambda s: int((s == 0).sum())),
    "강수량=0 비율(원본)": df.groupby("season", observed=True)["강수량"].apply(lambda s: (s.dropna() == 0).mean()),
    "무강수일 비율(일 최댓값=0)": df.groupby("season", observed=True).apply(
        lambda x: (x.groupby("date")["강수량"].max() == 0).mean(), include_groups=False),
    "일수": df.groupby("season", observed=True)["date"].nunique(),
})
total = {
    "풍속=0 비율": (df["풍속"].dropna() == 0).mean(), "풍속=0 행": int((df["풍속"] == 0).sum()),
    "강수량=0 비율(원본)": (df["강수량"].dropna() == 0).mean(),
    "무강수일 비율(일 최댓값=0)": (daily_max == 0).mean(), "일수": df["date"].nunique(),
}
zero.loc["전체"] = total
print(zero.round(3).to_string())
zero.round(4).to_csv(OUT_DIR / "step2_zero_ratio.csv", encoding="utf-8-sig")

print("\n월별 풍속=0 행:", df.groupby("month")["풍속"].apply(lambda s: int((s == 0).sum())).to_dict())
print("풍속 값의 최소 단위(0 다음 값):", sorted(df["풍속"].dropna().unique())[:4])
wz = (df["풍속"] == 0).astype(int)
wz_run = wz.groupby((wz == 0).cumsum()).sum()
print("풍속=0 연속 길이 분포:", wz_run[wz_run > 0].value_counts().sort_index().to_dict())

# %% 10. 기온 급변 ① 연속한 행(1시간) 사이 변화
df["dT_1h"] = df["기온"].diff()
dT = df["dT_1h"].dropna()
print("1시간 기온 변화 분포:")
print(dT.describe(percentiles=[0.005, 0.01, 0.99, 0.995]).round(2).to_string())
for th in [3, 4, 5]:
    print(f"  |변화| ≥ {th}°C: {int((dT.abs() >= th).sum())}회")
big = df.loc[df["dT_1h"].abs() >= 4, ["date", "slot", "기온", "dT_1h", "풍속", "습도", "강수량"]]
print("\n|1시간 변화| ≥ 4°C 목록:\n", big.to_string(index=False))
big.to_csv(OUT_DIR / "step2_temp_jumps_1h.csv", index=False, encoding="utf-8-sig")

# %% 11. 기온 급변 ② 일 단위: 일교차, 전날 대비 일평균 변화
day = df.groupby("date").agg(평균=("기온", "mean"), 최저=("기온", "min"), 최고=("기온", "max"),
                            season=("season", "first"))
day["일교차"] = day["최고"] - day["최저"]
day["전날대비"] = day["평균"].diff()
print("일교차 분포:\n", day["일교차"].describe().round(2).to_string())
print("\n전날 대비 일평균 변화 분포:\n", day["전날대비"].describe().round(2).to_string())
print(f"\n|전날 대비| ≥ 5°C: {int((day['전날대비'].abs() >= 5).sum())}일")
print(day.loc[day["전날대비"].abs() >= 5, ["season", "평균", "전날대비", "일교차"]].round(2).to_string())
print("\n계절별 일교차 평균 / 전날대비 |변화| 평균:")
print(day.groupby("season", observed=True).agg(일교차=("일교차", "mean"),
                                               전날대비_abs=("전날대비", lambda s: s.abs().mean()),
                                               일수=("평균", "size")).round(2).to_string())
day.round(3).to_csv(OUT_DIR / "step2_temp_daily.csv", encoding="utf-8-sig")

# %% 12. 기온 급변 그래프
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(df["date"] + pd.to_timedelta(df["slot"], unit="h"), df["dT_1h"], lw=0.5)
for th in [-4, 4]:
    axes[0].axhline(th, color="red", ls="--", lw=0.8)
axes[0].set_ylabel("1시간 변화 (°C)")
axes[0].set_title("기온 1시간 변화 (빨간 점선: ±4°C)")
axes[1].bar(day.index, day["전날대비"], width=1.0)
for th in [-5, 5]:
    axes[1].axhline(th, color="red", ls="--", lw=0.8)
axes[1].set_ylabel("전날 대비 일평균 변화 (°C)")
axes[1].set_title("일평균 기온 전날 대비 변화 (빨간 점선: ±5°C)")
for ax in axes:
    for d0 in ["2021-03-01", "2021-06-01", "2021-09-01"]:
        ax.axvline(pd.Timestamp(d0), color="gray", ls=":", lw=0.8)
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "step2_temp_change.png", dpi=150)
plt.show()
