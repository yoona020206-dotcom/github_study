# %% [markdown]
# # Step 1. 내 파트 범위 확정 + 기본 진단
# - 범위: 계절(월 기준) + 기온 / 풍속 / 습도 / 강수량
# - '시간' 컬럼은 분석에 쓰지 않는다 (결측 위치 표시용으로만 출력)
# - 원본 CSV는 읽기만 하고, 가공 결과는 outputs/에 저장한다

# %% 0. 환경 설정 (경로, 한글 폰트)
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = ROOT / "data" / "okm_augumented_2021.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def set_korean_font():
    preferred = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system())
    installed = {f.name for f in font_manager.fontManager.ttflist}
    # 윈도우/맥 기본 폰트가 없으면(예: 리눅스) 설치된 한글 폰트로 대체
    for name in [preferred, "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
        if name and name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.family"]


print("사용 폰트:", set_korean_font())
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# %% 1. 원본 로드 (읽기 전용) + 전체 구조 확인
for enc in ["utf-8-sig", "cp949"]:
    try:
        raw = pd.read_csv(DATA_PATH, encoding=enc)
        print("인코딩:", enc)
        break
    except UnicodeDecodeError:
        continue

print("shape:", raw.shape, "(CLAUDE.md 기준 6,168행 × 18열)")
print(raw.dtypes.to_string())
raw.head()

# %% 2. 내 파트 컬럼 지정
# 실제 컬럼명이 다르면 이 딕셔너리만 고친다.
COLS = {
    "date": "날짜",
    "time": "시간",  # 결측 위치 표시용으로만 사용
    "temp": "기온",
    "wind": "풍속",
    "humid": "습도",
    "rain": "강수량",
}
missing_cols = [c for c in COLS.values() if c not in raw.columns]
assert not missing_cols, f"컬럼명 확인 필요: {missing_cols} / 실제 컬럼: {list(raw.columns)}"

WEATHER = [COLS[k] for k in ["temp", "wind", "humid", "rain"]]
df = raw[[COLS["date"], COLS["time"]] + WEATHER].copy()
df["date"] = pd.to_datetime(df[COLS["date"]].astype(str), format="%Y%m%d")  # 원본은 20210101 형식 정수
df["month"] = df["date"].dt.month
print("기간:", df["date"].min().date(), "~", df["date"].max().date(), "/ 일수:", df["date"].dt.date.nunique())

# %% 3. 자료형 점검: 숫자형이 아닌 값이 섞여 있는지
dtype_rows = []
for c in WEATHER:
    coerced = pd.to_numeric(df[c], errors="coerce")
    dtype_rows.append({
        "변수": c,
        "원본 dtype": str(raw[c].dtype),
        "결측(원본)": int(raw[c].isna().sum()),
        "숫자 변환 실패(결측 제외)": int(coerced.isna().sum() - raw[c].isna().sum()),
    })
    df[c] = coerced
dtype_tbl = pd.DataFrame(dtype_rows)
print(dtype_tbl.to_string(index=False))

# %% 4. 결측: 건수와 위치
na_rows = []
for c in WEATHER:
    for idx in df.index[df[c].isna()]:
        na_rows.append({"변수": c, "행 번호": idx,
                        "날짜": df.at[idx, "date"].date(), "시간(표시용)": df.at[idx, COLS["time"]]})
na_tbl = pd.DataFrame(na_rows, columns=["변수", "행 번호", "날짜", "시간(표시용)"])
print("변수별 결측 건수:\n", df[WEATHER].isna().sum().to_string())
print("\n결측 위치:\n", na_tbl.to_string(index=False))
na_tbl.to_csv(OUT_DIR / "step1_missing_locations.csv", index=False, encoding="utf-8-sig")

# %% 5. 기술통계 (전체 기간, 결측 제외)
desc = df[WEATHER].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
desc["skew"] = df[WEATHER].skew()
desc["0값 비율"] = (df[WEATHER] == 0).mean()
desc["음수 개수"] = (df[WEATHER] < 0).sum()
print(desc.round(3).to_string())
desc.round(4).to_csv(OUT_DIR / "step1_describe.csv", encoding="utf-8-sig")

# %% 6. 월 기준 계절 구분 + 계절별 일수 / 행수
SEASON_MAP = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
              6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
SEASON_ORDER = ["겨울", "봄", "여름", "가을"]
df["season"] = pd.Categorical(df["month"].map(SEASON_MAP), categories=SEASON_ORDER, ordered=True)

season_size = df.groupby("season", observed=True).agg(
    포함월=("month", lambda s: ",".join(map(str, sorted(s.unique())))),
    시작일=("date", "min"),
    종료일=("date", "max"),
    일수=("date", lambda s: s.dt.date.nunique()),
    행수=("date", "size"),
)
season_size["시작일"] = season_size["시작일"].dt.date
season_size["종료일"] = season_size["종료일"].dt.date
season_size["행수/일수"] = (season_size["행수"] / season_size["일수"]).round(2)
print(season_size.to_string())

month_size = df.groupby("month").agg(일수=("date", lambda s: s.dt.date.nunique()), 행수=("date", "size"))
print("\n월별:\n", month_size.T.to_string())
season_size.to_csv(OUT_DIR / "step1_season_size.csv", encoding="utf-8-sig")

# %% 7. 계절별 기본 통계 (결측 제외)
def q1(s): return s.quantile(0.25)
def q3(s): return s.quantile(0.75)

season_stats = (df.groupby("season", observed=True)[WEATHER]
                .agg(["count", "mean", "std", "min", q1, "median", q3, "max"]))
for c in WEATHER:
    print(f"\n[{c}]")
    print(season_stats[c].round(2).to_string())
season_stats.round(4).to_csv(OUT_DIR / "step1_season_stats.csv", encoding="utf-8-sig")

# 강수량: 0이 많아 평균만으로는 부족 → 비 온 시간 비율, 비 온 날 수
rain = COLS["rain"]
rain_tbl = df.groupby("season", observed=True).agg(
    관측행=(rain, "count"),
    강수_0초과_행=(rain, lambda s: int((s > 0).sum())),
)
rain_tbl["강수_0초과_비율"] = (rain_tbl["강수_0초과_행"] / rain_tbl["관측행"]).round(3)
rain_tbl["강수일수(하루 중 0초과 1회 이상)"] = (
    df[df[rain] > 0].groupby("season", observed=True)["date"].apply(lambda s: s.dt.date.nunique())
)
print("\n[강수량 보조]\n", rain_tbl.to_string())

# %% 7-1. 보조 점검: 습도 상한값 쏠림
humid = COLS["humid"]
print("습도 상위값 분포:", df[humid].value_counts().sort_index().tail(4).to_dict())
print(f"습도 = 최댓값({df[humid].max()}) 행: {(df[humid] == df[humid].max()).sum()}"
      f" ({(df[humid] == df[humid].max()).mean():.1%})")

# %% 7-2. 보조 점검: 강수량이 '시간당'인지 '일 누적'인지
# 일 누적값이라면 값이 줄어드는 시점은 하루가 바뀌는 시점(리셋)에만 나타나야 한다.
prev = df[rain].shift()
drops = df[df[rain] < prev].assign(직전값=prev)
print("직전 시간보다 값이 줄어든 횟수:", len(drops))
print("감소가 일어난 시각 분포:", drops[COLS["time"]].value_counts().to_dict())
for h in [0, 1]:
    dd = drops[drops[COLS["time"]] == h]["date"]
    print(f"  {h}시 감소: {len(dd)}회, 기간 {dd.min().date()} ~ {dd.max().date()}")
print("\n0·1시 외 감소 (리셋으로 설명되지 않는 경우):")
print(drops[~drops[COLS["time"]].isin([0, 1])][["date", COLS["time"], "직전값", rain]].to_string(index=False))
print("\n예시 3/1~3/2 (0이 아닌 값):")
ex = df[df["date"].between("2021-03-01", "2021-03-02") & (df[rain] > 0)]
print(ex[["date", COLS["time"], rain]].to_string(index=False))

# %% 8. 그래프: 계절별 분포 박스플롯
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, c in zip(axes, WEATHER):
    data = [df.loc[df["season"] == s, c].dropna() for s in SEASON_ORDER]
    ax.boxplot(data, tick_labels=[f"{s}\n(n={len(d)})" for s, d in zip(SEASON_ORDER, data)],
               showfliers=True, flierprops={"markersize": 2, "alpha": 0.4})
    ax.set_title(c)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("계절별 외부요인 분포 (시간 단위 관측치, 가을은 9/1~9/14만 포함)")
fig.tight_layout()
fig.savefig(OUT_DIR / "step1_season_boxplot.png", dpi=150)
plt.show()

# %% 9. 그래프: 일평균 시계열 (계절 경계 표시)
daily = df.groupby(df["date"].dt.date)[WEATHER].mean()
daily.index = pd.to_datetime(daily.index)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for ax, c in zip(axes, WEATHER):
    ax.plot(daily.index, daily[c], lw=1)
    for d in ["2021-03-01", "2021-06-01", "2021-09-01"]:
        ax.axvline(pd.Timestamp(d), color="gray", ls="--", lw=0.8)
    ax.set_ylabel(c)
    ax.grid(alpha=0.3)
axes[0].set_title("일평균 외부요인 추이 (점선: 계절 경계)")
fig.tight_layout()
fig.savefig(OUT_DIR / "step1_daily_timeseries.png", dpi=150)
plt.show()
