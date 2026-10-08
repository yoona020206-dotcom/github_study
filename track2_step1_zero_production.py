# %% [markdown]
# # Track2 Step T1. 생산량=0 현황 (결측 정의 전 점검)
# - 생산량 열: NaN 여부, 0의 개수
# - 생산량=0 행을 전력 수준별로 나눠 날짜·시간대·요일·월 분포 확인
# - 이 Step은 값을 바꾸지 않는다 (보정 대상 규칙은 사용자 결정 후 T2에서 적용)
# 주의: 시간 오류 48행(7/13, 7/15)은 시간대 분석에서 제외한다 (날짜 단위 분석에는 포함).

# %% 0. 환경 설정
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_PATH = ROOT / "data" / "okm_augumented_2021.csv"
OUT_DIR = ROOT / "outputs" / "track2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)

# %% 1. 로드 (원본 읽기 전용)
raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = raw.copy()
df["date"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.dayofweek  # 0=월 ... 6=일
df["time_bad"] = ~df["시간"].between(0, 23)
print("shape:", df.shape)
print("생산량 NaN:", int(df["생산량"].isna().sum()), "| 생산량=0:", int((df["생산량"] == 0).sum()),
      "| 생산량>0:", int((df["생산량"] > 0).sum()))
print("시간 오류 행:", int(df["time_bad"].sum()))
print(df["생산량"].describe().round(1))

# %% 2. 생산량=0 행을 전력 수준으로 나누기
# 전력 수준은 PDF 5x4 매트릭스 경계(쉬는 날 1~49, 낮은 50~109, 중간 110~151, 높은 >=152)를 그대로 사용
bins = [-np.inf, 0, 49, 109, 151, np.inf]
labels = ["전력0", "쉬는날(1~49)", "낮은가동(50~109)", "중간가동(110~151)", "높은가동(>=152)"]
df["power_lvl"] = pd.cut(df["평균"], bins=bins, labels=labels)
zero = df[df["생산량"] == 0]
tab = zero["power_lvl"].value_counts().reindex(labels)
print("\n생산량=0 행의 전력 수준 분포 (n=%d)" % len(zero))
print(pd.DataFrame({"행수": tab, "비율%": (tab / len(zero) * 100).round(2)}))
print("\n[참고] 전체 행의 전력 수준 분포")
print(df["power_lvl"].value_counts().reindex(labels))
print("\n[참고] 생산량>0 행의 전력 수준 분포")
print(df.loc[df["생산량"] > 0, "power_lvl"].value_counts().reindex(labels))

# %% 3. 의심 0 (생산량=0 & 평균>=50) 날짜·요일·월 분포
sus = zero[zero["평균"] >= 50].copy()
print("\n의심 0 (생산량=0 & 평균>=50):", len(sus), "행")
print("  평균전력>40:", int((zero["평균"] > 40).sum()), "행 (사전점검 기준)")
by_day = sus.groupby("date").size().rename("n_rows")
print("  해당 날짜 수:", by_day.shape[0])
print("  하루 24행 전부가 의심 0인 날:", int((by_day == 24).sum()))
print("  상위 15일(행수):")
print(by_day.sort_values(ascending=False).head(15))

print("\n월별 의심 0 행수 / 월 전체 행수")
m = pd.DataFrame({"의심0": sus.groupby("month").size(), "전체": df.groupby("month").size()}).fillna(0).astype(int)
m["비율%"] = (m["의심0"] / m["전체"] * 100).round(1)
print(m)

print("\n요일별 의심 0 행수 (0=월 ... 6=일)")
print(sus.groupby("dow").size().reindex(range(7), fill_value=0))

# 시간대(행 순번 대신 '시간' 열)는 오류 행 제외하고 참고만
sus_ok = sus[~sus["time_bad"]]
print("\n시간대별 의심 0 행수 (시간 오류 행 제외, 참고용)")
print(sus_ok.groupby("시간").size().reindex(range(24), fill_value=0).to_string())

# %% 4. 날짜 단위: 그날 생산량 합 vs 평균전력 합
daily = df.groupby("date").agg(prod_sum=("생산량", "sum"), power_mean=("평균", "mean"),
                              n_zero=("생산량", lambda s: int((s == 0).sum())))
daily["prod_all_zero"] = daily["prod_sum"] == 0
print("\n날짜 수:", len(daily), "| 하루 생산량 합=0인 날:", int(daily["prod_all_zero"].sum()))
print("그 중 일평균전력 >= 50인 날:", int((daily.loc[daily["prod_all_zero"], "power_mean"] >= 50).sum()))
print(daily.loc[daily["prod_all_zero"], "power_mean"].describe().round(1))
print("생산 있는 날 중 0이 섞인 시간 수 분포 (n_zero):")
print(daily.loc[~daily["prod_all_zero"], "n_zero"].describe().round(1))

# %% 5. 그래프: 생산량=0 행의 전력 분포 + 의심 0 월별
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].hist(zero["평균"], bins=np.arange(0, 400, 10), color="#4C78A8")
axes[0].axvline(50, color="red", ls="--", lw=1)
axes[0].set(title="생산량=0 행의 평균전력 분포", xlabel="평균전력", ylabel="행수")
axes[1].bar(m.index.astype(str), m["의심0"], color="#F58518")
axes[1].set(title="의심 0 (생산량=0 & 평균>=50) 월별 행수", xlabel="월")
d = daily.reset_index()
axes[2].scatter(d["date"], d["prod_sum"], s=8, c=np.where(d["prod_all_zero"], "#E45756", "#4C78A8"))
axes[2].set(title="일 생산량 합 (빨강=하루 합 0)", xlabel="날짜")
axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(OUT_DIR / "t1_zero_production.png", dpi=130)
plt.close()

# %% 6. 표 저장 (원본 불변)
tab.rename("rows").to_frame().to_csv(OUT_DIR / "t1_zero_by_power_level.csv", encoding="utf-8-sig")
sus[["날짜", "시간", "평균", "생산량", "공장인원"]].to_csv(OUT_DIR / "t1_suspect_zero_rows.csv", index=False, encoding="utf-8-sig")
daily.to_csv(OUT_DIR / "t1_daily_summary.csv", encoding="utf-8-sig")
print("저장 완료:", sorted(p.name for p in OUT_DIR.glob("t1_*")))
