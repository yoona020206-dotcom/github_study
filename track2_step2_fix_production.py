# %% [markdown]
# # Track2 Step T2. 생산량 보정 (대상: 생산량=0 & 평균>=50, 685행)
# 규칙 (순서대로, 먼저 해당되는 것만 적용)
#  1) 시간 오류 행(7/13, 7/15): 보정 제외 -> excluded_time_bad
#  2) 앞뒤 생산량 모두 >0 이고 연속 구간 길이 <= L 시간: 선형 보간 -> interp
#  3) 같은 요일·같은 하루 순번(slot)의 생산량>0 기증 행이 >= K개: 중앙값 -> dow_median
#  4) 나머지: 0 유지 -> unfixed
# 원본 생산량은 prod_orig로 보존, 보정값은 prod_fixed. '시간' 열 대신 하루 안의 행 순번(slot)을 쓴다.

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
L_DEFAULT, K_DEFAULT = 6, 5
POWER_MIN = 50


def set_korean_font():
    preferred = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system())
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in [preferred, "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
        if name and name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


set_korean_font()
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 30)

# %% 1. 로드 + 전제 확인
raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df = raw.copy()
df["date"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
assert df["date"].is_monotonic_increasing and (df.groupby("date").size() == 24).all()
df["slot"] = df.groupby("date").cumcount()          # 하루 안의 순번 0~23
df["dow"] = df["date"].dt.dayofweek
df["time_bad"] = ~df["시간"].between(0, 23)
df["sus"] = (df["생산량"] == 0) & (df["평균"] >= POWER_MIN)
print("의심 0:", int(df["sus"].sum()), "| 그 중 시간 오류 행:", int((df["sus"] & df["time_bad"]).sum()))


# %% 2. 보정 함수 (L, K를 바꿔 민감도도 볼 수 있게)
def fix_production(d, L, K):
    out = d[["날짜", "시간", "slot", "dow", "time_bad", "sus", "평균", "생산량", "공장인원"]].copy()
    out = out.rename(columns={"생산량": "prod_orig"})
    out["prod_fixed"] = out["prod_orig"].astype(float)
    out["fix"] = np.where(out["sus"], "unfixed", "none")
    prod = out["prod_orig"].to_numpy(dtype=float)
    sus = out["sus"].to_numpy()
    bad = out["time_bad"].to_numpy()
    n = len(out)

    # 기증 행: 생산량>0, 시간 오류 아님 (의심 0은 원래 0이라 자동 제외)
    donors = out[(out["prod_orig"] > 0) & (~out["time_bad"])]
    med = donors.groupby(["dow", "slot"])["prod_orig"].agg(["median", "size"])

    # 연속 의심 구간 찾기 (파일 순서)
    i = 0
    while i < n:
        if not sus[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and sus[j + 1]:
            j += 1
        length = j - i + 1
        prev_v = prod[i - 1] if i > 0 else np.nan
        next_v = prod[j + 1] if j + 1 < n else np.nan
        run_bad = bad[i:j + 1].any()
        for k in range(i, j + 1):
            if bad[k]:
                out.iloc[k, out.columns.get_loc("fix")] = "excluded_time_bad"
        if not run_bad and length <= L and prev_v > 0 and next_v > 0:
            vals = np.linspace(prev_v, next_v, length + 2)[1:-1]
            out.iloc[i:j + 1, out.columns.get_loc("prod_fixed")] = vals
            out.iloc[i:j + 1, out.columns.get_loc("fix")] = "interp"
        else:
            for k in range(i, j + 1):
                if bad[k]:
                    continue
                key = (out["dow"].iat[k], out["slot"].iat[k])
                if key in med.index and med.loc[key, "size"] >= K:
                    out.iloc[k, out.columns.get_loc("prod_fixed")] = med.loc[key, "median"]
                    out.iloc[k, out.columns.get_loc("fix")] = "dow_median"
        i = j + 1
    return out


fixed = fix_production(df, L_DEFAULT, K_DEFAULT)
print("\n[기본 L=%d, K=%d] 보정 방식별 행수" % (L_DEFAULT, K_DEFAULT))
print(fixed.loc[fixed["sus"], "fix"].value_counts())

# %% 3. 민감도: L, K 조합별 보정 건수
rows = []
for L in [3, 6, 12]:
    for K in [3, 5, 10]:
        f = fix_production(df, L, K)
        c = f.loc[f["sus"], "fix"].value_counts()
        rows.append({"L": L, "K": K, **{k: int(c.get(k, 0)) for k in ["interp", "dow_median", "unfixed", "excluded_time_bad"]}})
sens = pd.DataFrame(rows)
print("\n민감도 (685행 중 방식별 행수)")
print(sens.to_string(index=False))

# %% 4. 보정값의 성격 확인
chg = fixed[fixed["fix"].isin(["interp", "dow_median"])]
print("\n보정값 요약 (prod_fixed)")
print(chg.groupby("fix")["prod_fixed"].describe().round(1))
pos = fixed.loc[(fixed["prod_orig"] > 0), "prod_orig"]
print("\n[참고] 원래 생산량>0 행의 분포: 중앙값 %.1f, 75%% %.1f, 최대 %.1f" % (pos.median(), pos.quantile(.75), pos.max()))
print("전체 생산량 평균: 보정 전 %.1f -> 보정 후 %.1f" % (fixed["prod_orig"].mean(), fixed["prod_fixed"].mean()))
print("생산량=0 행수: 보정 전 %d -> 보정 후 %d" % ((fixed["prod_orig"] == 0).sum(), (fixed["prod_fixed"] == 0).sum()))

# 같은 전력 수준(평균 50~109 등) 정상 생산 행과 비교: 보정값이 전력 대비 터무니없지 않은지
fixed["pwr_bin"] = pd.cut(fixed["평균"], [49, 109, 151, np.inf], labels=["50~109", "110~151", ">=152"])
cmp = pd.DataFrame({
    "보정행 중앙값": chg.assign(b=pd.cut(chg["평균"], [49, 109, 151, np.inf], labels=["50~109", "110~151", ">=152"]))
                    .groupby("b", observed=True)["prod_fixed"].median(),
    "정상행(원생산>0) 중앙값": fixed[fixed["prod_orig"] > 0].groupby("pwr_bin", observed=True)["prod_orig"].median(),
}).round(1)
print("\n전력 수준별 보정값 vs 정상 생산 행 (생산량 중앙값)")
print(cmp)

# %% 5. 그래프
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for name, c in [("interp", "#4C78A8"), ("dow_median", "#F58518")]:
    s = chg.loc[chg["fix"] == name, "prod_fixed"]
    axes[0].hist(s, bins=40, alpha=0.7, label=f"{name} (n={len(s)})", color=c)
axes[0].set(title="보정값 분포", xlabel="prod_fixed", ylabel="행수")
axes[0].legend()
cnt = fixed.loc[fixed["sus"], "fix"].value_counts()
axes[1].bar(cnt.index, cnt.values, color="#72B7B2")
for x, v in zip(cnt.index, cnt.values):
    axes[1].text(x, v, str(v), ha="center", va="bottom")
axes[1].set(title="의심 0 685행 보정 방식", ylabel="행수")
plt.tight_layout()
plt.savefig(OUT_DIR / "t2_production_fix.png", dpi=130)
plt.close()

# %% 6. 저장 (가공본만, 원본 불변)
fixed.drop(columns=["pwr_bin"]).to_csv(OUT_DIR / "t2_production_fixed.csv", index=False, encoding="utf-8-sig")
sens.to_csv(OUT_DIR / "t2_sensitivity.csv", index=False, encoding="utf-8-sig")
print("저장:", sorted(p.name for p in OUT_DIR.glob("t2_*")))
