import streamlit as st
import pandas as pd

from calc import TruckSpec, run_calc

st.set_page_config(page_title="Расчет транспорта по грузам", layout="wide")

st.title("🚚 Расчет необходимого транспорта (геометрия + shelf packing)")
st.caption("Загрузи Excel с грузами → получи оценку количества машин и список негабаритов.")

with st.expander("Поля Excel-формы", expanded=True):
    st.markdown("""
**Обязательные колонки:**
- `наименование`
- `длина` (мм)
- `ширина` (мм)
- `высота` (мм)
- `штабелируется` (`да`/другое)
- `количество` **или** `qty`

**Рекомендуемая:**
- `вес` (кг) — может быть пустым (тогда считаем `вес=0` и предупреждаем)

**Опциональная:**
- `max_top_weight` (кг) — сейчас только нормализуется
""")

st.sidebar.header("Настройки транспорта")

reserve_len = st.sidebar.slider("Запас по длине, %", 0, 30, 15) / 100.0
reserve_wid = st.sidebar.slider("Запас по ширине, %", 0, 30, 10) / 100.0

# Можешь расширять список типов как угодно
trucks = [
    TruckSpec(name="Фура 82м3", L=13600, W=2450, H=2700, max_payload=20000, reserve_len=reserve_len, reserve_wid=reserve_wid),
    TruckSpec(name="5т",       L=6000,  W=2400, H=2400, max_payload=5000,  reserve_len=reserve_len, reserve_wid=reserve_wid),
]

uploaded = st.file_uploader("Загрузи Excel (.xlsx)", type=["xlsx"])

if not uploaded:
    st.stop()

try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Не удалось прочитать Excel: {e}")
    st.stop()

try:
    out = run_calc(df_raw, trucks=trucks)
except Exception as e:
    st.error(f"Ошибка расчета: {e}")
    st.stop()

meta = out["meta"]
summary_df = out["summary_df"]
best_truck = out["best_truck"]
best_res = out["best_res"]

col1, col2, col3 = st.columns(3)
col1.metric("Грузовых мест (после количества)", meta["total_items_after_qty"])
col2.metric("Без веса", meta["missing_weight_count"])
col3.metric("Доля без веса", f"{meta['missing_weight_ratio']:.1%}")

if meta["missing_weight_count"] > 0:
    st.warning("Есть позиции без веса. Подбор транспорта выполняется по геометрии; ограничение по грузоподъемности учитывается только по известным весам.")

st.subheader("Сравнение типов транспорта (по геометрии)")
st.dataframe(summary_df, use_container_width=True)

if best_truck is not None:
    st.subheader("Итог по выбранному транспорту")
    st.write(
        f"**{best_truck.name}** | эффективный пол: **{best_truck.eff_L}×{best_truck.eff_W} мм** "
        f"| высота: **{best_truck.eff_H} мм** | запас: **{int(best_truck.reserve_len*100)}% / {int(best_truck.reserve_wid*100)}%**"
    )
    st.write(f"**Нужно машин (по геометрии): {best_res['trucks_used']}**")

st.subheader("Характеристики груза")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Самый длинный (мм)", meta["longest_len_mm"])
c2.metric("Самый широкий (мм)", meta["widest_mm"])
c3.metric(">150 кг (из известных)", meta["count_gt_150_known"])
c4.metric(">500 кг (из известных)", meta["count_gt_500_known"])

if meta["heaviest_known"] is not None:
    hk = meta["heaviest_known"]
    st.info(f"Самый тяжелый (по известным весам): **{hk['вес']} кг** — {hk['наименование']} ({int(hk['длина'])}×{int(hk['ширина'])}×{int(hk['высота'])} мм)")
else:
    st.info("Вес отсутствует во всех строках — самый тяжелый определить нельзя.")

tabs = st.tabs(["Статистика по машинам", "Размещения", "Негабарит", "Не уложилось"])

with tabs[0]:
    st.dataframe(best_res["truck_stats_df"], use_container_width=True)

with tabs[1]:
    st.dataframe(best_res["placements_df"].head(2000), use_container_width=True)

with tabs[2]:
    overs = best_res["oversize_df"][["наименование", "длина", "ширина", "высота", "вес", "weight_missing"]]
    st.dataframe(overs, use_container_width=True)

with tabs[3]:
    notp = best_res["not_packed_df"][["наименование", "длина", "ширина", "высота", "вес", "weight_missing"]]
    st.dataframe(notp, use_container_width=True)