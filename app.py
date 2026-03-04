# app.py
import streamlit as st
import pandas as pd

from calc import TruckSpec, run_calc

st.set_page_config(page_title="Расчет транспорта по грузам", layout="wide")
st.title("🚚 Расчет необходимого транспорта под перевозку грузов")

with st.expander("Поля Excel", expanded=True):
    st.markdown("""
**Обязательные:**
- `наименование`
- `длина` (мм), `ширина` (мм), `высота` (мм)
- `штабелируется` (`да`/другое)
- `количество` **или** `qty`

**Вес (`вес`)**:
- может отсутствовать или быть пустым → считаем `вес = 0`, `weight_missing = True`
- отрицательный вес → строка исключается

**Опционально:**
- `max_top_weight` (если `штабелируется="да"` и пусто → 50)
""")

# Список транспортов — как в твоем финальном Colab коде
TRUCKS = [
    TruckSpec(name="Фура 82м3", L=13600, W=2450, H=2600, max_payload=20000, reserve_len=0.0, reserve_wid=0.00),
    TruckSpec(name="Фура 82м3 запас", L=13600, W=2450, H=2600, max_payload=20000, reserve_len=0.10, reserve_wid=0.05),
    TruckSpec(name="10т", L=7000, W=2400, H=2400, max_payload=10000, reserve_len=0.10, reserve_wid=0.05),
    TruckSpec(name="5т", L=6000, W=2400, H=2400, max_payload=5000, reserve_len=0.10, reserve_wid=0.10),
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
    out = run_calc(df_raw, trucks=TRUCKS)
except Exception as e:
    st.error(f"Ошибка расчета: {e}")
    st.stop()

summary_df = out["summary_df"]
best_truck = out["best_truck"]
best_res = out["best_res"]
metrics = out["metrics"]

missing_weight_count = out["missing_weight_count"]
use_payload = out["use_payload"]

# Метрики сверху
c1, c2, c3, c4 = st.columns(4)
c1.metric("Грузовых мест (после количества)", metrics["total_items_after_qty"])
c2.metric("Без веса", missing_weight_count)
c3.metric("Учет грузоподъемности", "ДА" if use_payload else "НЕТ")
c4.metric("Нужно машин (лучший)", int(best_res["trucks_used"]) if best_res else 0)

if not use_payload:
    st.warning(
        "Обнаружены грузы без веса. Подбор транспорта выполнен БЕЗ учета грузоподъемности "
        "(только геометрия), чтобы не занизить кол-во машин."
    )

st.subheader("Сравнение типов транспорта")
st.dataframe(summary_df, width="stretch")

if best_truck is not None and best_res is not None:
    st.subheader("Итог")
    st.write(f"**Выбран транспорт:** {best_truck.name}")
    st.write(f"**Нужно машин:** {best_res['trucks_used']}")
    st.write(
        f"Эффективные размеры пола: **{best_truck.eff_L} × {best_truck.eff_W} мм** | "
        f"Запас: **{int(best_truck.reserve_len*100)}% / {int(best_truck.reserve_wid*100)}%** | "
        f"Высота: **{best_truck.eff_H} мм**"
    )

st.subheader("Характеристики груза")
cc1, cc2, cc3, cc4 = st.columns(4)
cc1.metric("Самый длинный (мм)", metrics["longest_len_mm"])
cc2.metric("Самый широкий (мм)", metrics["widest_mm"])
cc3.metric(">150 кг (из известных)", metrics["count_gt_150_known"])
cc4.metric(">500 кг (из известных)", metrics["count_gt_500_known"])

if metrics["heaviest_row_known"] is not None:
    hr = metrics["heaviest_row_known"]
    st.info(
        f"Самый тяжелый (по известным весам): **{metrics['heaviest_weight_known']:.1f} кг** | "
        f"{hr['наименование']} ({int(hr['длина'])}×{int(hr['ширина'])}×{int(hr['высота'])} мм)"
    )
else:
    st.info("Самый тяжелый груз: веса отсутствуют во всех строках (или все веса пустые).")

tabs = st.tabs(["Статистика по машинам", "Негабарит", "Примеры размещений", "Не уложилось"])

with tabs[0]:
    if best_res is not None and not best_res["truck_stats_df"].empty:
        st.dataframe(best_res["truck_stats_df"], width="stretch")
    else:
        st.write("Статистика по машинам: нет (ничего не уложилось)")

with tabs[1]:
    if best_res is not None and isinstance(best_res.get("oversize_df"), pd.DataFrame):
        overs_df = best_res["oversize_df"]
        st.dataframe(overs_df, width="stretch")

with tabs[2]:
    if best_res is not None and isinstance(best_res.get("placements_df"), pd.DataFrame):
        st.dataframe(best_res["placements_df"].head(30), width="stretch")

with tabs[3]:
    if best_res is not None and isinstance(best_res.get("not_packed_df"), pd.DataFrame):
        st.dataframe(best_res["not_packed_df"], width="stretch")
