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

# ------------------------------------------------------------
# Транспорт (как в твоем финальном Colab коде)
# ------------------------------------------------------------
ALL_TRUCKS = [
    TruckSpec(name="Фура 82м3", L=13600, W=2450, H=2600, max_payload=20000, reserve_len=0.0,  reserve_wid=0.00),
    TruckSpec(name="Фура 82м3 запас", L=13600, W=2450, H=2600, max_payload=20000, reserve_len=0.10, reserve_wid=0.05),
    TruckSpec(name="10т",       L=7000,  W=2400, H=2400, max_payload=10000, reserve_len=0.10, reserve_wid=0.05),
    TruckSpec(name="5т",        L=6000,  W=2400, H=2400, max_payload=5000,  reserve_len=0.10, reserve_wid=0.10),
]

st.sidebar.header("Выбор транспорта")
names = [t.name for t in ALL_TRUCKS]
selected_names = st.sidebar.multiselect(
    "Типы транспорта для расчёта",
    options=names,
    default=names
)

TRUCKS = [t for t in ALL_TRUCKS if t.name in selected_names]
if not TRUCKS:
    st.warning("Выбери хотя бы один тип транспорта слева в сайдбаре.")
    st.stop()

# ------------------------------------------------------------
# Загрузка Excel
# ------------------------------------------------------------
uploaded = st.file_uploader("Загрузи Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.stop()

try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Не удалось прочитать Excel: {e}")
    st.stop()

# ------------------------------------------------------------
# Расчет
# ------------------------------------------------------------
try:
    out = run_calc(df_raw, trucks=TRUCKS)
except Exception as e:
    st.error(f"Ошибка расчета: {e}")
    st.stop()

fleet_summary_df = out["fleet_summary_df"]
total_trucks = out["total_trucks"]

placements_df = out["placements_df"]
truck_stats_df = out["truck_stats_df"]
oversize_df = out["oversize_df"]
not_packed_df = out["not_packed_df"]

metrics = out["metrics"]
missing_weight_count = out["missing_weight_count"]
use_payload = out["use_payload"]

# ------------------------------------------------------------
# Верхние метрики
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Грузовых мест (после количества)", metrics["total_items_after_qty"])
c2.metric("Без веса", missing_weight_count)
c3.metric("Учет грузоподъемности", "ДА" if use_payload else "НЕТ")
c4.metric("Всего машин (смешанный парк)", total_trucks)

if not use_payload:
    st.warning(
        "Обнаружены грузы без веса. Подбор транспорта выполнен БЕЗ учета грузоподъемности "
        "(только геометрия), чтобы не занизить кол-во машин."
    )

# ------------------------------------------------------------
# Состав парка
# ------------------------------------------------------------
st.subheader("Состав парка (смешанный подбор)")
st.dataframe(fleet_summary_df, width="stretch")

# ------------------------------------------------------------
# Характеристики груза (добавили самый высокий)
# ------------------------------------------------------------
st.subheader("Характеристики груза")
cc1, cc2, cc3, cc4, cc5 = st.columns(5)
cc1.metric("Самый длинный (мм)", metrics["longest_len_mm"])
cc2.metric("Самый широкий (мм)", metrics["widest_mm"])
cc3.metric("Самый высокий (мм)", metrics["tallest_mm"])
cc4.metric(">150 кг (из известных)", metrics["count_gt_150_known"])
cc5.metric(">500 кг (из известных)", metrics["count_gt_500_known"])

if metrics["heaviest_row_known"] is not None:
    hr = metrics["heaviest_row_known"]
    st.info(
        f"Самый тяжелый (по известным весам): **{metrics['heaviest_weight_known']:.1f} кг** | "
        f"{hr['наименование']} ({int(hr['длина'])}×{int(hr['ширина'])}×{int(hr['высота'])} мм)"
    )
else:
    st.info("Самый тяжелый груз: веса отсутствуют во всех строках (или все веса пустые).")

# ------------------------------------------------------------
# Таблицы
# ------------------------------------------------------------
tabs = st.tabs(["Статистика по машинам", "Негабарит", "Примеры размещений", "Не уложилось"])

with tabs[0]:
    st.dataframe(truck_stats_df, width="stretch")

with tabs[1]:
    st.dataframe(oversize_df, width="stretch")

with tabs[2]:
    st.dataframe(placements_df.head(100), width="stretch")

with tabs[3]:
    st.dataframe(not_packed_df, width="stretch")
