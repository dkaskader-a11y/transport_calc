# ============================
# Streamlit Cloud version
#  - адаптировано под "финальную" логику из Colab-кода пользователя
#  - интерфейс: загрузка Excel -> сравнение типов транспорта -> итог + таблицы
# ============================

# ----------------------------
# FILE: requirements.txt
# ----------------------------
# streamlit
# pandas
# numpy
# openpyxl

# ----------------------------
# FILE: calc.py
# ----------------------------
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ============================================================
# D) Packing module (2D shelf packing + запасы L/W)
# ============================================================

@dataclass
class TruckSpec:
    name: str
    L: int  # мм (паспорт)
    W: int  # мм (паспорт)
    H: int  # мм (паспорт)
    max_payload: float  # кг
    reserve_len: float = 0.15  # по длине
    reserve_wid: float = 0.10  # по ширине

    @property
    def eff_L(self) -> int:
        return int(self.L * (1 - self.reserve_len))

    @property
    def eff_W(self) -> int:
        return int(self.W * (1 - self.reserve_wid))

    @property
    def eff_H(self) -> int:
        return int(self.H)  # без запаса


@dataclass
class Item:
    idx: int
    name: str
    L: int
    W: int
    H: int
    weight: float
    weight_missing: bool


@dataclass
class Placement:
    truck_no: int
    shelf_no: int
    x: int
    y: int
    placed_L: int
    placed_W: int
    rotated: bool
    item_idx: int
    item_name: str
    item_weight: float
    weight_missing: bool


def normalize_input(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Адаптация 1-в-1 под твою "финальную" версию:
      - обязательные: наименование, длина, ширина, высота, штабелируется
      - вес НЕ обязательный: если нет -> создаём, NaN -> вес=0 + weight_missing=True
      - qty из количество/qty, иначе 1
      - отрицательный вес -> исключаем строку
      - строки без габаритов/qty -> исключаем
      - размножаем по qty
    Возвращает: нормализованный df, missing_weight_count (после учета qty)
    """
    required_columns = ["наименование", "длина", "ширина", "высота", "штабелируется"]
    missing_cols = set(required_columns) - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

    if "вес" not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["вес"] = np.nan

    df = df_raw.copy()

    # qty
    if "qty" not in df.columns and "количество" in df.columns:
        df["qty"] = df["количество"]
    elif "qty" not in df.columns and "количество" not in df.columns:
        df["qty"] = 1

    # строго числа: габариты/qty
    num_cols_strict = ["длина", "ширина", "высота", "qty"]
    for col in num_cols_strict:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # вес допускаем пустой
    df["вес"] = (
        df["вес"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["вес"] = pd.to_numeric(df["вес"], errors="coerce")

    df["weight_missing"] = df["вес"].isna()
    df.loc[df["weight_missing"], "вес"] = 0.0

    # штабелируется
    df["штабелируется"] = df["штабелируется"].astype(str).str.strip().str.lower()
    df["штабелируется"] = np.where(df["штабелируется"].eq("да"), "да", "нет")

    # max_top_weight
    if "max_top_weight" not in df.columns:
        df["max_top_weight"] = np.nan

    df["max_top_weight"] = pd.to_numeric(
        df["max_top_weight"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df.loc[(df["штабелируется"] == "да") & (df["max_top_weight"].isna()), "max_top_weight"] = 50.0

    # qty int
    df["qty"] = df["qty"].fillna(1)
    df["qty"] = np.ceil(df["qty"]).astype(int)

    # удаляем плохие габариты/qty
    bad_numeric = df[["длина", "ширина", "высота", "qty"]].isna().any(axis=1)
    df = df.loc[~bad_numeric].copy()

    # неположительные габариты/qty или отрицательный вес
    bad_nonpos = (df[["длина", "ширина", "высота"]] <= 0).any(axis=1) | (df["qty"] <= 0) | (df["вес"] < 0)
    df = df.loc[~bad_nonpos].copy()

    # размножаем по qty
    df = df.loc[df.index.repeat(df["qty"])].reset_index(drop=True)

    # row_id
    df["row_id"] = df.index

    missing_weight_count = int(df["weight_missing"].sum()) if len(df) else 0
    return df, missing_weight_count


def fits_item_3d(item: Item, truck: TruckSpec, allow_rotate_floor: bool = True) -> bool:
    if item.H > truck.eff_H:
        return False
    if item.L <= truck.eff_L and item.W <= truck.eff_W:
        return True
    if allow_rotate_floor and item.W <= truck.eff_L and item.L <= truck.eff_W:
        return True
    return False


def _choose_orientation_for_shelf(
    item: Item,
    remaining_len: int,
    shelf_remaining_w: int,
    allow_rotate_floor: bool
) -> Optional[Tuple[int, int, bool]]:
    candidates = []
    if item.L <= remaining_len and item.W <= shelf_remaining_w:
        candidates.append((item.L, item.W, False, remaining_len - item.L))
    if allow_rotate_floor and item.W <= remaining_len and item.L <= shelf_remaining_w:
        candidates.append((item.W, item.L, True, remaining_len - item.W))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[3], x[1]))
    placed_L, placed_W, rotated, _ = candidates[0]
    return placed_L, placed_W, rotated


def pack_one_truck_shelf(
    items: List[Item],
    truck: TruckSpec,
    allow_rotate_floor: bool = True,
    sort_by: str = "area_desc",
    use_payload_constraint: bool = True
) -> Tuple[List[Item], List[Item], List[Placement], Dict]:
    if sort_by == "area_desc":
        items_sorted = sorted(items, key=lambda it: it.L * it.W, reverse=True)
    elif sort_by == "max_side_desc":
        items_sorted = sorted(items, key=lambda it: max(it.L, it.W), reverse=True)
    else:
        items_sorted = items[:]

    remaining = items_sorted[:]
    placed: List[Item] = []
    placements: List[Placement] = []

    weight_left = truck.max_payload
    used_w = 0
    shelf_no = -1
    progress_made = True

    while remaining and used_w < truck.eff_W and progress_made:
        progress_made = False
        shelf_no += 1

        shelf_y = used_w
        shelf_remaining_w = truck.eff_W - used_w
        shelf_remaining_len = truck.eff_L
        shelf_height_w = 0

        i = 0
        while i < len(remaining):
            item = remaining[i]

            if use_payload_constraint and (item.weight > weight_left):
                i += 1
                continue

            orient = _choose_orientation_for_shelf(
                item=item,
                remaining_len=shelf_remaining_len,
                shelf_remaining_w=shelf_remaining_w,
                allow_rotate_floor=allow_rotate_floor
            )
            if orient is None:
                i += 1
                continue

            placed_L, placed_W, rotated = orient
            x = truck.eff_L - shelf_remaining_len
            y = shelf_y

            placements.append(
                Placement(
                    truck_no=0,
                    shelf_no=shelf_no,
                    x=x,
                    y=y,
                    placed_L=placed_L,
                    placed_W=placed_W,
                    rotated=rotated,
                    item_idx=item.idx,
                    item_name=item.name,
                    item_weight=item.weight,
                    weight_missing=item.weight_missing
                )
            )

            shelf_remaining_len -= placed_L
            shelf_height_w = max(shelf_height_w, placed_W)

            if use_payload_constraint:
                weight_left -= item.weight

            placed.append(item)
            remaining.pop(i)
            progress_made = True

            if shelf_remaining_len <= 0:
                break

        if shelf_height_w == 0:
            break

        used_w += shelf_height_w

    truck_stats = {
        "placed_count": len(placed),
        "remaining_count": len(remaining),
        "used_width_mm": int(used_w),
        "eff_width_mm": int(truck.eff_W),
        "payload_used_kg": float(truck.max_payload - weight_left) if use_payload_constraint else np.nan,
        "payload_limit_kg": float(truck.max_payload) if use_payload_constraint else np.nan,
        "use_payload_constraint": bool(use_payload_constraint),
    }

    return placed, remaining, placements, truck_stats


def pack_many_trucks_shelf(
    df_norm: pd.DataFrame,
    truck: TruckSpec,
    allow_rotate_floor: bool = True,
    sort_by: str = "area_desc",
    use_payload_constraint: bool = True
) -> Dict:
    items_ok: List[Item] = []
    items_bad: List[Item] = []

    for _, r in df_norm.iterrows():
        item = Item(
            idx=int(r["row_id"]),
            name=str(r["наименование"]),
            L=int(r["длина"]),
            W=int(r["ширина"]),
            H=int(r["высота"]),
            weight=float(r["вес"]),
            weight_missing=bool(r["weight_missing"])
        )
        if fits_item_3d(item, truck, allow_rotate_floor=allow_rotate_floor):
            items_ok.append(item)
        else:
            items_bad.append(item)

    truck_no = 0
    remaining = items_ok[:]
    all_placements: List[Placement] = []
    per_truck_stats: List[Dict] = []

    while remaining:
        placed, remaining_next, placements, stats = pack_one_truck_shelf(
            items=remaining,
            truck=truck,
            allow_rotate_floor=allow_rotate_floor,
            sort_by=sort_by,
            use_payload_constraint=use_payload_constraint
        )
        if len(placed) == 0:
            break

        for p in placements:
            p.truck_no = truck_no

        all_placements.extend(placements)
        stats["truck_no"] = truck_no
        per_truck_stats.append(stats)

        remaining = remaining_next
        truck_no += 1

    placements_df = pd.DataFrame([p.__dict__ for p in all_placements])
    stats_df = pd.DataFrame(per_truck_stats)

    oversize_df = pd.DataFrame([{
        "row_id": it.idx,
        "наименование": it.name,
        "длина": it.L,
        "ширина": it.W,
        "высота": it.H,
        "вес": it.weight,
        "weight_missing": it.weight_missing
    } for it in items_bad])

    not_packed_df = pd.DataFrame([{
        "row_id": it.idx,
        "наименование": it.name,
        "длина": it.L,
        "ширина": it.W,
        "высота": it.H,
        "вес": it.weight,
        "weight_missing": it.weight_missing
    } for it in remaining])

    trucks_used = int(stats_df["truck_no"].max() + 1) if not stats_df.empty else 0

    return {
        "truck_name": truck.name,
        "reserve_len": truck.reserve_len,
        "reserve_wid": truck.reserve_wid,
        "eff_L": truck.eff_L,
        "eff_W": truck.eff_W,
        "eff_H": truck.eff_H,
        "allow_rotate_floor": allow_rotate_floor,
        "sort_by": sort_by,
        "trucks_used": trucks_used,
        "placements_df": placements_df,
        "truck_stats_df": stats_df,
        "oversize_df": oversize_df,
        "not_packed_df": not_packed_df,
        "use_payload_constraint": bool(use_payload_constraint),
    }


def run_calc(df_raw: pd.DataFrame, trucks: List[TruckSpec]) -> Dict:
    """
    Полностью повторяет логику блока F/G/H в твоем Colab коде:
      - нормализация
      - use_payload = (missing_weight_count == 0)
      - прогон по всем типам транспорта
      - выбор best по (машин_нужно, не_уложилось_шт, негабарит_шт)
      - метрики по грузу (длинный/широкий/тяжелый по известным)
    """
    df, missing_weight_count = normalize_input(df_raw)
    use_payload = (missing_weight_count == 0)

    pack_results = []
    res_by_truck = {}

    for t in trucks:
        res = pack_many_trucks_shelf(
            df_norm=df,
            truck=t,
            allow_rotate_floor=True,
            sort_by="area_desc",
            use_payload_constraint=use_payload
        )
        res_by_truck[t.name] = res
        pack_results.append({
            "тип_транспорта": res["truck_name"],
            "эфф_длина_мм": res["eff_L"],
            "эфф_ширина_мм": res["eff_W"],
            "высота_мм": res["eff_H"],
            "запас_длина_%": int(res["reserve_len"] * 100),
            "запас_ширина_%": int(res["reserve_wid"] * 100),
            "машин_нужно": res["trucks_used"],
            "негабарит_шт": int(len(res["oversize_df"])),
            "не_уложилось_шт": int(len(res["not_packed_df"])),
            "учет_грузоподъемности": res["use_payload_constraint"],
        })

    summary_df = pd.DataFrame(pack_results).sort_values(["машин_нужно", "не_уложилось_шт", "негабарит_шт"])

    best_truck_name = summary_df.iloc[0]["тип_транспорта"] if len(summary_df) else None
    best_res = res_by_truck[best_truck_name] if best_truck_name else None
    best_truck = next((t for t in trucks if t.name == best_truck_name), None)

    # доп метрики
    longest_len_mm = int(df["длина"].max()) if len(df) else 0
    widest_mm = int(df["ширина"].max()) if len(df) else 0

    df_known_w = df.loc[~df["weight_missing"]].copy() if len(df) else df
    if len(df_known_w):
        heaviest_row = df_known_w.loc[df_known_w["вес"].idxmax(), ["наименование", "длина", "ширина", "высота", "вес"]]
        heaviest_row = heaviest_row.to_dict()
        heaviest_weight = float(heaviest_row["вес"])
    else:
        heaviest_row = None
        heaviest_weight = 0.0

    count_gt_150 = int((df_known_w["вес"] > 150).sum()) if len(df_known_w) else 0
    count_gt_500 = int((df_known_w["вес"] > 500).sum()) if len(df_known_w) else 0

    return {
        "df_norm": df,
        "missing_weight_count": missing_weight_count,
        "use_payload": use_payload,
        "summary_df": summary_df,
        "best_truck": best_truck,
        "best_res": best_res,
        "metrics": {
            "total_items_after_qty": int(len(df)),
            "longest_len_mm": longest_len_mm,
            "widest_mm": widest_mm,
            "heaviest_row_known": heaviest_row,
            "heaviest_weight_known": heaviest_weight,
            "count_gt_150_known": count_gt_150,
            "count_gt_500_known": count_gt_500,
        }
    }


# ----------------------------
# FILE: app.py
# ----------------------------
import streamlit as st
import pandas as pd

from calc import TruckSpec, run_calc

st.set_page_config(page_title="Расчет транспорта (Shelf Packing)", layout="wide")
st.title("🚚 Расчет необходимого транспорта под перевозку грузов")

with st.expander("Поля Excel", expanded=True):
    st.markdown("""
**Минимум (обязательные):**
- `наименование`
- `длина` (мм)
- `ширина` (мм)
- `высота` (мм)
- `штабелируется` (`да`/любое другое)
- `количество` **или** `qty`

**Вес (`вес`)**:
- может отсутствовать или быть пустым → считаем `вес = 0`, `weight_missing = True`
- отрицательный вес → строка исключается из расчета

**Опционально:**
- `max_top_weight` (если `штабелируется="да"` и пусто → 50)
""")

st.sidebar.header("Типы транспорта (как в твоем финальном коде)")
st.sidebar.caption("Можно менять запасы прямо в списке ниже — если нужно, вынесу в UI.")

# ТВОИ ТИПЫ ТРАНСПОРТА (из финального Colab кода)
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

# Верхние метрики
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
st.dataframe(summary_df, use_container_width=True)

if best_truck is not None and best_res is not None:
    st.subheader("Итог")
    st.write(f"**Выбран транспорт:** {best_truck.name}")
    st.write(f"**Нужно машин:** {best_res['trucks_used']}")
    st.write(
        f"Эффективные размеры пола (с запасом): **{best_truck.eff_L} × {best_truck.eff_W} мм** | "
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
        st.dataframe(best_res["truck_stats_df"], use_container_width=True)
    else:
        st.write("Статистика по машинам: нет (ничего не уложилось)")

with tabs[1]:
    if best_res is not None:
        overs = best_res["oversize_df"][["наименование", "длина", "ширина", "высота", "вес", "weight_missing"]].copy()
        overs = overs.sort_values(["длина", "ширина", "вес"], ascending=False)
        st.dataframe(overs, use_container_width=True)

with tabs[2]:
    if best_res is not None:
        st.dataframe(best_res["placements_df"].head(30), use_container_width=True)

with tabs[3]:
    if best_res is not None:
        notp = best_res["not_packed_df"][["наименование", "длина", "ширина", "высота", "вес", "weight_missing"]].copy()
        notp = notp.sort_values(["длина", "ширина", "вес"], ascending=False)
        st.dataframe(notp, use_container_width=True)
