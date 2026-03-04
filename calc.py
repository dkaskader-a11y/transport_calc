# calc.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ============================================================
# C) Validate + Normalize (как в твоем финальном Colab коде)
# ============================================================

def normalize_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["наименование", "длина", "ширина", "высота", "штабелируется"]
    missing_cols = set(required_columns) - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

    # "вес" НЕ обязательный
    if "вес" not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["вес"] = np.nan

    df = df_raw.copy()

    # C1) qty
    if "qty" not in df.columns and "количество" in df.columns:
        df["qty"] = df["количество"]
    elif "qty" not in df.columns and "количество" not in df.columns:
        df["qty"] = 1

    # C2) габариты/qty — строго
    num_cols_strict = ["длина", "ширина", "высота", "qty"]
    for col in num_cols_strict:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # C3) вес — допускаем пустой
    df["вес"] = (
        df["вес"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["вес"] = pd.to_numeric(df["вес"], errors="coerce")

    df["weight_missing"] = df["вес"].isna()
    df.loc[df["weight_missing"], "вес"] = 0.0

    # C4) штабелируется -> "да"/"нет"
    df["штабелируется"] = df["штабелируется"].astype(str).str.strip().str.lower()
    df["штабелируется"] = np.where(df["штабелируется"].eq("да"), "да", "нет")

    # C5) max_top_weight (опционально; дефолт 50 для штабелируемых)
    if "max_top_weight" not in df.columns:
        df["max_top_weight"] = np.nan

    df["max_top_weight"] = pd.to_numeric(
        df["max_top_weight"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df.loc[(df["штабелируется"] == "да") & (df["max_top_weight"].isna()), "max_top_weight"] = 50.0

    # C6) qty int
    df["qty"] = df["qty"].fillna(1)
    df["qty"] = np.ceil(df["qty"]).astype(int)

    # C7) убираем строки с некорректными габаритами/qty
    bad_numeric = df[["длина", "ширина", "высота", "qty"]].isna().any(axis=1)
    df = df.loc[~bad_numeric].copy()

    # C8) убираем строки с неположительными значениями + отрицательный вес
    bad_nonpos = (df[["длина", "ширина", "высота"]] <= 0).any(axis=1) | (df["qty"] <= 0) | (df["вес"] < 0)
    df = df.loc[~bad_nonpos].copy()

    # C9) размножаем по qty
    df = df.loc[df.index.repeat(df["qty"])].reset_index(drop=True)

    # C10) row_id
    df["row_id"] = df.index
    return df


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
    reserve_len: float = 0.15
    reserve_wid: float = 0.10

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
    """
    use_payload_constraint:
      - True  -> учитываем грузоподъемность
      - False -> игнорируем ограничение по весу (если есть неизвестные веса)
    """
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


# ============================================================
# Mixed fleet: логика подбора "сверху вниз, остаток вниз"
# ============================================================

def _truck_sort_key(truck: TruckSpec) -> int:
    # "размер" транспорта по площади пола
    return truck.eff_L * truck.eff_W


def _items_from_df(df: pd.DataFrame) -> List[Item]:
    items: List[Item] = []
    for _, r in df.iterrows():
        items.append(
            Item(
                idx=int(r["row_id"]),
                name=str(r["наименование"]),
                L=int(r["длина"]),
                W=int(r["ширина"]),
                H=int(r["высота"]),
                weight=float(r["вес"]),
                weight_missing=bool(r["weight_missing"]),
            )
        )
    return items


# ============================================================
# RUN: основной расчет (СМЕШАННЫЙ ПАРК)
# ============================================================

def run_calc(df_raw: pd.DataFrame, trucks: List[TruckSpec]) -> Dict:
    """
    Новая логика подбора:
      - если есть weight_missing -> use_payload=False
      - подбираем смешанный парк:
          1) пробуем закрыть остаток самым маленьким типом
          2) если нельзя -> берем тип, который увозит максимум мест за 1 машину
          3) повторяем, и каждый раз пытаемся "опуститься" в меньшую категорию
    """
    df = normalize_input(df_raw)

    missing_weight_count = int(df["weight_missing"].sum()) if len(df) else 0
    use_payload = (missing_weight_count == 0)

    if not trucks:
        raise ValueError("Не выбран ни один тип транспорта.")

    trucks_small_to_big = sorted(trucks, key=_truck_sort_key)
    trucks_big_to_small = sorted(trucks, key=_truck_sort_key, reverse=True)

    all_items = _items_from_df(df)

    # Негабарит (глобально): не влезает ни в один выбранный тип
    oversize_items: List[Item] = []
    packable_items: List[Item] = []
    for it in all_items:
        fits_any = any(fits_item_3d(it, t, allow_rotate_floor=True) for t in trucks_big_to_small)
        if fits_any:
            packable_items.append(it)
        else:
            oversize_items.append(it)

    remaining = packable_items[:]
    fleet_rows: List[Dict] = []
    per_truck_stats: List[Dict] = []
    all_placements: List[Dict] = []

    truck_no = 0

    while remaining:
        chosen: Optional[TruckSpec] = None
        chosen_placed: Optional[List[Item]] = None
        chosen_remaining: Optional[List[Item]] = None
        chosen_placements: Optional[List[Placement]] = None
        chosen_stats: Optional[Dict] = None

        # 1) пытаемся уместить весь остаток самым маленьким типом
        for t in trucks_small_to_big:
            placed, rem, placements, stats = pack_one_truck_shelf(
                items=remaining,
                truck=t,
                allow_rotate_floor=True,
                sort_by="area_desc",
                use_payload_constraint=use_payload
            )
            if len(placed) == len(remaining) and len(placed) > 0:
                chosen = t
                chosen_placed = placed
                chosen_remaining = rem
                chosen_placements = placements
                chosen_stats = stats
                break

        # 2) иначе выбираем тип, который увозит максимум мест
        if chosen is None:
            best_t: Optional[TruckSpec] = None
            best_bundle = None
            best_count = -1

            for t in trucks_big_to_small:
                placed, rem, placements, stats = pack_one_truck_shelf(
                    items=remaining,
                    truck=t,
                    allow_rotate_floor=True,
                    sort_by="area_desc",
                    use_payload_constraint=use_payload
                )
                if len(placed) > best_count:
                    best_count = len(placed)
                    best_t = t
                    best_bundle = (placed, rem, placements, stats)

            if best_t is None or best_bundle is None or best_count <= 0:
                # Редкий случай: не можем уложить ни одной позиции из remaining
                break

            chosen = best_t
            chosen_placed, chosen_remaining, chosen_placements, chosen_stats = best_bundle

        # фиксируем одну машину
        fleet_rows.append({"truck_no": truck_no, "тип_транспорта": chosen.name})

        for p in chosen_placements:
            p.truck_no = truck_no
            all_placements.append({**p.__dict__, "truck_name": chosen.name})

        stats_row = dict(chosen_stats)
        stats_row["truck_no"] = truck_no
        stats_row["truck_name"] = chosen.name
        per_truck_stats.append(stats_row)

        remaining = chosen_remaining
        truck_no += 1

    not_packed_items = remaining

    fleet_df = pd.DataFrame(fleet_rows)
    if not fleet_df.empty:
        fleet_summary_df = (
            fleet_df.groupby("тип_транспорта", as_index=False)
            .agg(машин_нужно=("truck_no", "count"))
            .sort_values("машин_нужно", ascending=False)
        )
        total_trucks = int(fleet_df["truck_no"].nunique())
    else:
        fleet_summary_df = pd.DataFrame(columns=["тип_транспорта", "машин_нужно"])
        total_trucks = 0

    placements_df = pd.DataFrame(all_placements)
    truck_stats_df = pd.DataFrame(per_truck_stats)

    oversize_df = pd.DataFrame([{
        "row_id": it.idx, "наименование": it.name, "длина": it.L, "ширина": it.W, "высота": it.H,
        "вес": it.weight, "weight_missing": it.weight_missing
    } for it in oversize_items])

    not_packed_df = pd.DataFrame([{
        "row_id": it.idx, "наименование": it.name, "длина": it.L, "ширина": it.W, "высота": it.H,
        "вес": it.weight, "weight_missing": it.weight_missing
    } for it in not_packed_items])

    # Метрики по грузам (+ самый высокий)
    longest_len_mm = int(df["длина"].max()) if len(df) else 0
    widest_mm = int(df["ширина"].max()) if len(df) else 0
    tallest_mm = int(df["высота"].max()) if len(df) else 0

    df_known_w = df.loc[~df["weight_missing"]].copy() if len(df) else df
    if len(df_known_w):
        heaviest_row = df_known_w.loc[df_known_w["вес"].idxmax(), ["наименование", "длина", "ширина", "высота", "вес"]]
        heaviest_weight = float(heaviest_row["вес"])
        heaviest_row = heaviest_row.to_dict()
    else:
        heaviest_row = None
        heaviest_weight = 0.0

    count_gt_150 = int((df_known_w["вес"] > 150).sum()) if len(df_known_w) else 0
    count_gt_500 = int((df_known_w["вес"] > 500).sum()) if len(df_known_w) else 0

    return {
        "df_norm": df,
        "missing_weight_count": missing_weight_count,
        "use_payload": use_payload,

        "fleet_summary_df": fleet_summary_df,
        "total_trucks": total_trucks,

        "placements_df": placements_df,
        "truck_stats_df": truck_stats_df,
        "oversize_df": oversize_df,
        "not_packed_df": not_packed_df,

        "metrics": {
            "total_items_after_qty": int(len(df)),
            "longest_len_mm": longest_len_mm,
            "widest_mm": widest_mm,
            "tallest_mm": tallest_mm,
            "heaviest_row_known": heaviest_row,
            "heaviest_weight_known": heaviest_weight,
            "count_gt_150_known": count_gt_150,
            "count_gt_500_known": count_gt_500,
        }
    }
