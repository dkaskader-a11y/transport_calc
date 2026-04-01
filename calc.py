# calc.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ============================================================
# Validate + Normalize
# ============================================================

def normalize_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["наименование", "длина", "ширина", "высота", "штабелируется"]
    missing_cols = set(required_columns) - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

    if "вес" not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["вес"] = np.nan

    df = df_raw.copy()

    if "qty" not in df.columns and "количество" in df.columns:
        df["qty"] = df["количество"]
    elif "qty" not in df.columns and "количество" not in df.columns:
        df["qty"] = 1

    num_cols_strict = ["длина", "ширина", "высота", "qty"]
    for col in num_cols_strict:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["вес"] = (
        df["вес"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["вес"] = pd.to_numeric(df["вес"], errors="coerce")

    df["weight_missing"] = df["вес"].isna()
    df.loc[df["weight_missing"], "вес"] = 0.0

    df["штабелируется"] = df["штабелируется"].astype(str).str.strip().str.lower()
    df["штабелируется"] = np.where(df["штабелируется"].eq("да"), "да", "нет")

    if "max_top_weight" not in df.columns:
        df["max_top_weight"] = np.nan

    df["max_top_weight"] = pd.to_numeric(
        df["max_top_weight"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df.loc[(df["штабелируется"] == "да") & (df["max_top_weight"].isna()), "max_top_weight"] = 50.0
    df["max_top_weight"] = df["max_top_weight"].fillna(0.0)

    df["qty"] = df["qty"].fillna(1)
    df["qty"] = np.ceil(df["qty"]).astype(int)

    bad_numeric = df[["длина", "ширина", "высота", "qty"]].isna().any(axis=1)
    df = df.loc[~bad_numeric].copy()

    bad_nonpos = (
        (df[["длина", "ширина", "высота"]] <= 0).any(axis=1)
        | (df["qty"] <= 0)
        | (df["вес"] < 0)
    )
    df = df.loc[~bad_nonpos].copy()

    df = df.loc[df.index.repeat(df["qty"])].reset_index(drop=True)
    df["row_id"] = df.index
    return df


# ============================================================
# Models
# ============================================================

@dataclass
class TruckSpec:
    name: str
    L: int
    W: int
    H: int
    max_payload: float
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
        return int(self.H)


@dataclass
class Item:
    idx: int
    name: str
    L: int
    W: int
    H: int
    weight: float
    weight_missing: bool
    stackable: bool
    max_top_weight: float


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
    stack_level: int
    stacked_on_item_idx: Optional[int]
    stacked_on_item_name: Optional[str]


# ============================================================
# Geometry helpers
# ============================================================

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


def _choose_orientation_on_top(
    item: Item,
    top_L: int,
    top_W: int,
) -> Optional[Tuple[int, int, bool]]:
    candidates = []
    if item.L <= top_L and item.W <= top_W:
        candidates.append((item.L, item.W, False, (top_L - item.L) * (top_W - item.W)))
    if item.W <= top_L and item.L <= top_W:
        candidates.append((item.W, item.L, True, (top_L - item.W) * (top_W - item.L)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[3])
    placed_L, placed_W, rotated, _ = candidates[0]
    return placed_L, placed_W, rotated


def _floor_sort_key(item: Item, sort_by: str = "area_desc"):
    area = item.L * item.W
    max_side = max(item.L, item.W)

    if not item.stackable:
        priority = 0
    else:
        if area >= 1_000_000:
            priority = 1
        else:
            priority = 2

    if sort_by == "max_side_desc":
        size_key = max_side
    else:
        size_key = area

    return (priority, -size_key, -max_side)


def _stack_sort_key(item: Item):
    return (-(item.L * item.W), -item.weight, -item.H)


# ============================================================
# Single truck packing
# ============================================================

def pack_one_truck_shelf(
    items: List[Item],
    truck: TruckSpec,
    allow_rotate_floor: bool = True,
    sort_by: str = "area_desc",
    use_payload_constraint: bool = True
) -> Tuple[List[Item], List[Item], List[Placement], Dict]:
    items_sorted = sorted(items, key=lambda it: _floor_sort_key(it, sort_by=sort_by))

    remaining = items_sorted[:]
    placed: List[Item] = []
    placements: List[Placement] = []

    weight_left = truck.max_payload
    used_w = 0
    shelf_no = -1
    progress_made = True

    stacks: List[Dict] = []

    # Фаза 1: укладка по полу
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
                    weight_missing=item.weight_missing,
                    stack_level=0,
                    stacked_on_item_idx=None,
                    stacked_on_item_name=None,
                )
            )

            if item.stackable:
                stacks.append({
                    "x": x,
                    "y": y,
                    "shelf_no": shelf_no,
                    "top_L": placed_L,
                    "top_W": placed_W,
                    "total_height": item.H,
                    "remaining_caps": [float(item.max_top_weight)],
                    "top_item_idx": item.idx,
                    "top_item_name": item.name,
                    "stack_level": 0,
                })

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

    # Фаза 2: штабелирование
    stacked_items_count = 0
    if remaining and stacks:
        remaining_non_stackable = [it for it in remaining if not it.stackable]
        remaining_stackable = [it for it in remaining if it.stackable]
        remaining_stackable = sorted(remaining_stackable, key=_stack_sort_key)

        still_unplaced_stackable: List[Item] = []

        for item in remaining_stackable:
            if use_payload_constraint and (item.weight > weight_left):
                still_unplaced_stackable.append(item)
                continue

            best_stack_idx = None
            best_fit = None
            best_score = None

            for s_idx, stack in enumerate(stacks):
                # Высота учитывается здесь:
                if stack["total_height"] + item.H > truck.eff_H:
                    continue

                orient = _choose_orientation_on_top(
                    item=item,
                    top_L=stack["top_L"],
                    top_W=stack["top_W"],
                )
                if orient is None:
                    continue

                if any(cap < item.weight for cap in stack["remaining_caps"]):
                    continue

                placed_L, placed_W, rotated = orient
                waste = (stack["top_L"] * stack["top_W"]) - (placed_L * placed_W)
                score = (waste, stack["total_height"])

                if best_score is None or score < best_score:
                    best_score = score
                    best_stack_idx = s_idx
                    best_fit = (placed_L, placed_W, rotated)

            if best_stack_idx is None:
                still_unplaced_stackable.append(item)
                continue

            stack = stacks[best_stack_idx]
            placed_L, placed_W, rotated = best_fit

            placements.append(
                Placement(
                    truck_no=0,
                    shelf_no=stack["shelf_no"],
                    x=stack["x"],
                    y=stack["y"],
                    placed_L=placed_L,
                    placed_W=placed_W,
                    rotated=rotated,
                    item_idx=item.idx,
                    item_name=item.name,
                    item_weight=item.weight,
                    weight_missing=item.weight_missing,
                    stack_level=stack["stack_level"] + 1,
                    stacked_on_item_idx=stack["top_item_idx"],
                    stacked_on_item_name=stack["top_item_name"],
                )
            )

            stack["remaining_caps"] = [cap - item.weight for cap in stack["remaining_caps"]]
            stack["remaining_caps"].append(float(item.max_top_weight))

            stack["top_L"] = placed_L
            stack["top_W"] = placed_W
            stack["total_height"] += item.H
            stack["top_item_idx"] = item.idx
            stack["top_item_name"] = item.name
            stack["stack_level"] += 1

            if use_payload_constraint:
                weight_left -= item.weight

            placed.append(item)
            stacked_items_count += 1

        remaining = remaining_non_stackable + still_unplaced_stackable

    floor_placements = [p for p in placements if p.stack_level == 0]
    used_length_mm = max((p.x + p.placed_L for p in floor_placements), default=0)

    truck_stats = {
        "placed_count": len(placed),
        "remaining_count": len(remaining),
        "used_width_mm": int(used_w),
        "eff_width_mm": int(truck.eff_W),
        "used_length_mm": int(used_length_mm),
        "eff_length_mm": int(truck.eff_L),
        "stacked_items_count": int(stacked_items_count),
        "payload_used_kg": float(truck.max_payload - weight_left) if use_payload_constraint else np.nan,
        "payload_limit_kg": float(truck.max_payload) if use_payload_constraint else np.nan,
        "use_payload_constraint": bool(use_payload_constraint),
    }

    return placed, remaining, placements, truck_stats


# ============================================================
# Mixed fleet
# ============================================================

def _truck_sort_key(truck: TruckSpec) -> int:
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
                stackable=bool(r["штабелируется"] == "да"),
                max_top_weight=float(r["max_top_weight"]),
            )
        )
    return items


# ============================================================
# Main run
# ============================================================

def run_calc(df_raw: pd.DataFrame, trucks: List[TruckSpec]) -> Dict:
    df = normalize_input(df_raw)

    missing_weight_count = int(df["weight_missing"].sum()) if len(df) else 0
    use_payload = (missing_weight_count == 0)

    if not trucks:
        raise ValueError("Не выбран ни один тип транспорта.")

    trucks_small_to_big = sorted(trucks, key=_truck_sort_key)
    trucks_big_to_small = sorted(trucks, key=_truck_sort_key, reverse=True)

    all_items = _items_from_df(df)

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
        chosen_remaining: Optional[List[Item]] = None
        chosen_placements: Optional[List[Placement]] = None
        chosen_stats: Optional[Dict] = None

        # Пробуем закрыть остаток самым маленьким типом
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
                chosen_remaining = rem
                chosen_placements = placements
                chosen_stats = stats
                break

        # Иначе берём тип, который увозит максимум мест за один рейс
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
                break

            chosen = best_t
            _, chosen_remaining, chosen_placements, chosen_stats = best_bundle

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
        "row_id": it.idx,
        "наименование": it.name,
        "длина": it.L,
        "ширина": it.W,
        "высота": it.H,
        "вес": it.weight,
        "weight_missing": it.weight_missing,
        "штабелируется": "да" if it.stackable else "нет",
        "max_top_weight": it.max_top_weight,
    } for it in oversize_items])

    not_packed_df = pd.DataFrame([{
        "row_id": it.idx,
        "наименование": it.name,
        "длина": it.L,
        "ширина": it.W,
        "высота": it.H,
        "вес": it.weight,
        "weight_missing": it.weight_missing,
        "штабелируется": "да" if it.stackable else "нет",
        "max_top_weight": it.max_top_weight,
    } for it in not_packed_items])

    longest_len_mm = int(df["длина"].max()) if len(df) else 0
    widest_mm = int(df["ширина"].max()) if len(df) else 0
    tallest_mm = int(df["высота"].max()) if len(df) else 0

    df_known_w = df.loc[~df["weight_missing"]].copy() if len(df) else df
    if len(df_known_w):
        heaviest_row = df_known_w.loc[
            df_known_w["вес"].idxmax(),
            ["наименование", "длина", "ширина", "высота", "вес"]
        ]
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
