# calc.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# ============================================================
# Normalize (как в финальном Colab коде)
# ============================================================

def normalize_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["наименование", "длина", "ширина", "высота", "штабелируется"]
    missing_cols = set(required_columns) - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

    # "вес" не обязательный
    if "вес" not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["вес"] = np.nan

    df = df_raw.copy()

    # qty
    if "qty" not in df.columns and "количество" in df.columns:
        df["qty"] = df["количество"]
    elif "qty" not in df.columns and "количество" not in df.columns:
        df["qty"] = 1

    # габариты/qty — строго
    num_cols_strict = ["длина", "ширина", "высота", "qty"]
    for col in num_cols_strict:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # вес — допускаем пустой
    df["вес"] = (
        df["вес"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["вес"] = pd.to_numeric(df["вес"], errors="coerce")

    df["weight_missing"] = df["вес"].isna()
    df.loc[df["weight_missing"], "вес"] = 0.0

    # штабелируется -> да/нет
    df["штабелируется"] = df["штабелируется"].astype(str).str.strip().str.lower()
    df["штабелируется"] = np.where(df["штабелируется"].eq("да"), "да", "нет")

    # max_top_weight (опционально)
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

    # убираем строки с некорректными габаритами/qty
    bad_numeric = df[["длина", "ширина", "высота", "qty"]].isna().any(axis=1)
    df = df.loc[~bad_numeric].copy()

    # убираем строки с неположительными значениями + отрицательным весом
    bad_nonpos = (df[["длина", "ширина", "высота"]] <= 0).any(axis=1) | (df["qty"] <= 0) | (df["вес"] < 0)
    df = df.loc[~bad_nonpos].copy()

    # размножаем по qty
    df = df.loc[df.index.repeat(df["qty"])].reset_index(drop=True)

    # row_id
    df["row_id"] = df.index
    return df


# ============================================================
# Packing (как в финальном Colab коде)
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
            weight_missing=bool(r["weight_missing"]),
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


# ============================================================
# RUN (как в финальном Colab коде)
# ============================================================

def run_calc(df_raw: pd.DataFrame, trucks: List[TruckSpec]) -> Dict:
    df = normalize_input(df_raw)

    missing_weight_count = int(df["weight_missing"].sum()) if len(df) else 0
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

    # метрики по грузам
    longest_len_mm = int(df["длина"].max()) if len(df) else 0
    widest_mm = int(df["ширина"].max()) if len(df) else 0

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
