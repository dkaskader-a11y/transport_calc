def _truck_sort_key(truck: TruckSpec) -> int:
    # Сортируем по "размеру" (эффективная площадь пола)
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


def run_calc(df_raw: pd.DataFrame, trucks: List[TruckSpec]) -> Dict:
    """
    Новая логика:
      - если есть weight_missing -> use_payload=False
      - считаем НЕ 'сколько фур нужно', а подбираем смешанный парк:
          * пробуем закрыть остаток самым маленьким типом
          * если нельзя -> берем тип, который увозит максимум мест за 1 машину
    """
    df = normalize_input(df_raw)

    missing_weight_count = int(df["weight_missing"].sum()) if len(df) else 0
    use_payload = (missing_weight_count == 0)

    # Если список типов пустой — ошибка (в UI это тоже проверим)
    if not trucks:
        raise ValueError("Не выбран ни один тип транспорта.")

    # Сортировки типов
    trucks_small_to_big = sorted(trucks, key=_truck_sort_key)
    trucks_big_to_small = sorted(trucks, key=_truck_sort_key, reverse=True)

    # Грузы как Items
    all_items = _items_from_df(df)

    # Негабарит "глобально": то, что не влезает ни в один выбранный тип
    oversize_items: List[Item] = []
    packable_items: List[Item] = []
    for it in all_items:
        fits_any = any(fits_item_3d(it, t, allow_rotate_floor=True) for t in trucks_big_to_small)
        if fits_any:
            packable_items.append(it)
        else:
            oversize_items.append(it)

    # Основной цикл смешанного парка
    remaining = packable_items[:]
    fleet_rows: List[Dict] = []
    per_truck_stats: List[Dict] = []
    all_placements: List[Dict] = []

    truck_no = 0

    while remaining:
        # 1) Пытаемся закрыть весь остаток самым маленьким типом
        chosen = None
        chosen_placed = None
        chosen_remaining = None
        chosen_placements = None
        chosen_stats = None

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
                break  # самый маленький, который уместил всё

        # 2) Если не нашли — берём тип, который увозит максимум мест за 1 машину
        if chosen is None:
            best_t = None
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
                # Теоретически может случиться из-за ограничений по payload (когда use_payload=True)
                # или из-за очень неудачной комбинации, но это редкость.
                break

            chosen = best_t
            chosen_placed, chosen_remaining, chosen_placements, chosen_stats = best_bundle

        # Фиксируем одну машину chosen-типа
        fleet_rows.append({"truck_no": truck_no, "тип_транспорта": chosen.name})

        # Проставим номер машины + добавим truck_name в placements
        for p in chosen_placements:
            p.truck_no = truck_no
            all_placements.append({
                **p.__dict__,
                "truck_name": chosen.name,
            })

        chosen_stats = dict(chosen_stats)
        chosen_stats["truck_no"] = truck_no
        chosen_stats["truck_name"] = chosen.name
        per_truck_stats.append(chosen_stats)

        remaining = chosen_remaining
        truck_no += 1

    # То, что не уложилось (после всех попыток)
    not_packed_items = remaining

    # DataFrames
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
        "row_id": it.idx, "наименование": it.name, "длина": it.L, "ширина": it.W, "высота": it.H, "вес": it.weight, "weight_missing": it.weight_missing
    } for it in oversize_items])

    not_packed_df = pd.DataFrame([{
        "row_id": it.idx, "наименование": it.name, "длина": it.L, "ширина": it.W, "высота": it.H, "вес": it.weight, "weight_missing": it.weight_missing
    } for it in not_packed_items])

    # Метрики по грузам (добавили "самый высокий")
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

        # Новые ключи под “смешанный парк”
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
