ALL_TRUCKS = [
    TruckSpec(name="Фура 82м3", L=13600, W=2450, H=2600, max_payload=20000, reserve_len=0.0, reserve_wid=0.00),
    TruckSpec(name="Фура 82м3 запас", L=13600, W=2450, H=2600, max_payload=20000, reserve_len=0.10, reserve_wid=0.05),
    TruckSpec(name="10т", L=7000, W=2400, H=2400, max_payload=10000, reserve_len=0.10, reserve_wid=0.05),
    TruckSpec(name="5т", L=6000, W=2400, H=2400, max_payload=5000, reserve_len=0.10, reserve_wid=0.10),
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
