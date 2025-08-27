# -*- coding: utf-8 -*-
SEED = 42
TARGET = "cycles_to_80pct_SOH"
GROUP_COL = "lot_id"

NUM_KEPT = [
    "cap_ret_100", "initial_DCIR_mOhm", "energy_density",
    "early_deg_slope", "deltaT_50", "deltaV_50", "safety_flag"
]
CAT_COLS = [
    "chemistry","form_factor","electrolyte_vendor",
    "formation_recipe","line_id","separator_type","equipment_id"
]