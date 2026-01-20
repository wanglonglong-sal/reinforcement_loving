def straight_config(render: bool = True) -> dict:
    return {
        "map": "S",
        "traffic_density": 0.0,
        "num_scenarios": 1,
        "start_seed": 0,

        "use_render": render,
        "manual_control": False,
    }