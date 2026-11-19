
# Normal inflow while the supply valve is closed, measured from the
# leak-free training scenarios (see inspect_data.py: about 15.6 L/s).
NORMAL_OFF_SUPPLY_FLOW_LPS = 16.0


def estimate_water_loss(inflow_history: list[float], valve_open: bool):
    """
    Water lost to the leak, estimated only while supply is off.
    During supply hours households are filling their tanks, so extra
    inflow cannot be separated from normal use.
    """
    if valve_open:
        return None

    recent_inflow = sum(inflow_history[-4:]) / len(inflow_history[-4:])
    loss_lps = max(0.0, recent_inflow - NORMAL_OFF_SUPPLY_FLOW_LPS)

    return {
        "estimated_loss_lps": round(loss_lps, 1),
        "estimated_loss_kl_per_day": round(loss_lps * 86.4, 0),
    }


def analyze_leak(leak_probability: float, inflow_history: list[float], valve_open: bool) -> dict:
    """
    Turn the model's leak probability into a status and
    a recommendation for the water utility.
    """

    leak_probability = float(leak_probability)

    if not 0 <= leak_probability <= 1:
        raise ValueError("Leak probability must be between 0 and 1")

    if leak_probability < 0.3:
        status = "NO_LEAK"
        recommendation = (
            "Inflow matches normal use for this point in the "
            "supply cycle. No action is required."
        )

    elif leak_probability < 0.6:
        status = "WATCH"
        recommendation = (
            "Inflow is slightly unusual. Check again during "
            "the next off-supply period."
        )

    elif leak_probability < 0.85:
        status = "LIKELY_LEAK"
        recommendation = (
            "Schedule a leak survey (acoustic listening) in this "
            "zone within 48 hours."
        )

    else:
        status = "CRITICAL_LEAK"
        recommendation = (
            "Send a field crew now and consider isolating the "
            "zone before the next supply cycle."
        )

    return {
        "leak_probability": round(leak_probability, 3),
        "status": status,
        "recommendation": recommendation,
        "water_loss": estimate_water_loss(inflow_history, valve_open),
    }
