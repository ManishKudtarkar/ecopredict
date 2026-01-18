def assign_risk(score):
    if score < 0.3:
        return "Low"
    elif score < 0.6:
        return "Medium"
    return "High"
