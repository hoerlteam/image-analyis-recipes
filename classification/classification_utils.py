def get_class_mapping():
    return {
        0: 'EarlyS',
        1: 'G1G2',
        2: 'LateS',
        3: 'MidS',
        4: 'ambiguous'
    }

def get_class_colors():
    return {
        'EarlyS': (255, 0, 0),
        'G1G2': (0, 255, 0),
        'LateS': (0, 0, 255),
        'MidS': (255, 255, 0),
        'ambiguous': (255, 255, 255)
    }

def create_result_table():
    return {
        "cell_id": [],
        "prob_early": [],
        "prob_g1g2": [],
        "prob_late": [],
        "prob_mid": [],
        "prob_ambiguous": [],
        "max_class": [],
        "max_class_name": []
    }
