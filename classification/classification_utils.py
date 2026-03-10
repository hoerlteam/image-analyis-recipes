def get_class_mapping():
    return {
        0: 'EarlyS',
        1: 'G1G2',
        2: 'LateS',
        3: 'MidS',
        4: 'ambiguous'
    }

def get_class_colors():
    # TODO: make more general!
    return {
        'EarlyS': (255, 0, 0),
        'G1G2': (0, 255, 0),
        'LateS': (0, 0, 255),
        'MidS': (255, 255, 0),
        'ambiguous': (255, 255, 255)
    }

def create_result_table(classes):
    return {
        "cell_id": [],
        "max_class": [],
        "max_class_name": [],
        "img_file": [],
        "mask_file": [],
    } | {f'prob_{cls}': []  for cls in classes}
