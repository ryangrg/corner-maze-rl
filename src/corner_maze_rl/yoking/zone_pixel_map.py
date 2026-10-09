"""Physical pixel-coordinate boundaries for each maze zone (0-21).

Extracted from the Raspberry Pi motion-tracking camera code
(MotionTrack_ZeroMQ_picam2_mwb.py). The camera produces a 240x240 pixel
image; ``return_zone(x, y)`` maps a pixel centroid to one of the 22 maze
zones. Zone 0 is the fallback for coordinates outside all defined regions.

Zone layout in the 240x240 tracking frame (x: 0 = south .. 239 = north;
y: 0 = west .. 239 = east; the rig's zone map is the ground truth):
    Wells:          1=SW, 5=NW, 17=SE, 21=NE
    Centre:         11
    Plus-maze arms: 7=W arm, 12=N arm, 10=S arm, 15=E arm (inner zones; the
                    outer arm zones are the junctions below)
    T-junctions:    3=W, 13=N, 9=S, 19=E (where each arm meets the ring)
    Ring segments:  2/4=W side, 8/16=N side, 6/14=S side, 18/20=E side
    0 = outside every zone (perimeter corners between a well and the ring)
"""


ZONE_PIXEL_BOUNDS = {
    # --- SW corner well (zone 1) ---
    # Triangular: 0<=x<=47, 0<=y<=47, x <= (47-y)
    1: {'x': (0, 47), 'y': (0, 47), 'constraint': 'x <= (47 - y)'},

    # --- W outer ring ---
    2: {'x': (43, 97),  'y': (0, 35)},
    3: {'x': (98, 138), 'y': (0, 35)},
    4: {'x': (139, 194), 'y': (0, 35)},

    # --- NW corner well (zone 5) ---
    # Triangular: 193<=x<=239, 0<=y<=47, (x-193) >= y
    5: {'x': (193, 239), 'y': (0, 47), 'constraint': '(x - 193) >= y'},

    # --- SW-S / W-arm corridors ---
    6:  {'x': (0, 34),   'y': (45, 100)},
    7:  {'x': (96, 138), 'y': (36, 96)},
    8:  {'x': (205, 239), 'y': (41, 97)},
    9:  {'x': (0, 34),   'y': (101, 139)},
    10: {'x': (35, 95),  'y': (103, 137)},
    11: {'x': (96, 143), 'y': (97, 142)},
    12: {'x': (144, 204), 'y': (98, 139)},
    13: {'x': (205, 239), 'y': (98, 139)},
    14: {'x': (0, 34),   'y': (140, 196)},
    15: {'x': (102, 137), 'y': (143, 204)},
    16: {'x': (205, 239), 'y': (140, 195)},

    # --- SE corner well (zone 17) ---
    # Triangular: 0<=x<=46, 194<=y<=239, x <= (y-194)
    17: {'x': (0, 46), 'y': (194, 239), 'constraint': 'x <= (y - 194)'},

    # --- S outer ring ---
    18: {'x': (43, 97),  'y': (205, 239)},
    19: {'x': (98, 139), 'y': (205, 239)},
    20: {'x': (140, 195), 'y': (205, 239)},

    # --- NE corner well (zone 21) ---
    # Triangular: 192<=x<=239, 192<=y<=239, (x-192) >= (239-y)
    21: {'x': (192, 239), 'y': (192, 239), 'constraint': '(x - 192) >= (239 - y)'},
}


def return_zone(x: int, y: int) -> int:
    """Map pixel coordinates (240x240 camera frame) to maze zone 0-21."""
    # SW well
    if 0 <= x <= 47 and 0 <= y <= 47 and x <= (47 - y):
        return 1
    elif 43 <= x <= 97 and 0 <= y <= 35:
        return 2
    elif 98 <= x <= 138 and 0 <= y <= 35:
        return 3
    elif 139 <= x <= 194 and 0 <= y <= 35:
        return 4
    # NW well
    elif 193 <= x <= 239 and 0 <= y <= 47 and (x - 193) >= y:
        return 5
    elif 0 <= x <= 34 and 45 <= y <= 100:
        return 6
    elif 96 <= x <= 138 and 36 <= y <= 96:
        return 7
    elif 205 <= x <= 239 and 41 <= y <= 97:
        return 8
    elif 0 <= x <= 34 and 101 <= y <= 139:
        return 9
    elif 35 <= x <= 95 and 103 <= y <= 137:
        return 10
    elif 96 <= x <= 143 and 97 <= y <= 142:
        return 11
    elif 144 <= x <= 204 and 98 <= y <= 139:
        return 12
    elif 205 <= x <= 239 and 98 <= y <= 139:
        return 13
    elif 0 <= x <= 34 and 140 <= y <= 196:
        return 14
    elif 102 <= x <= 137 and 143 <= y <= 204:
        return 15
    elif 205 <= x <= 239 and 140 <= y <= 195:
        return 16
    # SE well
    elif 0 <= x <= 46 and 194 <= y <= 239 and x <= (y - 194):
        return 17
    elif 43 <= x <= 97 and 205 <= y <= 239:
        return 18
    elif 98 <= x <= 139 and 205 <= y <= 239:
        return 19
    elif 140 <= x <= 195 and 205 <= y <= 239:
        return 20
    # NE well
    elif 192 <= x <= 239 and 192 <= y <= 239 and (x - 192) >= (239 - y):
        return 21
    else:
        return 0
