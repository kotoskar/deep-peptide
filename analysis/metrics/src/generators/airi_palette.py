"""AIRI brand palette, transcribed from article/assets/Colors.pdf.

Keys follow the swatch labels in that document. Use `LINE_CYCLE` for
line/marker series: it is ordered for maximum separation both in colour and,
where the caller also varies marker and dash, in grayscale.
"""

AIRI = {
    "A":   "#005A60",  # dark teal
    "B":   "#2FBEAD",  # teal
    "C":   "#8E5F44",  # brown
    "D1":  "#D1A684",  # tan
    "D2":  "#FDE74C",  # yellow
    "E1":  "#FE7D0E",  # orange
    "E21": "#E94262",  # red
    "E22": "#4A4A4A",  # dark grey
    "E3":  "#909393",  # mid grey
    "F1":  "#2DAAF0",  # blue
    "F2":  "#7565FF",  # violet
    "F3":  "#D458FB",  # magenta
}

# Ordered for series plots: neutral baseline first, then saturated accents.
LINE_CYCLE = [AIRI["E22"], AIRI["F1"], AIRI["E1"], AIRI["A"], AIRI["F2"], AIRI["E21"]]

INK  = AIRI["E22"]
GRID = "#DDDDDD"
