import json

import matplotlib as mpl
import numpy as np
from matplotlib import style

from ..constants import Constants as Const


class Visualizer:
    def __init__(self):
        style.use(Const.MATPLOTLIB_STYLE)
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
                "pgf.preamble": "\n".join(
                    [
                        "\\usepackage[utf8]{inputenc}",
                        "\\DeclareUnicodeCharacter{2212}{-}",
                    ]
                ),
                "font.size": Const.FONT_SIZE,
                "legend.fontsize": Const.LEGEND_FONT_SIZE,
                "legend.title_fontsize": Const.LEGEND_FONT_SIZE,
                "axes.labelsize": Const.AXIS_FONT_SIZE,
                "xtick.labelsize": Const.TICKS_FONT_SIZE,
                "ytick.labelsize": Const.TICKS_FONT_SIZE,
            }
        )
        mpl.rcParams["savefig.format"] = Const.OUT_FORMAT


class ConsoleColors:
    COLORS = [
        "\033[95m",
        "\033[94m",
        "\033[96m",
        "\033[92m",
        "\033[93m",
        "\033[91m",
        "\033[1m",
        "\033[4m",
    ]
    END_COLOR = "\033[0m"


def format_matrix(matrix, spacing=None, decimals=4):
    lines = []
    matrix = np.squeeze(matrix)
    matrix = np.atleast_2d(matrix)

    max_value = np.max(np.abs(matrix))
    if not spacing:
        if max_value > 0 and not np.isinf(max_value):
            spacing = max([int(np.log10(max_value)) + 5, 6])
        else:
            spacing = 6

    for row in matrix:
        line = ""
        for cell in row:
            if not np.isinf(np.abs(cell)):
                if cell == 0 or np.abs(int(cell) - cell) < 1e-12:
                    pattern = "{:>" + str(int(spacing)) + "}"
                    line = line + pattern.format(int(cell))
                else:
                    pattern = (
                        "{:>" + str(int(spacing)) + "." + str(int(decimals)) + "f}"
                    )
                    line = line + pattern.format(cell)
            else:
                pattern = "{:>" + str(int(spacing)) + "}"
                line = line + pattern.format(cell)

        lines.append(line)
    return lines


def print_matrix(matrix, name=None, spacing=None, decimals=4):
    if type(matrix) == np.ndarray:
        shape = matrix.shape
    else:
        shape = len(matrix)

    if name:
        print(name, "=", str(shape), str(type(matrix)))
    else:
        print(str(shape), str(type(matrix)))

    lines = format_matrix(matrix, spacing=spacing, decimals=decimals)
    print("\n".join(lines))
    print("\n")


def print_dict(dictionary):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    print(json.dumps(dictionary, indent=1, cls=NumpyEncoder))


def pprint(*args, shift=40, color=None):
    if len(args) < 2:
        raise ValueError("At least two arguments for printing.")

    format_str = (
        "{:<" + str(shift) + "}" + "\t".join(["{}" for _ in range(len(args) - 1)])
    )

    if color is not None:
        assert isinstance(color, int)
        format_str = ConsoleColors.COLORS[color] + format_str + ConsoleColors.END_COLOR

    print(format_str.format(*args))


def color_mask(values, mask, spacing=6, color=0):
    color_str = ConsoleColors.COLORS[color]

    mask_str = [
        color_str
        + ("{:>" + str(spacing) + "}").format(values[i])
        + ConsoleColors.END_COLOR
        if mask[i]
        else ("{:>" + str(spacing) + "}").format(values[i])
        for i in range(len(mask))
    ]
    return "".join(mask_str)
