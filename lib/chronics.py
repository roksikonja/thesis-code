import datetime
import os


def get_sorted_chronics(env):
    chronics_dir = env.chronics_handler.path
    chronics = os.listdir(chronics_dir)

    # Filter meta files
    chronics = list(
        filter(lambda x: os.path.isdir(os.path.join(chronics_dir, x)), chronics)
    )

    if env.name in ["rte_case5_example", "l2rpn_2019"]:
        chronics_sorted = sorted(chronics, key=lambda x: int(x))
    else:
        chronics_sorted = sorted(
            chronics,
            key=lambda x: (
                datetime.datetime.strptime(x.split("_")[1].capitalize(), "%B").month,
                int(x.split("_")[-1]),
            ),
        )

    return chronics_dir, chronics, chronics_sorted
