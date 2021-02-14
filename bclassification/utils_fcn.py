import numpy as np

from lib.action_space import is_do_nothing_action, is_sub_set_action, is_line_set_action
from lib.data_utils import (
    moving_window,
    extract_history_windows,
    indices_to_hot,
    backshift_and_hstack,
    is_nonetype,
)
from lib.dc_opf import TopologyConverter
from lib.visualizer import pprint


def obs_to_vects(obs, tc, mode):
    lines_or_to_sub_bus = tc.lines_or_to_sub_bus(obs)
    lines_ex_to_sub_bus = tc.lines_ex_to_sub_bus(obs)
    gens_to_sub_bus = tc.gens_to_sub_bus(obs)
    loads_to_sub_bus = tc.loads_to_sub_bus(obs)

    if mode == "lines":
        line_vect = list()
        line_vect.append(obs.p_or)
        line_vect.append(obs.rho)

        for bus in [1, 2]:
            mask_or = np.equal(lines_or_to_sub_bus, bus).astype(np.float)
            mask_ex = np.equal(lines_ex_to_sub_bus, bus).astype(np.float)
            mask_or = np.multiply(mask_or, obs.line_status.astype(np.float))
            mask_ex = np.multiply(mask_ex, obs.line_status.astype(np.float))
            line_vect.append(mask_or)
            line_vect.append(mask_ex)

        line_vect = np.nan_to_num(np.hstack(line_vect))

        temp_info = list()
        temp_info.append(obs.time_before_cooldown_sub.astype(np.float))
        temp_info.append(obs.time_before_cooldown_line.astype(np.float))
        temp_info.append(obs.timestep_overflow.astype(np.float))
        # temp_info.append(obs.time_next_maintenance.astype(np.float))
        # temp_info.append(obs.duration_next_maintenance.astype(np.float))
        temp_info = np.nan_to_num(np.hstack(temp_info))

        vects = (line_vect.astype(np.float), temp_info.astype(np.float))
    elif mode == "binary":
        inj_vect = list()
        inj_vect.append(obs.prod_p)
        inj_vect.append(obs.gen_pmax)
        inj_vect.append(obs.gen_pmin)
        inj_vect.append(obs.load_p)
        inj_vect = np.nan_to_num(np.hstack(inj_vect))

        line_vect = list()
        line_vect.append(obs.p_or)
        line_vect.append(obs.rho)
        line_vect = np.nan_to_num(np.hstack(line_vect))

        topo_vect = list()
        for bus in [1, 2]:
            mask_or = np.equal(lines_or_to_sub_bus, bus).astype(np.float)
            mask_ex = np.equal(lines_ex_to_sub_bus, bus).astype(np.float)
            mask_or = np.multiply(mask_or, obs.line_status.astype(np.float))
            mask_ex = np.multiply(mask_ex, obs.line_status.astype(np.float))
            topo_vect.append(mask_or)
            topo_vect.append(mask_ex)

            mask_gen = np.equal(gens_to_sub_bus, bus).astype(np.float)
            mask_load = np.equal(loads_to_sub_bus, bus).astype(np.float)
            topo_vect.append(mask_gen)
            topo_vect.append(mask_load)
        topo_vect = np.nan_to_num(np.hstack(topo_vect))

        temp_info = list()
        temp_info.append(obs.time_before_cooldown_sub.astype(np.float))
        temp_info.append(obs.time_before_cooldown_line.astype(np.float))
        temp_info.append(obs.timestep_overflow.astype(np.float))
        # temp_info.append(obs.time_next_maintenance.astype(np.float))
        # temp_info.append(obs.duration_next_maintenance.astype(np.float))
        temp_info = np.nan_to_num(np.hstack(temp_info))

        vects = (
            inj_vect.astype(np.float),
            line_vect.astype(np.float),
            topo_vect.astype(np.float),
            temp_info.astype(np.float),
        )
    elif mode == "structured":
        inj_vect = list()
        for bus in [1, 2]:
            prod = np.multiply(
                tc.sub_bus_mask(gens_to_sub_bus, bus, np.float), obs.prod_p
            )
            load = np.multiply(
                tc.sub_bus_mask(loads_to_sub_bus, bus, np.float), obs.load_p
            )
            inj_vect.append(prod)
            inj_vect.append(load)

        inj_vect.append(obs.gen_pmax)
        inj_vect.append(obs.gen_pmin)
        inj_vect = np.nan_to_num(np.hstack(inj_vect))

        line_vect = []
        for bus_or in [1, 2]:
            for bus_ex in [1, 2]:
                mask = np.multiply(
                    tc.sub_bus_mask(lines_or_to_sub_bus, bus_or, np.float),
                    tc.sub_bus_mask(lines_ex_to_sub_bus, bus_ex, np.float),
                )
                flow = np.multiply(obs.p_or, mask).astype(np.float)
                line_vect.append(flow)

        line_vect.append(obs.rho)
        line_vect = np.nan_to_num(np.hstack(line_vect))

        temp_info = list()
        temp_info.append(obs.time_before_cooldown_sub.astype(np.float))
        temp_info.append(obs.time_before_cooldown_line.astype(np.float))
        temp_info.append(obs.timestep_overflow.astype(np.float))
        # temp_info.append(obs.time_next_maintenance.astype(np.float))
        # temp_info.append(obs.duration_next_maintenance.astype(np.float))
        temp_info = np.nan_to_num(np.hstack(temp_info))

        vects = (
            inj_vect.astype(np.float),
            line_vect.astype(np.float),
            temp_info.astype(np.float),
        )
    elif mode == "unstructured":
        vects = (np.nan_to_num(obs.to_vect()).astype(np.float),)
    else:
        raise ValueError("Invalid mode.")

    return vects


def obs_to_vect(obs, tc, mode):
    return np.hstack(obs_to_vects(obs, tc, mode)).astype(np.float)


def action_to_vect(action):
    return indices_to_hot([int(action)], length=2, dtype=np.float)


def action_vects_to_vect(action_vects):
    return np.concatenate(action_vects)


def create_dataset(
    case,
    collector,
    input_mode,
    label_mode="dn",
    n_window_targets=0,
    n_window_history=0,
    downsampling_rate=None,
    n_window_forecasts=1,
    use_actions=True,
    feature_scaling=True,
):
    tc = TopologyConverter(case.env)
    process_fn = lambda obs: obs_to_vect(obs, tc, input_mode)
    combine_fn = lambda obs_vects: np.concatenate(obs_vects)

    if label_mode == "dn":
        label_fn = lambda actions, env, dtype: is_do_nothing_action(actions, env, dtype)
    elif "sub" in label_mode.split("-"):
        sub_id = int(label_mode.split("-")[-1])
        label_fn = lambda actions, env, dtype: is_sub_set_action(
            actions, sub_id=sub_id, env=env, dtype=dtype
        )
    elif "line" in label_mode.split("-"):
        line_id = int(label_mode.split("-")[-1])
        label_fn = lambda actions, env, dtype: is_line_set_action(
            actions, line_id=line_id, env=env, dtype=dtype
        )
    else:
        raise ValueError("Invalid label mode.")

    X_obses = []
    X_actions = []
    X_forecasts = []
    Y_all = []
    mask_targets = []

    obs_sample = None

    pprint("    - Input structure:", input_mode)
    pprint("    - Label structure:", label_mode)

    for chronic_idx, chronic_data in collector.data.items():
        chronic_len = len(chronic_data["actions"])
        chronic_obses = chronic_data["obses"][:chronic_len]

        chronic_labels = label_fn(
            chronic_data["actions"][:chronic_len], case.env, np.bool
        )

        # Mask
        mask_positives = extract_history_windows(
            chronic_labels, n_window=n_window_targets
        )

        if is_nonetype(downsampling_rate):
            downsampling_rate = chronic_labels.mean()

        mask_negatives = np.logical_and(
            np.random.binomial(1, downsampling_rate, len(chronic_labels)).astype(
                np.bool
            ),
            ~mask_positives,
        )
        chronic_mask_targets = np.logical_or(chronic_labels, mask_negatives)

        # Observation
        obs_sample = chronic_obses[0]
        chronic_X_obses = moving_window(
            chronic_obses,
            n_window=n_window_history,
            process_fn=process_fn,
            combine_fn=combine_fn,
            padding=np.zeros_like(process_fn(obs_sample)),
        )
        chronic_X_obses = np.vstack(chronic_X_obses)

        chronic_X_actions = None
        if use_actions:
            action_sample = 0.0
            chronic_actions = np.roll(chronic_labels, 1).astype(np.float)
            chronic_actions[0] = action_sample
            chronic_X_actions = moving_window(
                chronic_actions,
                n_window=n_window_history,
                process_fn=action_to_vect,
                combine_fn=action_vects_to_vect,
                padding=np.zeros_like(action_to_vect(action_sample)),
            )

        chronic_X_forecasts = None
        if n_window_forecasts > 0:
            prods, loads = collector.load_forecasts(case.env, chronic_idx)
            chronic_X_forecasts = np.concatenate(
                (prods[1 : chronic_len + 1], loads[1 : chronic_len + 1]), axis=1
            )

            chronic_X_forecasts = backshift_and_hstack(
                chronic_X_forecasts, max_shift=n_window_forecasts - 1
            )

        X_obses.append(chronic_X_obses)
        Y_all.extend(chronic_labels.astype(np.float))
        mask_targets.extend(chronic_mask_targets)

        if use_actions:
            X_actions.append(chronic_X_actions)

        if n_window_forecasts > 0:
            X_forecasts.append(chronic_X_forecasts)

    X_obses = np.vstack(X_obses)
    if use_actions:
        X_actions = np.vstack(X_actions)
    if n_window_forecasts > 0:
        X_forecasts = np.vstack(X_forecasts)
    Y_all = np.array(Y_all)
    mask_targets = np.array(mask_targets)

    if feature_scaling:
        s = len(process_fn(obs_sample))
        if input_mode == "lines":
            n_features = case.env.n_line
            std = X_obses[:, :n_features].std()

            for h in range(n_window_history + 1):
                X_obses[:, h * s : (h * s + n_features)] = (
                    X_obses[:, h * s : (h * s + n_features)] / std
                )

            if n_window_forecasts > 0:
                X_forecasts = X_forecasts / std

        elif input_mode == "binary":
            n_features = 3 * case.env.n_gen + case.env.n_load + case.env.n_line
            std = X_obses[:, :n_features].std()

            for h in range(n_window_history + 1):
                X_obses[:, h * s : (h * s + n_features)] = (
                    X_obses[:, h * s : (h * s + n_features)] / std
                )

            if n_window_forecasts > 0:
                X_forecasts = X_forecasts / std

        elif input_mode == "structured":
            n_features = 4 * case.env.n_gen + 2 * case.env.n_load
            std = X_obses[:, :n_features].std()

            n_features = n_features + 4 * case.env.n_line
            for h in range(n_window_history + 1):
                X_obses[:, h * s : (h * s + n_features)] = (
                    X_obses[:, h * s : (h * s + n_features)] / std
                )

            if n_window_forecasts > 0:
                X_forecasts = X_forecasts / std
        elif input_mode == "unstructured":
            std = X_obses.std(axis=0)
            mask_zero = np.equal(std, 0.0)
            X_obses = X_obses[:, ~mask_zero]
            X_obses = X_obses / std[~mask_zero]

            if n_window_forecasts > 0:
                std = X_forecasts.std(axis=0)
                mask_zero = np.equal(std, 0.0)
                X_forecasts = X_forecasts[:, ~mask_zero]
                X_forecasts = X_forecasts / std[~mask_zero]

    X_all = X_obses
    if use_actions:
        X_all = np.hstack((X_all, X_actions))
    if n_window_forecasts > 0:
        X_all = np.hstack((X_all, X_forecasts))

    X = X_all[mask_targets, :]
    Y = Y_all[mask_targets]

    pprint(
        "    - Labels:",
        f"{int(Y_all.sum())}/{Y_all.size}",
        "{:.2f} %".format(100 * Y_all.mean()),
    )

    return X, Y, mask_targets, X_all, Y_all
