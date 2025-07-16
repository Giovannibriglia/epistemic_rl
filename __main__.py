from typing import Dict, List

import torch

from src.preprocessing import GraphDataPipeline
from src.utils import get_dataloaders, seed_everything, select_model


def print_values(samples):
    d = {}
    for s in samples:
        ss = s["target"].item()
        if ss in d.keys():
            d[ss] += 1
        else:
            d[ss] = 1

    x = ""
    z = 0
    for target in sorted(d):
        x += f"| Target {target}: {d[target]}"
        z += d[target]
    print(x, " -- Total samples: ", z)


def f(value, slope, min_value_nn, if_forward: bool = True):
    if if_forward:
        return value * slope + min_value_nn
    else:
        return value / slope - min_value_nn


def prepare_samples(t_s_copy: List[Dict], t_t_copy: List[Dict]):

    t_s_copy = [s for s in t_s_copy if s["target"].item() != UNREACHABLE_STATE_VALUE]
    t_t_copy = [s for s in t_t_copy if s["target"].item() != UNREACHABLE_STATE_VALUE]

    """def find_max(sss):
        max_v = -1
        for s in sss:
            v = s["target"].item()
            if v > max_v and v != UNREACHABLE_STATE_VALUE:
                max_v = v
        return max_v

    max_train = find_max(t_s_copy)
    max_test = find_max(t_t_copy)

    max_tot = max(max_train, max_test)"""

    MIN_DEPTH = 0
    MAX_DEPTH = 50  # if max_tot * 2 > 50 else max_tot * 2

    MIN_V_NN = 1e-3
    MAX_V_NN = 1 - MIN_V_NN

    slope = (MAX_V_NN - MIN_V_NN) / (MAX_DEPTH - MIN_DEPTH)

    params = {"slope": slope, "min_value_nn": MIN_V_NN}

    print("params: ", params)
    for s in t_s_copy:
        v = s["target"].item()
        if v != UNREACHABLE_STATE_VALUE:
            s["target"] = torch.tensor(f(v, slope, MIN_V_NN), dtype=torch.float)
        else:
            s["target"] = torch.tensor(f(MAX_DEPTH, slope, MIN_V_NN), dtype=torch.float)

    for s in t_t_copy:
        v = s["target"].item()
        if v != UNREACHABLE_STATE_VALUE:
            s["target"] = torch.tensor(f(v, slope, MIN_V_NN), dtype=torch.float)
        else:
            s["target"] = torch.tensor(f(MAX_DEPTH, slope, MIN_V_NN), dtype=torch.float)

    return t_s_copy, t_t_copy, params


if __name__ == "__main__":
    seed_everything()

    DOMAIN = "ALL_CC"
    IF_BUILD_DATA = True
    IF_TRAIN = True

    FOLDER_DATA = f"out/NN/Training_{DOMAIN}"
    UNREACHABLE_STATE_VALUE = 1000000
    TEST_SIZE = 0.2

    MODELS = [
        "distance_estimator",
    ]

    PATH_SAVE_DATA = "data"
    PATH_SAVE_MODEL = "trained_models"

    kinds_of_ordering = ["hash"]  # , "map"
    kinds_of_data = ["merged"]  # "separated",
    use_goals = [False]  # , True
    use_depths = [False]  # , True

    N_TRAIN_EPOCHS = 500
    BATCH_SIZE = 2048

    for KIND_OF_ORDERING in kinds_of_ordering:
        for KIND_OF_DATA in kinds_of_data:
            for USE_GOAL in use_goals:
                for USE_DEPTH in use_depths:

                    path_save_data = (
                        f"{PATH_SAVE_DATA}/{DOMAIN}/{KIND_OF_ORDERING}_{KIND_OF_DATA}"
                    )
                    path_save_data += "_goal" if USE_GOAL else "_no_goal"
                    path_save_data += "_depth" if USE_DEPTH else "_no_depth"

                    if IF_BUILD_DATA:
                        pipe = GraphDataPipeline(
                            folder_data=FOLDER_DATA,
                            kind_of_ordering=KIND_OF_ORDERING,
                            kind_of_data=KIND_OF_DATA,
                            unreachable_state_value=UNREACHABLE_STATE_VALUE,
                            test_size=TEST_SIZE,
                            use_goal=USE_GOAL,
                            use_depth=USE_DEPTH,
                        )

                        saved_path = pipe.save(
                            out_dir=path_save_data,
                            extra_params={},
                        )

                    data_path = path_save_data + "/samples.pt"
                    data = torch.load(data_path, weights_only=False)
                    train_samples = data["train_samples"]
                    test_samples = data["test_samples"]

                    for model_name in MODELS:
                        print(
                            f"\n Domain: {DOMAIN} | {KIND_OF_ORDERING} | {KIND_OF_DATA} | Use goal: {USE_GOAL} | Use depth: {USE_DEPTH} | Model: {model_name}",
                        )
                        train_samples_copy = train_samples.copy()
                        test_samples_copy = test_samples.copy()

                        train_samples_copy, test_samples_copy, params_f = (
                            prepare_samples(
                                train_samples_copy,
                                test_samples_copy,
                            )
                        )

                        print("Train values:")
                        print_values(train_samples_copy)
                        print("Test values:")
                        print_values(test_samples_copy)

                        train_loader, val_loader = get_dataloaders(
                            train_samples_copy, test_samples_copy, batch_size=BATCH_SIZE
                        )

                        path_model = (
                            f"{PATH_SAVE_MODEL}/{DOMAIN}/"
                            + path_save_data.split("/")[-1]
                            + "/"
                            + model_name
                        )

                        # instantiate
                        m = select_model(model_name, USE_GOAL, USE_DEPTH)

                        # train
                        if IF_TRAIN:
                            m.train(
                                train_loader,
                                val_loader,
                                n_epochs=N_TRAIN_EPOCHS,
                                checkpoint_dir=path_model,
                            )

                        # load
                        m.load_model(f"{path_model}/best.pt")

                        with open(f"{path_model}/C.txt", "w", encoding="utf-8") as fh:
                            for key, value in params_f.items():
                                fh.write(f"{key} = {value}\n")

                        kwargs = {"C": params_f["slope"] / 2}

                        m.evaluate(val_loader, verbose=True, **kwargs)
                        # single inference
                        if USE_GOAL:
                            if USE_DEPTH:
                                out = m.predict_single(
                                    "to_predict.dot", 5, "goal_tree.dot"
                                )
                            else:
                                out = m.predict_single(
                                    "to_predict.dot",
                                    depth=None,
                                    goal_dot="goal_tree.dot",
                                )
                        else:
                            if USE_DEPTH:
                                out = m.predict_single("to_predict.dot", 5)
                            else:
                                out = m.predict_single("to_predict.dot")
                        print("PyTorch output: ", out)

                        onnx_model_file = f"{path_model}/{model_name}.onnx"
                        m.to_onnx(onnx_model_file, USE_GOAL, USE_DEPTH)

                        sss = ["to_predict.dot", "to_predict.dot"]
                        ggg = ["goal_tree.dot", "goal_tree.dot"]
                        ddd = [5, 5]
                        if USE_GOAL:
                            if USE_DEPTH:
                                out_onnx = m.try_onnx(onnx_model_file, sss, ddd, ggg)
                            else:
                                out_onnx = m.try_onnx(
                                    onnx_model_file, sss, None, goal_dot_files=ggg
                                )
                        else:
                            if USE_DEPTH:
                                out_onnx = m.try_onnx(onnx_model_file, sss, ddd)
                            else:
                                out_onnx = m.try_onnx(onnx_model_file, sss)
                        print("Out onnx: ", out_onnx)
