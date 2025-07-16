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
    print(x)
    print(z)


def prepare_samples(t_s_copy: List[Dict], t_t_copy: List[Dict], max_p: float):
    t_s_copy = [s for s in t_s_copy if s["target"].item() != UNREACHABLE_STATE_VALUE]
    t_t_copy = [s for s in t_t_copy if s["target"].item() != UNREACHABLE_STATE_VALUE]

    max_train = -1
    for s in t_s_copy:
        v = s["target"].item()
        if v > max_train and v != UNREACHABLE_STATE_VALUE:
            max_train = v

    max_test = -1
    for s in t_t_copy:
        v = s["target"].item()
        if v > max_test and v != UNREACHABLE_STATE_VALUE:
            max_test = v

    max_all = 2 * max(max_train, max_test)

    C = max_p / max_all
    print("C = ", C)
    for s in t_s_copy:
        v = s["target"].item()
        if v != UNREACHABLE_STATE_VALUE:
            s["target"] *= C
        else:
            s["target"] = torch.tensor([0.99], dtype=torch.float32)

    for s in t_t_copy:
        v = s["target"].item()
        if v != UNREACHABLE_STATE_VALUE:
            s["target"] *= C
        else:
            s["target"] = torch.tensor([0.99], dtype=torch.float32)

    return t_s_copy, t_t_copy, C, max_all


if __name__ == "__main__":
    seed_everything()

    DOMAIN = "ALL_CC"
    IF_BUILD_DATA = True
    IF_TRAIN = True

    FOLDER_DATA = f"out/NN/Training_{DOMAIN}"
    UNREACHABLE_STATE_VALUE = 1000000

    MODELS = [
        "distance_estimator",
    ]

    PATH_SAVE_DATA = "data_ok_new"
    PATH_SAVE_MODEL = "trained_models_ok_new_no_unreachable"

    kinds_of_ordering = ["hash"]  # , "map"
    kinds_of_data = ["merged"]  # "separated",
    use_goals = [False]  # , True
    use_depths = [False]  # , True

    N_TRAIN_EPOCHS = 500
    BATCH_SIZE = 2048
    MAX_UNREACHABLE_SAMPLES_RATIO = 0.1
    TEST_SIZE = 0.2

    MAX_PERCENTAGE_REACHABLE_STATES = 0.7

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
                            max_unreachable_samples_ratio=MAX_UNREACHABLE_SAMPLES_RATIO,
                            test_size=TEST_SIZE,
                            use_goal=USE_GOAL,
                            use_depth=USE_DEPTH,
                        )

                        saved_path = pipe.save(
                            out_dir=path_save_data,
                            extra_params={
                                "max_unreachable_ratio": MAX_UNREACHABLE_SAMPLES_RATIO,
                            },
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

                        train_samples_copy, test_samples_copy, C, max_all = (
                            prepare_samples(
                                train_samples_copy,
                                test_samples_copy,
                                MAX_PERCENTAGE_REACHABLE_STATES,
                            )
                        )

                        print_values(train_samples_copy)
                        print("\n")
                        print_values(test_samples_copy)

                        train_loader, val_loader = get_dataloaders(
                            train_samples_copy, test_samples_copy, batch_size=BATCH_SIZE
                        )

                        path_model = (
                            f"{PATH_SAVE_MODEL}/{DOMAIN}/"
                            + path_save_data.split("/")[-1]
                            + "/"
                            + model_name
                            + "_07perc_d2maxtrain"
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

                        with open(f"{path_model}/C.txt", "w", encoding="utf-8") as f:
                            f.write(f"C = {C}\n")
                            f.write(
                                f"max percentage seen reachable distances = {MAX_PERCENTAGE_REACHABLE_STATES}\n"
                            )
                            f.write(f"max depth = {max_all}\n")
                        kwargs = {"C": C / 2}

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
