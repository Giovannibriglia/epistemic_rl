import torch

from src.preprocessing import GraphDataPipeline
from src.utils import get_dataloaders, seed_everything, select_model, split_samples

if __name__ == "__main__":
    seed_everything()

    DOMAIN = "ALL_CC"
    IF_BUILD_DATA = True
    IF_TRAIN = True

    FOLDER_DATA = f"out/NN/Training_{DOMAIN}"
    UNREACHABLE_STATE_VALUE = 1000000

    MODELS = [
        "distance_estimator",
        # "reachability_classifier",
    ]

    PATH_SAVE_DATA = "data_no_depth"
    PATH_SAVE_MODEL = "trained_models_no_depth2"

    kinds_of_ordering = ["hash"]  # , "map"
    kinds_of_data = ["separated"]  # "separated",
    use_goals = [False]  # , True
    use_depths = [False]  # , True

    MAX_SAMPLES_FOR_CLASS = 15000
    MAX_UNREACHABLE_SAMPLES_RATIO = 0.15
    N_TRAIN_EPOCHS = 500

    for KIND_OF_ORDERING in kinds_of_ordering:
        for KIND_OF_DATA in kinds_of_data:
            for USE_GOAL in use_goals:
                for USE_DEPTH in use_depths:

                    path_save_data = (
                        f"{PATH_SAVE_DATA}/{DOMAIN}/{KIND_OF_ORDERING}_{KIND_OF_DATA}"
                    )
                    path_save_data += "_goal" if USE_GOAL else "_no_goal"

                    if IF_BUILD_DATA:
                        pipe = GraphDataPipeline(
                            folder_data=FOLDER_DATA,
                            kind_of_ordering=KIND_OF_ORDERING,
                            kind_of_data=KIND_OF_DATA,
                            max_samples_for_prob=MAX_SAMPLES_FOR_CLASS,
                            max_unreachable_ratio=MAX_UNREACHABLE_SAMPLES_RATIO,
                            UNREACHABLE_STATE_VALUE=UNREACHABLE_STATE_VALUE,
                            use_goal=USE_GOAL,
                            use_depth=USE_DEPTH,
                        )

                        saved_path = pipe.save(
                            out_dir=path_save_data,
                            extra_params={
                                "max_unreachable_ratio": MAX_UNREACHABLE_SAMPLES_RATIO,
                            },
                        )

                    data_path = path_save_data + "/dataloader_info.pt"
                    data = torch.load(data_path, weights_only=False)
                    samples = data["samples"]

                    for model_name in MODELS:
                        print(
                            f"\n Domain: {DOMAIN} | {KIND_OF_ORDERING} | {KIND_OF_DATA} | Use goal: {USE_GOAL} | Use depth: {USE_DEPTH} | Model: {model_name}",
                        )
                        samples_copy = samples.copy()

                        if "reachability_classifier" in model_name:
                            for s in samples_copy:
                                # extract the old scalar
                                old = s["target"].item()
                                # decide new label
                                new_label = (
                                    1.0 if old != UNREACHABLE_STATE_VALUE else 0.0
                                )
                                # store back as a float tensor of shape [1]
                                s["target"] = torch.tensor(
                                    [new_label], dtype=torch.float32
                                )

                        elif "distance_estimator" in model_name:
                            # Filter out unreachable states
                            samples_copy = [
                                s
                                for s in samples_copy
                                if s["target"] <= UNREACHABLE_STATE_VALUE
                            ]

                            # Build a dict of real values
                            real_values = {}
                            for s in samples_copy:
                                ss = int(s["target"].item())
                                if (
                                    ss != UNREACHABLE_STATE_VALUE
                                    and ss not in real_values
                                ):
                                    real_values[ss] = -1.0  # placeholder

                            # Compute normalized values
                            MAX_PERCENTAGE = 0.5
                            max_real_value = max(real_values.keys())

                            for k in list(real_values.keys()):
                                real_values[k] = max(
                                    (k * MAX_PERCENTAGE) / max_real_value,
                                    MAX_PERCENTAGE / (2 * max_real_value),
                                )

                            # Add mapping for the unreachable state
                            real_values[UNREACHABLE_STATE_VALUE] = 0.99

                            # Now rewrite targets in-place
                            for s in samples_copy:
                                v = int(
                                    s["target"].item()
                                )  # this is an int or the unreachable-state constant
                                normalized = real_values[v]  # lookup by integer
                                s["target"] = torch.tensor(
                                    normalized, dtype=torch.float32
                                )

                            samples_copy = [
                                s
                                for s in samples_copy
                                if s["target"] < 0.99  # UNREACHABLE_STATE_VALUE
                            ]

                        d = {}
                        for s in samples_copy:
                            ss = s["target"].item()
                            if ss in d.keys():
                                d[ss] += 1
                            else:
                                d[ss] = 1
                        print("Target values: ", d)

                        train_samples, eval_samples = split_samples(samples_copy)

                        train_loader, val_loader = get_dataloaders(
                            train_samples, eval_samples, batch_size=2048
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
                        m.evaluate(val_loader, verbose=True)
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
