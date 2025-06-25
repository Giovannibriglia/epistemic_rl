import torch

from src.preprocessing import GraphDataPipeline
from src.utils import get_dataloaders, seed_everything, select_model, split_samples

if __name__ == "__main__":
    seed_everything()

    DOMAIN = "CC_3_2_3"

    FOLDER_DATA = f"out/NN/Training_{DOMAIN}"
    UNREACHABLE_STATE_VALUE = -1  # 1_000_000 or -1

    MODELS = [
        "distance_estimator",
        # "reachability_classifier",
    ]

    kinds_of_ordering = ["hash"]  # , "map"
    kinds_of_data = ["separated", "merged"]
    use_goals = [True, False]

    MAX_SAMPLES_FOR_CLASS = 15000
    MAX_UNREACHABLE_SAMPLES_RATIO = 0.15
    N_TRAIN_EPOCHS = 200

    for KIND_OF_ORDERING in kinds_of_ordering:
        for KIND_OF_DATA in kinds_of_data:
            for USE_GOAL in use_goals:

                path_save_data = f"data/{DOMAIN}/{KIND_OF_ORDERING}_{KIND_OF_DATA}"
                path_save_data += "_goal" if USE_GOAL else "_no_goal"

                pipe = GraphDataPipeline(
                    folder_data=FOLDER_DATA,
                    kind_of_ordering=KIND_OF_ORDERING,
                    kind_of_data=KIND_OF_DATA,
                    max_samples_for_prob=MAX_SAMPLES_FOR_CLASS,
                    max_unreachable_ratio=MAX_UNREACHABLE_SAMPLES_RATIO,
                    UNREACHABLE_STATE_VALUE=UNREACHABLE_STATE_VALUE,
                    use_goal=USE_GOAL,
                )

                saved_path = pipe.save(
                    out_dir=path_save_data,
                    extra_params={
                        "max_unreachable_ratio": MAX_UNREACHABLE_SAMPLES_RATIO,
                    },
                )

                # Later, to reload:
                data_path = path_save_data + "/dataloader_info.pt"
                data = torch.load(data_path, weights_only=False)
                samples = data["samples"]

                for model_name in MODELS:
                    print(
                        f"\n Domain: {DOMAIN} | {KIND_OF_ORDERING} | {KIND_OF_DATA} | Use goal: {USE_GOAL} | Model: {model_name}",
                    )
                    samples_copy = samples.copy()

                    if "reachability_classifier" in model_name:
                        for s in samples_copy:
                            # extract the old scalar
                            old = s["target"].item()
                            # decide new label
                            new_label = 1.0 if old != UNREACHABLE_STATE_VALUE else 0.0
                            # store back as a float tensor of shape [1]
                            s["target"] = torch.tensor([new_label], dtype=torch.float32)

                    elif "distance_estimator" in model_name:
                        samples_copy = [
                            s
                            for s in samples_copy
                            if s["target"] != UNREACHABLE_STATE_VALUE
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
                        train_samples, eval_samples
                    )

                    path_model = (
                        f"trained_models/{DOMAIN}/"
                        + path_save_data.split("/")[-1]
                        + "/"
                        + model_name
                    )

                    # instantiate
                    m = select_model(model_name, USE_GOAL)

                    # train
                    m.train(
                        train_loader,
                        val_loader,
                        n_epochs=N_TRAIN_EPOCHS,
                        checkpoint_dir=path_model,
                    )

                    # load
                    m.load_model(f"{path_model}/best.pt")
                    m.evaluate(val_loader)
                    # single inference
                    if USE_GOAL:
                        out = m.predict_single("state.dot", 5, "goal_tree.dot")
                    else:
                        out = m.predict_single("state.dot", 5)
                    print("PyTorch output: ", out)

                    onnx_model_file = f"{path_model}/{model_name}.onnx"
                    m.to_onnx(onnx_model_file)

                    sss = ["state.dot", "state.dot"]
                    ggg = ["goal_tree.dot", "goal_tree.dot"]
                    ddd = [5, 5]
                    if USE_GOAL:
                        out_onnx = m.try_onnx(onnx_model_file, sss, ddd, ggg)
                    else:
                        out_onnx = m.try_onnx(onnx_model_file, sss, ddd)
                    print("Out onnx: ", out_onnx)
