import argparse

import torch

from src.preprocessing import GraphDataPipeline
from src.utils import (
    get_dataloaders,
    prepare_samples,
    print_values,
    seed_everything,
    select_model,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "y", "true", "t"):
        return True
    if v in ("no", "n", "false", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and/or build data for the distance estimator model."
    )
    # simple string / numeric args
    parser.add_argument(
        "--domain", type=str, default="CC__pl_4_5_6", help="Dataset domain identifier"
    )
    parser.add_argument(
        "--model-name",
        default="distance_estimator",
        type=str,
        help="Name for the distance estimator model",
    )
    parser.add_argument(
        "--folder-data",
        type=str,
        default="out/NN/Training",
        help="Where to find/build the data",
    )

    parser.add_argument(
        "--unreachable-state-value",
        type=int,
        default=1000000,
        help="Value to use for unreachable states",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction of held-out test data"
    )
    parser.add_argument(
        "--dir-save-data",
        type=str,
        default="data",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--dir-save-model",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--n-train-epochs", type=int, default=500, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Training batch size"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random State")

    # boolean flags
    parser.add_argument(
        "--build-data",
        type=str2bool,
        default=True,
        help="Whether to (re)build the data",
    )
    parser.add_argument(
        "--train", type=str2bool, default=True, help="Whether to train the models"
    )

    parser.add_argument(
        "--kind-of-ordering",
        type=str,
        choices=["hash", "map"],
        default="hash",
        help="Ordering strategy to consider",
    )
    parser.add_argument(
        "--kind-of-data",
        type=str,
        choices=["merged", "separated"],
        default="merged",
        help="Data split type to use",
    )

    parser.add_argument(
        "--use-goal",
        type=str2bool,
        default=False,
        help="Whether to include goal info (true/false)",
    )
    parser.add_argument(
        "--use-depth",
        type=str2bool,
        default=False,
        help="Whether to include depth info (true/false)",
    )

    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to print detailed eval errors",
    )

    args = parser.parse_args()

    return args


def main(args):
    seed = args.seed
    seed_everything(seed)

    domain = args.domain
    if_build_data = args.build_data
    if_train = args.train
    kind_of_ordering = args.kind_of_ordering
    kind_of_data = args.kind_of_data
    use_goal = args.use_goal
    use_depth = args.use_depth

    folder_data = f"{args.folder_data}_{domain}"
    path_save_data = args.dir_save_data
    path_save_model = args.dir_save_model

    model_name = args.model_name

    unreachable_state_value = args.unreachable_state_value
    test_size = args.test_size

    n_train_epochs = args.n_train_epochs
    batch_size = args.batch_size

    verbose = args.verbose

    path_save_data = f"{path_save_data}/{domain}/{kind_of_ordering}_{kind_of_data}"
    path_save_data += "_goal" if use_goal else "_no_goal"
    path_save_data += "_depth" if use_depth else "_no_depth"
    data_path = path_save_data + "/samples.pt"

    print("\n************************************************")

    if if_build_data:
        pipe = GraphDataPipeline(
            folder_data=folder_data,
            kind_of_ordering=kind_of_ordering,
            kind_of_data=kind_of_data,
            unreachable_state_value=unreachable_state_value,
            max_percentage_per_class=0.15,
            test_size=test_size,
            use_goal=use_goal,
            use_depth=use_depth,
            random_state=seed,
        )

        pipe.save(
            out_dir=data_path,
            extra_params={},
        )

    data = torch.load(data_path, weights_only=False)
    train_samples = data["train_samples"]
    test_samples = data["test_samples"]

    print(
        f"Domain: {domain} | {kind_of_ordering} | {kind_of_data} | Use goal: {use_goal} | Use depth: {use_depth} | Model name: {model_name}",
    )
    train_samples_copy = train_samples.copy()
    test_samples_copy = test_samples.copy()

    train_samples_copy, test_samples_copy, params_f = prepare_samples(
        train_samples_copy, test_samples_copy, unreachable_state_value
    )

    if verbose:
        print("Train values:")
        print_values(train_samples_copy)
        print("Test values:")
        print_values(test_samples_copy)

    train_loader, val_loader = get_dataloaders(
        train_samples_copy,
        test_samples_copy,
        batch_size=batch_size,
    )

    path_model = (
        path_save_model
        + "/"
        + domain
        + "/"
        + path_save_data.split("/")[-1]
        + "/"
        + model_name
    )

    # instantiate
    m = select_model(model_name, use_goal, use_depth)

    # train
    if if_train:
        m.train(
            train_loader,
            val_loader,
            n_epochs=n_train_epochs,
            checkpoint_dir=path_model,
        )

    # load
    m.load_model(f"{path_model}/{model_name}.pt")

    with open(
        f"{path_model}/{model_name}_normalization_constants.txt",
        "w",
        encoding="utf-8",
    ) as fh:
        for key, value in params_f.items():
            fh.write(f"{key} = {value}\n")

    kwargs = {"th": params_f["slope"] / 2}

    m.evaluate(val_loader, verbose=verbose, **kwargs)

    example_state_to_predict = f"./examples/{kind_of_ordering}_{kind_of_data}_state.pt"
    example_goal = "./examples/goal_tree.dot"
    example_depth = 5

    # single inference
    if use_goal:
        if use_depth:
            out = m.predict_single(
                example_state_to_predict,
                example_depth,
                example_goal,
            )
        else:
            out = m.predict_single(
                example_state_to_predict,
                depth=None,
                goal_dot=example_goal,
            )
    else:
        if use_depth:
            out = m.predict_single(example_state_to_predict, example_depth)
        else:
            out = m.predict_single(example_state_to_predict)
    print("PyTorch output: ", out)

    onnx_model_file = f"{path_model}/{model_name}.onnx"
    m.to_onnx(onnx_model_file, use_goal, use_depth)

    sss = [example_state_to_predict, example_state_to_predict]
    ggg = [example_goal, example_goal]
    ddd = [example_depth, example_depth]
    if use_goal:
        if use_depth:
            out_onnx = m.try_onnx(onnx_model_file, sss, ddd, ggg)
        else:
            out_onnx = m.try_onnx(onnx_model_file, sss, None, goal_dot_files=ggg)
    else:
        if use_depth:
            out_onnx = m.try_onnx(onnx_model_file, sss, ddd)
        else:
            out_onnx = m.try_onnx(onnx_model_file, sss)
    print("Out onnx: ", out_onnx)


if __name__ == "__main__":
    args = parse_args()
    main(args)
