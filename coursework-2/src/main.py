import time

from experiment import (
    get_default_args,
    compare_experiment,
    run_experiment,
    evaluate,
    get_parameters_count,
    compress_model,
)


def compare_experiment_number_of_layers():
    args1 = get_default_args()

    args2 = get_default_args()
    args2.conv_channels = [3, 6, 6]

    args3 = get_default_args()
    args3.conv_channels = [3, 3, 6, 6]

    compare_experiment(
        (args1, "2 Conv Layers"), (args2, "3 Conv Layers"), (args3, "4 Conv Layers")
    )


def compare_experiment_kernel_size():
    args1 = get_default_args()

    args2 = get_default_args()
    args2.kernel_size = 5

    args3 = get_default_args()
    args3.kernel_size = 10

    args4 = get_default_args()
    args4.kernel_size = 30

    compare_experiment(
        (args1, "Kernel size 3"),
        (args2, "Kernel size 5"),
        (args3, "Kernel size 10"),
        (args4, "Kernel size 30"),
    )


def compare_experiment_skip_connections():
    args1 = get_default_args()
    args2 = get_default_args()
    args2.skip = True

    compare_experiment(
        (args1, "Without skip connection"), (args2, "With skip connection")
    )


def compare_experiment_dropout_regularization():
    args1 = get_default_args()
    args1.weight_decay = 0

    args2 = get_default_args()
    args2.dropout = True
    args2.weight_decay = 0

    args3 = get_default_args()

    args4 = get_default_args()
    args4.dropout = True

    compare_experiment(
        (args1, "Without dropout, without regularization term"),
        (args2, "With dropout, without regularization term"),
        (args3, "Without dropout, with regularization term"),
        (args4, "With dropout, with regularization term"),
    )


def compare_experiment_loss_function():
    args1 = get_default_args()
    args2 = get_default_args()
    args2.loss = "SquaredHinge"

    compare_experiment((args1, "Cross-entropy"), (args2, "Squared hinge"))


def compare_experiment_learning_rates():
    args1 = get_default_args()
    args1.learning_rate = 1e-2

    args2 = get_default_args()

    args3 = get_default_args()
    args3.learning_rate = 1e-4

    compare_experiment(
        (args1, "Learning rate 1e-2"),
        (args2, "Learning rate 1e-3"),
        (args3, "Learning rate 1e-4"),
    )


def compare_experiment_batch_sizes():
    args1 = get_default_args()
    args1.batch_size = 5

    args2 = get_default_args()

    args3 = get_default_args()
    args3.batch_size = 30

    args4 = get_default_args()
    args4.batch_size = 50

    compare_experiment(
        (args1, "Batch size 5"),
        (args2, "Batch size 10"),
        (args3, "Batch size 30"),
        (args4, "Batch size 50"),
    )


def compare_experiment_pretrained_weights():
    args1 = get_default_args()
    args1.model = "ResNet50"

    args2 = get_default_args()
    args2.model = "ResNet50"
    args2.use_pretrained = True

    compare_experiment(
        (args1, "Without pretrained weights"), (args2, "With pretrained weights")
    )


def compare_experiment_normalization():
    args1 = get_default_args()

    args2 = get_default_args()
    args2.norm = "BatchNorm"

    args3 = get_default_args()
    args3.norm = "InstanceNorm"

    args4 = get_default_args()
    args4.norm = "LayerNorm"

    compare_experiment(
        (args1, "Without normalization"),
        (args2, "With batch normalization"),
        (args3, "With instance normalization"),
        (args4, "With layer normalization"),
    )


def model_experiment_compression():
    args = get_default_args()
    (_, _, model), _, test_loader = run_experiment(args)

    start = time.time()
    accuracy = evaluate(model, test_loader)
    end = time.time()
    print(f"Without compression, elapsed testing time: {(end - start) * 1000:.2f}ms")
    print(
        f"Accuracy: {accuracy:.2f}%, Total number of parameter: {get_parameters_count(model)}"
    )

    compress_model(model)

    start = time.time()
    accuracy = evaluate(model, test_loader)
    end = time.time()
    print(f"With compression, elapsed testing time: {(end - start) * 1000:.2f}ms")
    print(
        f"Accuracy: {accuracy:.2f}%, Total number of parameter: {get_parameters_count(model)}"
    )


if __name__ == "__main__":
    # compare_experiment_number_of_layers()
    # compare_experiment_kernel_size()

    # Residual Block 활성화 (이 실험만)
    # compare_experiment_skip_connections()

    # compare_experiment_dropout_regularization()
    # compare_experiment_loss_function()
    # compare_experiment_learning_rates()
    # compare_experiment_batch_sizes()
    # compare_experiment_pretrained_weights()
    # compare_experiment_normalization()
    model_experiment_compression()
