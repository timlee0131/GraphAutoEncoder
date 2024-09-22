import ml_collections

# Cora (Planetoid datasets) paper: https://arxiv.org/pdf/1603.08861
# features are binary bag-of-words vectors


def get_config():
    config = ml_collections.ConfigDict()

    config.TEST = False

    # General info
    config.computer = "superpod"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "datasets"
    )
    config.artifacts_dir = "/models"

    # dataset info
    config.task = "node_transductive"
    config.dataset = "CiteSeer"
    config.num_node_features = 3703
    config.num_classes = 6

    # Encoder models info
    config.model_type = "GCN"
    config.num_layers = 2
    config.hidden_channels = 16

    # training info
    config.epochs = 200
    config.runs = 10
    config.lr = 0.001
    config.loss_fn = "sce"

    # evaluation info
    config.eval_epochs = 100
    config.eval_lr = 0.01
    config.eval_metric = "accuracy_score"
    config.eval_model = "logistic"

    return config