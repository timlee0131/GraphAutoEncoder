import ml_collections

# Cora (Planetoid datasets) paper: https://arxiv.org/pdf/1603.08861
# features are binary bag-of-words vectors


def get_config():
    config = ml_collections.ConfigDict()

    # General info
    config.computer = "superpod"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "datasets"
    )
    config.artifacts_dir = "/models"

    config.task = "node_transductive"
    config.dataset = "Cora"
    config.num_node_features = 1433
    config.num_classes = 7

    # Encoder models info
    config.model_type = "GCN"
    config.num_layers = 2
    config.hidden_channels = 128

    # training info
    config.epochs = 200
    config.runs = 10
    config.lr = 0.01
    config.loss_fn = "sce"

    # evaluation info
    config.eval_epochs = 100
    config.eval_lr = 0.01
    config.eval_metric = "accuracy_score"
    config.eval_model = "logistic"

    return config