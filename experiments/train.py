from models.EncodersDecoders import GCN_Encoder, GCN_Decoder, GAT_Encoder, GAT_Decoder
from models.VanillaGAE import VanillaGAE
from models.utils import sce_loss, mse_loss
from models.linear_classifier import LogisticRegression

import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import argparse
import importlib

from termcolor import colored, cprint

def get_args():
    parser = argparse.ArgumentParser(description="Train Graph JEPA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cora.py",
        help="Config file to use",
    )

    parser.add_argument(
        "--target_percentage",
        type=float,
        default=None,
        help="Percentage of target nodes to use",
    )

    return parser.parse_args()

def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

def get_dataset(config):
    dataset = None
    if config.dataset == 'Cora':
        dataset = Planetoid(root='datasets', name='Cora')
        dataset = dataset[0]
    elif config.dataset == 'CiteSeer':
        dataset = Planetoid(root='datasets', name='CiteSeer')
        dataset = dataset[0]
    elif config.dataset == 'PubMed':
        dataset = Planetoid(root="datasets", name="PubMed")
        dataset = dataset[0]
    else:
        cprint("invalid dataset...", "red")
    
    return dataset

def train(config, data):
    loss = mse_loss if config.loss_fn == 'mse' else sce_loss
    model = None
    if config.model_type == 'GCN':
        model = VanillaGAE(GCN_Encoder, GCN_Decoder, loss, config.num_node_features, config.hidden_channels)
    elif config.model_type == 'GAT':
        model = VanillaGAE(GAT_Encoder, GAT_Decoder, loss, config.num_node_features, config.hidden_channels)
    
    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # criterion = model.mse_loss
    criterion = model.loss

    model.train()
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(data.x, out)
        
        if epoch % 20 == 0:
            print(loss)
        
        loss.backward()
        optimizer.step()
    
    cprint("training complete...", "cyan")
    
    return model

def benchmark_logreg(config, data):
    accuracy = 0
    for i in range(config.runs):
        X_train, X_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.2, random_state=42)

        benchmark_classifier = LogisticRegression(config.num_node_features, config.num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(benchmark_classifier.parameters(), lr=0.01)

        # Training the benchmark_classifier
        num_epochs = config.eval_epochs

        for epoch in range(num_epochs):
            benchmark_classifier.train()

            # Forward pass
            outputs = benchmark_classifier(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 epochs
            # if (epoch + 1) % 50 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], enc_benchmark_classifier Loss: {loss.item():.4f}')

        benchmark_classifier.eval()
        with torch.no_grad():
            predicted = benchmark_classifier(X_test)
            _, pred = torch.max(predicted, 1)
            accuracy += accuracy_score(y_test.numpy(), pred.numpy())
            # print(f'Test Accuracy: {accuracy:.4f}')
    
    avg_acc = colored(accuracy / config.runs, "green", attrs=["bold"])
    print("benchmark linear classifier accuracy: ", avg_acc)

def linear_classifier(config, model, data):
    accuracy = 0
    for i in range(config.runs):

        with torch.no_grad():
            Z = model.encode(data.x, data.edge_index)

        # X_train, X_test, y_train, y_test = train_test_split(Z, data.y[data.train_mask], test_size=0.2, random_state=42, stratify=data.y)
        X_train = Z[data.train_mask]
        y_train = data.y[data.train_mask]
        X_test = Z[data.test_mask]
        y_test = data.y[data.test_mask]

        classifier = LogisticRegression(Z.shape[1], config.num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

        # Training the classifier
        num_epochs = config.eval_epochs

        for epoch in range(num_epochs):
            classifier.train()

            # Forward pass
            outputs = classifier(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 epochs
            # if (epoch + 1) % 50 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], enc_classifier Loss: {loss.item():.4f}')

        classifier.eval()
        with torch.no_grad():
            predicted = classifier(X_test)
            _, pred = torch.max(predicted, 1)
            accuracy += accuracy_score(y_test.numpy(), pred.numpy())
            # print(f'Test Accuracy: {accuracy:.4f}')

    avg_acc = colored(accuracy / config.runs, "green", attrs=["bold"])
    print("linear classifier accuracy (with GAE pretraining): ", avg_acc)
    
def driver(dataset_name):
    config = get_config(f"experiments/configs/{dataset_name}.py")
    
    data = get_dataset(config)
    model = train(config, data)
    
    linear_classifier(config, model, data)
    benchmark_logreg(config, data)