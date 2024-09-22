import argparse
from experiments import train

def get_args():
    parser = argparse.ArgumentParser(description="Graph Auto Encoder")

    parser.add_argument('-m', '--mode', choices=['train', 'test'], type=str, default='train', help='Mode: train, test')
    parser.add_argument('-d', '--dataset', choices=['cora', 'pubmed', 'citeseer'], type=str, default='cora', help='dataset to use')

    return parser.parse_args()

def main():
    args = get_args()
    
    if args.mode == 'train':
        train.driver(args.dataset)
    elif args.mode == 'test':
        pass

if __name__ == "__main__":
    main()