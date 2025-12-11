import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--epoch", type=int, help="No. of epochs for training", default=30)
    parser.add_argument("--root_dir", type=str, help="Directory of dataset", default="afhq_v2")
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.0002)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument("--beta1", type=float, help="First betas of optimizer", default=0.5)
    parser.add_argument("--beta2", type=float, help="Second betas of optimizer", default=0.999)
    parser.add_argument("--num_dims", type=int, help="No. dimensions of latent space", default=100)
    parser.add_argument("--step", type=int, help="No. of epoch to save generate images", default=3)
    parser.add_argument("--save_pth", type=str, help="Directory save anything", default="output")
    parser.add_argument("--decay", type=float, help="Decay of EMA", default=0.9996)
    parser.add_argument("--num_classes", type=int, help="Number of classes", default=2)
    parser.add_argument("--embed_dims", type=int, help="Number of embedding dims", default=2)

    args = parser.parse_args()

    return args