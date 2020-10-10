import argparse
import os
from torch.utils.data import DataLoader
import visdom
from tqdm import tqdm
from model.Descriptor.descnet import *
from model.Descriptor.descriptor_dataset import *
from model.Descriptor.loss_function import *

parser = argparse.ArgumentParser(description='DescriptorModel')
# parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
parser.add_argument('--dataset_dir', type=str, default="/media/admini/My_data/submap_database/05", help='dataset_dir')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
parser.add_argument('--triplet_loss_margin', type=float, default=0.5, help='triplet_loss_margin')
parser.add_argument('--alpha', type=float, default=2, help='alpha')
parser.add_argument('--load_checkpoints', type=bool, default=True, help='load_checkpoints')
parser.add_argument('--checkpoints_dir', type=str,
                    default="/media/admini/My_data/submap_database/checkpoints", help='checkpoints_dir')

args = parser.parse_args()


def train():
    h5_filename = os.path.join(args.dataset_dir, "submap_segments.h5")
    correspondences_filename = os.path.join(args.dataset_dir, "correspondences.txt")

    descriptor_dataset = DescriptorDataset(h5_filename, correspondences_filename, mode='train')
    train_loader = DataLoader(descriptor_dataset, batch_size=1, shuffle=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DgcnnModel(k=10, feature_dims=[64, 128, 512], emb_dims=[256, 256], output_classes=256)
    if args.load_checkpoints:
        dgcnn_model_checkpoint = torch.load(os.path.join(args.checkpoints_dir, "descriptor-dgcnn-kitti00.pth"),
                                            map_location=lambda storage, loc: storage)
        model.load_state_dict(dgcnn_model_checkpoint)
        print("Loaded model checkpoints from \'{}\'.".format(
            os.path.join(args.checkpoints_dir, "descriptor-dgcnn-kitti00.pth")))

    # opt = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-6)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    model = model.to(dev)

    model.train()
    # triplet_loss = nn.TripletMarginLoss(margin=args.triplet_loss_margin, p=2, reduction='sum')
    triplet_loss = ESMTripletLoss(device=dev)
    pairwise_loss = PairwiseLoss()

    viz = visdom.Visdom()
    win_loss = viz.scatter(X=np.asarray([[0, 0]]))

    item_idx = 0

    accumulated_loss = 0
    num_accumulation = 0

    with tqdm(train_loader) as tq:
        for item in tq:
            opt.zero_grad()
            anchor, positive, negative = item
            anchor_desc = model(torch.Tensor(anchor['segment']).to(dev))
            positive_desc = model(torch.Tensor(positive['segment']).to(dev))
            negative_desc = model(torch.Tensor(negative['segment']).to(dev))
            loss = triplet_loss(anchor_desc, positive_desc, negative_desc) \
                   + args.alpha * pairwise_loss(anchor_desc, positive_desc)

            loss.backward()
            opt.step()

            accumulated_loss += loss.item()
            num_accumulation += 1

            if (num_accumulation >= 100):
                viz.scatter(X=np.array([[item_idx, float(accumulated_loss / num_accumulation)]]),
                            name="train-loss",
                            win=win_loss,
                            update="append")
                print("loss: {}".format(accumulated_loss / num_accumulation))
                num_accumulation = 0
                accumulated_loss = 0
            item_idx += 1
            if item_idx % 10000 == 0:
                torch.save(model.state_dict(), os.path.join(args.checkpoints_dir, "descriptor-dgcnn-kitti00.pth"))


if __name__ == '__main__':
    train()
