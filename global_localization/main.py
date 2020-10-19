from model.Birdview.dataset import *
from sklearn.model_selection import train_test_split

import argparse


# 1. create database
# 2. query a lidar scan (birdview image)



parser = argparse.ArgumentParser(description='GlobalLocalization')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
# parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
parser.add_argument('--sequence', type=str, default='08', help='sequence')

# parser.add_argument('--dataset_dir', type=str, default='/home/li/Documents/wayz/image_data/dataset', help='dataset_dir')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
# parser.add_argument('--from_scratch', type=bool, default=True, help='from_scratch')
parser.add_argument('--pretrained_embedding', type=bool, default=False, help='pretrained_embedding')
parser.add_argument('--num_similar_neg', type=int, default=4, help='number of similar negative samples')
parser.add_argument('--margin', type=float, default=0.5, help='margin')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
# parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--positive_search_radius', type=float, default=8, help='positive_search_radius')
parser.add_argument('--negative_filter_radius', type=float, default=50, help='negative_filter_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/admini/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--epochs', type=int, default=120, help='epochs')
# parser.add_argument('--load_checkpoints', type=bool, default=True, help='load_checkpoints')
parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
parser.add_argument('--final_dim', type=int, default=256, help='final_dim')
args = parser.parse_args()

# TODO
def generate_database(model, database_images_info, query_images_info, images_dir):
    images_dir = os.path.join(args.dataset_dir, args.sequence)
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_validate + '.txt'))
    database_images_info, query_images_info = train_test_split(images_info_validate, test_size=0.2,
                                                                                 random_state=2)
    image_database = ImageDatabase(images_info=database_images_info,
                                   images_dir=images_dir, model=model,
                                   generate_database=True,
                                   transforms=input_transforms())
    true_count = 0
    for query_image_info in tqdm(query_images_info):
        query_results = image_database.query_image(
            image_filename=os.path.join(images_dir, query_image_info['image_file']), num_results=3)
        # print('query_result: \n{}'.format(query_results))
        for query_result in query_results:
            diff = query_image_info['position'] - query_result['position']
            if np.sqrt(diff @ diff) < args.positive_search_radius:
                true_count += 1
                break
    print("Precision: {}".format(true_count / len(query_images_info)))


def query_image():
    pass