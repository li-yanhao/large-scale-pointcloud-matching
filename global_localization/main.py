from model.Birdview.dataset import *
from model.Birdview.base_model import *
from model.Birdview.netvlad import *
from model.Superglue.matching import *
# from Superglue.Superglue import *
# from Superglue.superpoint import *
from scipy.spatial.transform import Rotation as R

from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import argparse

# 1. create database
# 2. query a lidar scan (birdview image)



parser = argparse.ArgumentParser(description='GlobalLocalization')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
# parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/admini/lavie/dataset/birdview_dataset/', help='dataset_dir')
# parser.add_argument('--dataset_dir', type=str, default='/media/li/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset', help='dataset_dir')
parser.add_argument('--sequence', type=str, default='08', help='sequence')

# parser.add_argument('--dataset_dir', type=str, default='/home/li/Documents/wayz/image_data/dataset', help='dataset_dir')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
# parser.add_argument('--from_scratch', type=bool, default=True, help='from_scratch')
parser.add_argument('--pretrained_embedding', type=bool, default=False, help='pretrained_embedding')
parser.add_argument('--num_similar_neg', type=int, default=4, help='number of similar negative samples')
# parser.add_argument('--margin', type=float, default=0.5, help='margin')
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
# def visualize_netvlad(model, database_images_info, query_images_info, images_dir):
def visualize_netvlad():
    base_model = BaseModel(300, 300)
    dim = 256

    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=dim, alpha=1.0, outdim=args.final_dim)
    model = EmbedNet(base_model, net_vlad)

    saved_model_file = os.path.join(args.saved_model_path, 'model-lazy-triplet.pth.tar')
    model_checkpoint = torch.load(saved_model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_checkpoint)
    print("Loaded model checkpoints from \'{}\'.".format(saved_model_file))


    images_dir = os.path.join(args.dataset_dir, args.sequence)
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence + '.txt'))
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

    plotted_images = []
    num_results = 3

    query_images_info_visualized = np.random.choice(query_images_info, 8, replace=False)
    for query_image_info in query_images_info_visualized:
        # query_results = image_database.query_image(queried_image_filename, num_results=num_results)
        query_results = image_database.query_image(
            image_filename=os.path.join(images_dir, query_image_info['image_file']), num_results=num_results)

        queried_image = Image.open(os.path.join(images_dir, query_image_info['image_file']))
        # result_image = Image.open(os.path.join(images_dir, query_results[0]['image_file']))
        result_images = [Image.open(os.path.join(images_dir, result['image_file'])) for result in query_results]

        plotted_images += [queried_image] + result_images
    plot_images(plotted_images, num_results + 1)


def plot_images(images, cols):
    plt.figure()
    # plt.subplot(1, 1, 1)
    # plt.imshow(images[0])

    for i in range(0, len(images)):
        # row = i // cols
        # col = i % cols
        plt.subplot(len(images) // cols, cols, i + 1)
        plt.imshow(images[i])
    plt.show()


def compute_relative_pose(target_points, source_points):
    # target_points: N * 2
    # source_points: N * 2
    assert(len(target_points) == len(target_points))

    target_points = torch.Tensor(target_points)
    source_points = torch.Tensor(source_points)

    target_centers = target_points.mean(dim=0)
    source_centers = source_points.mean(dim=0)

    target_points_centered = target_points - target_centers
    source_points_centered = source_points - source_centers

    cov = source_points_centered.transpose(0, 1) @ target_points_centered
    u, s, v = torch.svd(cov, some=False, compute_uv=True)

    v_neg = v.clone()
    v_neg[:, 1] *= -1
    rot_mat_neg = v_neg @ u.transpose(0, 1)
    rot_mat_pos = v @ u.transpose(0, 1)

    rot_mat = rot_mat_pos if torch.det(rot_mat_pos) > 0 else rot_mat_neg
    trans = -rot_mat @ source_centers + target_centers

    rot_mat = np.array(rot_mat)
    trans = np.array(trans).reshape(-1, 1)
    T_target_source_restored = np.hstack([rot_mat, trans])
    T_target_source_restored = np.vstack([T_target_source_restored, np.array([0, 0, 1])])

    # print('T_target_source_restored:\n', T_target_source_restored)
    return T_target_source_restored



def retrieve_inter_images_pose(image_target, image_source):
    target_keypoints, source_keypoints = superglue_match(image_target, image_source)
    compute_relative_pose(target_keypoints, source_keypoints)
    pass


def compute_relative_pose_with_ransac(target_keypoints, source_keypoints):
    num_matches = len(target_keypoints)
    if num_matches < 6:
        return None, None

    T_target_source = compute_relative_pose(target_keypoints, source_keypoints)
    n, k = 1000, 6
    selections = np.random.choice(num_matches, (n, k), replace=True)

    score = -99999
    T_target_source_best = None
    pixel_tolerance_= 1
    selection_best = None
    for selection in selections:
        T_target_source = compute_relative_pose(target_keypoints[selection], source_keypoints[selection])
        # centers_aligned_A = R.dot(centers_A[idx_A, :]) + T
        diff = source_keypoints @ T_target_source[:2,:2].transpose() + T_target_source[:2,2] - target_keypoints
        distances_squared = np.sum(diff[:, :2] * diff[:, :2], axis=1)

        if score < (distances_squared < pixel_tolerance_ ** 2).sum():
            T_target_source_best = T_target_source
            # score = np.sum(diff * diff, axis=1).mean()
            score = (distances_squared < pixel_tolerance_ ** 2).sum()
            selection_best = np.where(distances_squared < pixel_tolerance_ ** 2)
            # selection_best = selection
    # matches0_amended = np.ones(match_result["matches0"].reshape(-1).shape[0]) * (-1)
    # matches0_amended[correspondences_valid[0, selection_best]] = correspondences_valid[1, selection_best]
    # match_result_amended = {"matches0": matches0_amended}

    return T_target_source_best, score


def superglue_match(target_image, source_image, matching=None, device=None, resolution=400):
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'Superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if matching is None:
        matching = Matching(config).eval().to(device)

    def frame2tensor(frame, device):
        tf = transforms.Compose([
            transforms.Resize(size=(resolution, resolution)),
            transforms.ToTensor(),
        ])
        return tf(frame).float()[None].to(device)
        # return torch.from_numpy(frame/255.).float()[None, None].to(device)



    target_frame_tensor = frame2tensor(target_image, device)
    source_frame_tensor = frame2tensor(source_image, device)

    pred = matching({'image0': target_frame_tensor, 'image1': source_frame_tensor})
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().detach().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    # print(mkpts0, mkpts1)
    return mkpts0, mkpts1


def pipeline_test():
    torch.set_grad_enabled(False)
    # Define model for embedding
    base_model = BaseModel(300, 300)
    net_vlad = NetVLAD(num_clusters=args.num_clusters, dim=256, alpha=1.0, outdim=args.final_dim)
    model = EmbedNet(base_model, net_vlad)

    saved_model_file = os.path.join(args.saved_model_path, 'model-lazy-triplet.pth.tar')
    model_checkpoint = torch.load(saved_model_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_checkpoint)
    print("Loaded model checkpoints from \'{}\'.".format(saved_model_file))

    images_dir = os.path.join(args.dataset_dir, args.sequence)
    images_info_validate = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence + '.txt'))
    database_images_info, query_images_info = train_test_split(images_info_validate, test_size=0.1,
                                                               random_state=10)
    image_database = ImageDatabase(images_info=database_images_info,
                                   images_dir=images_dir, model=model,
                                   generate_database=True,
                                   transforms=input_transforms())

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'Superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        }
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matching = Matching(config).eval().to(device)

    translation_errors = []
    true_count = 0
    for query_image_info in tqdm(query_images_info):
        query_results = image_database.query_image(
            image_filename=os.path.join(images_dir, query_image_info['image_file']), num_results=3)
        # print('query_result: \n{}'.format(query_results))
        best_score = -1
        T_w_source_best = None
        min_inliers = 30
        resolution = 400
        meters_per_pixel = 100 / resolution
        for query_result in query_results:
            target_image = Image.open(os.path.join(images_dir, query_result['image_file']))
            source_image = Image.open(os.path.join(images_dir, query_image_info['image_file']))

            target_kpts, source_kpts = superglue_match(target_image, source_image, matching, resolution=resolution)
            # T_target_source, score = compute_relative_pose_with_ransac(target_kpts, source_kpts)
            T_target_source, score = compute_relative_pose(target_kpts, source_kpts), len(target_kpts)
            if score is None:
                continue
            if score > best_score and score > min_inliers:
                best_score = score
                T_target_source = np.array([[T_target_source[0,0], T_target_source[0,1], 0, T_target_source[0,2]*meters_per_pixel],
                                            [T_target_source[1,0], T_target_source[1,1], 0, T_target_source[1,2]*meters_per_pixel],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
                T_target_source = np.array(
                    [[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
                T_w_target = np.hstack([R.from_quat(query_result['orientation'][[1,2,3,0]]).as_matrix(), query_result['position'].reshape(3,1)])
                T_w_target = np.vstack([T_w_target, np.array([0,0,0,1])])
                T_w_source_best = T_w_target @ T_target_source
            # print(T_target_source)
            # T_target_source = compute_relative_pose(target_kpts, source_kpts)
        # print(T_w_source_best)
        if T_w_source_best is not None:
            T_w_source_gt = np.hstack([R.from_quat(query_image_info['orientation'][[1, 2, 3, 0]]).as_matrix(),
                                    query_image_info['position'].reshape(3, 1)])
            T_w_source_gt = np.vstack([T_w_source_gt, np.array([0, 0, 0, 1])])
            delta_T_w_source = np.linalg.inv(T_w_source_best) @  T_w_source_gt
            delta_translation = np.sqrt(delta_T_w_source[:3,3] @ delta_T_w_source[:3,3])
            print('Translation error: {}'.format(delta_translation))
            translation_errors.append(delta_translation)
        else:
            print('Global localization failed.')
    translation_errors = np.array(translation_errors)
    print('Mean translation error: {}'.format(translation_errors.mean()))
    for r in range(1,10):
        print('Percentage under {} m: {}'.format(r, (translation_errors<r).sum() / len(translation_errors)))

    plt.scatter(np.linspace(0, 50, num=len(translation_errors)), np.array(translation_errors))
    plt.show()

    pass


if __name__ == '__main__':
    ######################
    # visualization test #
    ######################
    # visualize_netvlad()

    #########################
    # pose computation test #
    #########################
    N = 100
    alpha = np.random.rand() * 3.14
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    translation = np.random.randn(2, 1) * 20
    T_target_source = np.hstack([rotation, translation])
    T_target_source = np.vstack([T_target_source, np.array([0, 0, 1])])
    T_source_target = np.linalg.inv(T_target_source)

    print("T_target_source ground truth: \n", T_target_source)
    target_points = np.random.randn(N, 2)
    source_points = (T_source_target[:2, :2] @ target_points.transpose()).transpose() + T_source_target[:2, 2]

    T_target_source = compute_relative_pose_with_ransac(target_points, source_points)
    print("T_target_source once: \n", T_target_source)
    T_target_source = compute_relative_pose(target_points, source_points)
    print("T_target_source ransac: \n", T_target_source)

    ###########################
    # Superglue matching test #
    ###########################
    # target_image = np.array(Image.open('/media/admini/lavie/dataset/birdview_dataset/00/submap_302.png'))
    # source_image = np.array(Image.open('/media/admini/lavie/dataset/birdview_dataset/00/submap_303.png'))
    # target_kpts, source_kpts = superglue_match(target_image, source_image)
    # # T_target_source = compute_relative_pose_with_ransac(target_kpts, source_kpts)
    # # print(T_target_source)
    # T_target_source = compute_relative_pose(target_kpts, source_kpts)
    # print(T_target_source)

    ##############################################
    # global localization pipeline matching test #
    ##############################################
    pipeline_test()
