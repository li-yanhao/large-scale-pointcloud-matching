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
from model.Superglue.dataset import pts_from_meter_to_pixel, pts_from_pixel_to_meter
from model.Superglue.dataset import input_transforms as superglue_input_transforms
import torchvision.transforms.functional as TF
# import model.Superglue.dataset


# 1. create database
# 2. query a lidar scan (birdview image)


parser = argparse.ArgumentParser(description='GlobalLocalization')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test'])
# parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--dataset_dir', type=str, default='/media/li/lavie/dataset/birdview_dataset/', help='dataset_dir')
# parser.add_argument('--dataset_dir', type=str, default='/media/li/LENOVO/dataset/kitti/lidar_odometry/birdview_dataset', help='dataset_dir')
parser.add_argument('--sequence', type=str, default='00', help='sequence_all')
parser.add_argument('--sequence_database', type=str, default='juxin_1023_map', help='sequence_database')
parser.add_argument('--sequence_query', type=str, default='juxin_1023_locate', help='sequence_query')
parser.add_argument('--use_different_sequence', type=bool, default=True, help='use_different_sequence')
# parser.add_argument('--dataset_dir', type=str, default='/home/li/Documents/wayz/image_data/dataset', help='dataset_dir')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
# parser.add_argument('--from_scratch', type=bool, default=True, help='from_scratch')
parser.add_argument('--pretrained_embedding', type=bool, default=False, help='pretrained_embedding')
parser.add_argument('--num_similar_neg', type=int, default=4, help='number of similar negative samples')
parser.add_argument('--use_gpu', type=bool, default=True, help='use_gpu')
parser.add_argument('--positive_search_radius', type=float, default=8, help='positive_search_radius')
parser.add_argument('--saved_model_path', type=str,
                    default='/media/li/lavie/dataset/birdview_dataset/saved_models', help='saved_model_path')
parser.add_argument('--epochs', type=int, default=120, help='epochs')
parser.add_argument('--num_clusters', type=int, default=64, help='num_clusters')
parser.add_argument('--final_dim', type=int, default=256, help='final_dim')
parser.add_argument('--meters_per_pixel', type=float, default=0.20, help='meters_per_pixel')
parser.add_argument('--top_k', type=int, default=1, help='top_k')
args = parser.parse_args()


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
    u, s, v = torch.svd(cov, some=True, compute_uv=True)

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
    """
    :param target_keypoints: N * 2
    :param source_keypoints: N * 2
    :return: T_target_source_best: 4 * 4
             score: float
    """
    assert(target_keypoints.shape == source_keypoints.shape)
    num_matches = len(target_keypoints)
    if num_matches < 6:
        return None, None

    T_target_source = compute_relative_pose(target_keypoints, source_keypoints)
    n, k = 1000, 6
    selections = np.random.choice(num_matches, (n, k), replace=True)

    score = -99999
    T_target_source_best = None
    distance_tolerance = 0.5
    selection_best = None
    for selection in selections:
        T_target_source = compute_relative_pose(target_keypoints[selection], source_keypoints[selection])
        # centers_aligned_A = R.dot(centers_A[idx_A, :]) + T
        diff = source_keypoints @ T_target_source[:2,:2].transpose() + T_target_source[:2,2] - target_keypoints
        distances_squared = np.sum(diff[:, :2] * diff[:, :2], axis=1)

        if score < (distances_squared < distance_tolerance ** 2).sum():
            T_target_source_best = T_target_source
            # score = np.sum(diff * diff, axis=1).mean()
            score = (distances_squared < distance_tolerance ** 2).sum()
            selection_best = np.where(distances_squared < distance_tolerance ** 2)
            # selection_best = selection
    # matches0_amended = np.ones(match_result["matches0"].reshape(-1).shape[0]) * (-1)
    # matches0_amended[correspondences_valid[0, selection_best]] = correspondences_valid[1, selection_best]
    # match_result_amended = {"matches0": matches0_amended}

    return T_target_source_best, score



def compute_relative_pose_with_ransac_test(target_keypoints, source_keypoints):
    """
    :param target_keypoints: N * 2
    :param source_keypoints: N * 2
    :return: T_target_source_best: 4 * 4
             score: float
    """
    assert(target_keypoints.shape == source_keypoints.shape)
    num_matches = len(target_keypoints)
    n, k = 1000, 10
    if num_matches < k:
        return None, None

    target_keypoints = torch.Tensor(target_keypoints)
    source_keypoints = torch.Tensor(source_keypoints)


    selections = np.random.choice(num_matches, (n, k), replace=True)

    target_sub_keypoints = target_keypoints[selections] # N * k * 2
    source_sub_keypoints = source_keypoints[selections] # N * k * 2
    target_centers = target_sub_keypoints.mean(dim=1) # N * 2
    source_centers = source_sub_keypoints.mean(dim=1) # N * 2
    target_sub_keypoints_centered = target_sub_keypoints - target_centers.unsqueeze(1)
    source_sub_keypoints_centered = source_sub_keypoints - source_centers.unsqueeze(1)
    cov = source_sub_keypoints_centered.transpose(1, 2) @ target_sub_keypoints_centered
    u, s, v = torch.svd(cov) # u: N*2*2, s: N*2, v: N*2*2

    v_neg = v.clone()
    v_neg[:,:, 1] *= -1

    rot_mats_neg = v_neg @ u.transpose(1, 2)
    rot_mats_pos = v @ u.transpose(1, 2)
    determinants = torch.det(rot_mats_pos)

    rot_mats_neg_list = [rot_mat_neg for rot_mat_neg in rot_mats_neg]
    rot_mats_pos_list = [rot_mat_neg for rot_mat_neg in rot_mats_pos]

    rot_mats_list = [rot_mat_pos if determinant > 0 else rot_mat_neg for (determinant, rot_mat_pos, rot_mat_neg) in zip(determinants, rot_mats_pos_list, rot_mats_neg_list)]
    rotations = torch.stack(rot_mats_list) # N * 2 * 2
    translations = torch.einsum("nab,nb->na", -rotations, source_centers) + target_centers # N * 2
    diff = source_keypoints @ rotations.transpose(1,2) + translations.unsqueeze(1) - target_keypoints
    distances_squared = torch.sum(diff * diff, dim=2)

    distance_tolerance = 1.0
    scores = (distances_squared < (distance_tolerance**2)).sum(dim=1)
    score = torch.max(scores)
    best_index = torch.argmax(scores)
    rotation = rotations[best_index]
    translation = translations[best_index]
    T_target_source = torch.cat((rotation, translation[...,None]), dim=1)
    T_target_source = torch.cat((T_target_source, torch.Tensor([[0,0,1]])), dim=0)
    return T_target_source, score


def superglue_match(target_image, source_image, resolution : int, matching=None, device=None):
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
    # confidence = pred['matching_scores0'][0].cpu().detach().numpy()

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

    saved_model_file_spinetvlad = os.path.join(args.saved_model_path, 'model-to-check-top1.pth.tar')
    model_checkpoint = torch.load(saved_model_file_spinetvlad, map_location=lambda storage, loc: storage)
    model.load_state_dict(model_checkpoint)
    print("Loaded spinetvlad checkpoints from \'{}\'.".format(saved_model_file_spinetvlad))

    # images_dir = os.path.join(args.dataset_dir, args.sequence)
    database_images_dir = os.path.join(args.dataset_dir, args.sequence)
    query_images_dir = os.path.join(args.dataset_dir, args.sequence)
    database_images_info = query_images_info = make_images_info(
        struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence + '.txt'))
    # database_images_info, query_images_info = train_test_split(images_info_validate, test_size=0.2,
    #                                                            random_state=10)

    if args.use_different_sequence:
        database_images_info = make_images_info(
            struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_database + '.txt'))
        query_images_info = make_images_info(
            struct_filename=os.path.join(args.dataset_dir, 'struct_file_' + args.sequence_query + '.txt'))
        database_images_dir = os.path.join(args.dataset_dir, args.sequence_database)
        query_images_dir = os.path.join(args.dataset_dir, args.sequence_query)

    image_database = ImageDatabase(images_info=database_images_info,
                                   images_dir=database_images_dir, model=model,
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

    saved_model_file_superglue = os.path.join(args.saved_model_path, 'spsg-rotation-invariant.pth.tar')
    # saved_model_file_superglue = os.path.join(args.saved_model_path, 'superglue-juxin.pth.tar')

    model_checkpoint = torch.load(saved_model_file_superglue, map_location=lambda storage, loc: storage)
    matching.load_state_dict(model_checkpoint)
    print("Loaded superglue checkpoints from \'{}\'.".format(saved_model_file_superglue))

    translation_errors = []
    rotation_errors = []

    success_records = []
    accumulated_distance = 0
    last_T_w_source_gt = None

    true_count = 0
    for query_image_info in tqdm(query_images_info):
        query_results = image_database.query_image(
            image_filename=os.path.join(query_images_dir, query_image_info['image_file']), num_results=args.top_k+1)
        if args.use_different_sequence:
            # avoid the same image from database
            query_results = query_results[:args.top_k]
        else:
            query_results = query_results[1:args.top_k+1]
        # print('query_result: \n{}'.format(query_results))
        best_score = -1
        T_w_source_best = None
        min_inliers = 20
        max_inliers = 30
        # min_inliers = 0
        # max_inliers = 0
        resolution = int(100 / args.meters_per_pixel)
        for query_result in query_results:
            target_image = Image.open(os.path.join(database_images_dir, query_result['image_file']))
            source_image = Image.open(os.path.join(query_images_dir, query_image_info['image_file']))
            target_kpts, source_kpts = superglue_match(target_image, source_image, resolution, matching)
            target_kpts_in_meters = pts_from_pixel_to_meter(target_kpts, args.meters_per_pixel)
            source_kpts_in_meters = pts_from_pixel_to_meter(source_kpts, args.meters_per_pixel)
            print("len of target_kpts_in_meters:", len(target_kpts_in_meters))
            T_target_source, score = compute_relative_pose_with_ransac_test(target_kpts_in_meters, source_kpts_in_meters)

            # T_target_source, score = compute_relative_pose_with_ransac(target_kpts_in_meters, source_kpts_in_meters)
            # T_target_source, score = compute_relative_pose(target_kpts_in_meters, source_kpts_in_meters), len(target_kpts)
            if score is None:
                continue
            if score > best_score and score > min_inliers:
                best_score = score
                # TODO: the way we handle the se3 may be inappropriate
                T_target_source = np.array([[T_target_source[0,0], T_target_source[0,1], 0, T_target_source[0,2]],
                                            [T_target_source[1,0], T_target_source[1,1], 0, T_target_source[1,2]],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
                # T_target_source = np.array(
                #     [[1, 0, 0, 0],
                #      [0, 1, 0, 0],
                #      [0, 0, 1, 0],
                #      [0, 0, 0, 1]])
                T_w_target = np.hstack([R.from_quat(query_result['orientation'][[1,2,3,0]]).as_matrix(), query_result['position'].reshape(3,1)])
                T_w_target = np.vstack([T_w_target, np.array([0,0,0,1])])
                T_w_source_best = T_w_target @ T_target_source
            # print(T_target_source)
            INVERSE_AUGMENTATION = False
            if INVERSE_AUGMENTATION:
                # tf = superglue_input_transforms(args.meters_per_pixel, 180)
                target_image_inv = TF.rotate(target_image, 180)
                target_kpts_inv, source_kpts = superglue_match(target_image_inv, source_image, resolution, matching)
                target_kpts_in_meters_inv = pts_from_pixel_to_meter(target_kpts_inv, args.meters_per_pixel)
                source_kpts_in_meters = pts_from_pixel_to_meter(source_kpts, args.meters_per_pixel)
                # T_target_source, score = compute_relative_pose_with_ransac_test(target_kpts_in_meters_inv,
                #                                                                 source_kpts_in_meters)
                T_target_inv_source, score = compute_relative_pose_with_ransac(target_kpts_in_meters_inv, source_kpts_in_meters)
                # T_target_inv_source, score = compute_relative_pose(target_kpts_in_meters_inv, source_kpts_in_meters), len(target_kpts)
                if score is None:
                    continue
                if score > best_score and score > min_inliers:
                    best_score = score
                    # Since the target image is rotated by 180 degrees, its pose is rotated in the same manner
                    T_target_source = np.array(
                        [[-T_target_source[0, 0], -T_target_source[0, 1], 0, -T_target_source[0, 2]],
                         [-T_target_source[1, 0], -T_target_source[1, 1], 0, -T_target_source[1, 2]],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
                    # T_target_source = np.array(
                    #     [[1, 0, 0, 0],
                    #      [0, 1, 0, 0],
                    #      [0, 0, 1, 0],
                    #      [0, 0, 0, 1]])
                    T_w_target = np.hstack([R.from_quat(query_result['orientation'][[1, 2, 3, 0]]).as_matrix(),
                                            query_result['position'].reshape(3, 1)])
                    T_w_target = np.vstack([T_w_target, np.array([0, 0, 0, 1])])
                    T_w_source_best = T_w_target @ T_target_source
            if best_score > max_inliers:
                break

        # ground truch pose
        T_w_source_gt = np.hstack([R.from_quat(query_image_info['orientation'][[1, 2, 3, 0]]).as_matrix(),
                                   query_image_info['position'].reshape(3, 1)])
        T_w_source_gt = np.vstack([T_w_source_gt, np.array([0, 0, 0, 1])])

        # record travelled distance
        if last_T_w_source_gt is not None:
            T_last_current = np.linalg.inv(last_T_w_source_gt) @ T_w_source_gt
            accumulated_distance += np.sqrt(T_last_current[:3,3] @ T_last_current[:3,3])
        last_T_w_source_gt = T_w_source_gt

        if T_w_source_best is not None:

            delta_T_w_source = np.linalg.inv(T_w_source_best) @  T_w_source_gt
            delta_translation = np.sqrt(delta_T_w_source[:3,3] @ delta_T_w_source[:3,3])
            delta_degree = np.arccos(min(1, 0.5 * (np.trace(delta_T_w_source[:3,:3]) - 1))) / np.pi * 180
            print('Translation error: {}'.format(delta_translation))
            print('Rotation error: {}'.format(delta_degree))
            translation_errors.append(delta_translation)
            rotation_errors.append(delta_degree)
            success_records.append((accumulated_distance, True))
        else:
            print('Global localization failed.')
            success_records.append((accumulated_distance, False))
            pass
            # translation_errors.append(float('nan'))
        # print('accumulated_distance', accumulated_distance)

    translation_errors = np.array(translation_errors)
    rotation_errors = np.array(rotation_errors)
    print('Mean translation error: {}'.format(translation_errors.mean()))
    for r in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print('Percentage of translation errors under {} m: {}'.format(r, (translation_errors<r).sum() / len(translation_errors)))
    for theta in [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print('Percentage of rotation errors under {} degrees: {}'.format(theta, (rotation_errors<theta).sum() / len(rotation_errors)))

    plt.scatter(np.linspace(0, 50, num=len(translation_errors)), np.array(translation_errors))
    plt.show()

    travelled_distances = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    probabilities = []
    for thres_distance in travelled_distances:
        probabilities.append(localization_probability(accumulated_distance, np.array(success_records), thres_distance))
    plt.plot(travelled_distances, probabilities, lw=1)
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label="Luck")
    plt.xlabel("travelled distance")
    plt.ylabel("probabilities")
    plt.show()

    translation_errors = translation_errors[~np.isnan(translation_errors)]
    rotation_errors = rotation_errors[~np.isnan(rotation_errors)]

    trans_err_avg = translation_errors.mean()
    trans_err_std = translation_errors - trans_err_avg
    trans_err_std = np.sqrt((trans_err_std * trans_err_std).mean())
    print("average translation error: {}".format(trans_err_avg))
    print("standard deviation of translation error: {}".format(trans_err_std))

    rotation_err_avg = rotation_errors.mean()
    rotation_err_std = rotation_errors - rotation_err_avg
    rotation_err_std = np.sqrt((rotation_err_std * rotation_err_std).mean())
    print("average rotation_errors error: {}".format(rotation_err_avg))
    print("standard deviation of rotation_errors error: {}".format(rotation_err_std))

    print("recall: {}".format(len(translation_errors) / len(query_images_info)))


    pass


def localization_probability(total_distance, localization_results, thres_distance):
    num_intervals = int(total_distance / thres_distance) + 1
    success = np.zeros(num_intervals)
    for result in localization_results:
        success[int(result[0] / thres_distance)] = 1
    return 1 - success.mean()


if __name__ == '__main__':
    ######################
    # visualization test #
    ######################
    # visualize_netvlad()

    #########################
    # pose computation test #
    #########################
    # N = 100
    # alpha = np.random.rand() * 3.14
    # rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    # translation = np.random.randn(2, 1) * 20
    # T_target_source = np.hstack([rotation, translation])
    # T_target_source = np.vstack([T_target_source, np.array([0, 0, 1])])
    # T_source_target = np.linalg.inv(T_target_source)
    #
    # print("T_target_source ground truth: \n", T_target_source)
    # target_points = np.random.randn(N, 2)
    # source_points = (T_source_target[:2, :2] @ target_points.transpose()).transpose() + T_source_target[:2, 2]
    #
    # T_target_source = compute_relative_pose_with_ransac(target_points, source_points)
    # print("T_target_source once: \n", T_target_source)
    # T_target_source = compute_relative_pose(target_points, source_points)
    # print("T_target_source ransac: \n", T_target_source)

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
