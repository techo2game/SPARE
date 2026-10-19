import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from pareconv.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(_C.working_dir)  # SPARE repository root
_C.exp_name = 'Real3DAD'
_C.output_dir = osp.join(_C.working_dir, 'output')
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'wandb_events')
_C.feature_dir = osp.join(_C.output_dir, 'features')
_C.registration_dir = osp.join(_C.output_dir, 'registration')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()
_C.data.dataset_root = os.environ.get('SPARE_REAL3D_ROOT', osp.join(_C.root_dir, 'data', 'Real3D'))

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005  # 5mm noise in normalized space
_C.train.augmentation_rotation = 0.5  # ±0.5 radians (~±28°) additional rotation
_C.train.augmentation_crop = True  # Plane crop (like ModelNet)
_C.train.point_keep_ratio = 0.5  # Keep 70% when cropping
_C.train.rotation_range = 180.0  # ±180° range for initial transformation
_C.train.translation_range = 0.2  # ±0.2 range in normalized space
_C.train.matching_radius = 0.05  # Same as 3DMatch

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 4
_C.test.point_limit = 30000  # Same as training (we load raw .pcd files)

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1  # Same as 3DMatch
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rmse_threshold = 0.2  # Same as 3DMatch
_C.eval.rre_threshold = 15.0  # 15° rotation error
_C.eval.rte_threshold = 0.3  # Same as 3DMatch
_C.eval.feat_rre_threshold = 20.0

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.1  # Same as 3DMatch
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 60  # Longer training due to smaller dataset
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.num_neighbors = [16, 16, 16, 16]  # Reduced from 35 (stage 4 has ~32-77 points)
_C.backbone.init_voxel_size = 0.025  # Same as 3DMatch
_C.backbone.subsample_ratio = 2
_C.backbone.kernel_size = 4
_C.backbone.share_nonlinearity = False
_C.backbone.conv_way = 'edge_conv'
_C.backbone.use_xyz = True
_C.backbone.init_dim = 96
_C.backbone.output_dim = 256

# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05  # Same as 3DMatch
_C.model.num_points_in_patch = 64

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 256
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 768
_C.geotransformer.hidden_dim = 192
_C.geotransformer.output_dim = 192
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 0.1  # Same as 3DMatch
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1  # Same as 3DMatch
_C.fine_matching.confidence_threshold = 0.005
_C.fine_matching.num_hypotheses = 2000
_C.fine_matching.num_refinement_steps = 5
_C.fine_matching.use_encoder_re_feats = True

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.1  # Same as 3DMatch
_C.fine_loss.negative_radius = 0.2  # 2x positive_radius
_C.fine_loss.positive_margin = 0.1
_C.fine_loss.negative_margin = 1.4

# loss - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_ri_loss = 1.0
_C.loss.weight_fine_re_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
