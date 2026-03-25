import fire

from tools.data_converter import nuscenes_converter as nuscenes_converter
from tools.data_converter.create_gt_database import create_groundtruth_database


def nuscenes_data_prep(
	root_path,
	info_prefix,
	version,
	dataset_name,
	out_dir,
	max_sweeps=10,
	load_augmented=None,
):
	"""Prepare data related to nuScenes dataset.

	Related data consists of '.pkl' files recording basic infos,
	2D annotations and groundtruth database.

	Args:
	    root_path (str): Path of dataset root.
	    info_prefix (str): The prefix of info filenames.
	    version (str): Dataset version.
	    dataset_name (str): The dataset class name.
	    out_dir (str): Output directory of the groundtruth database info.
	    max_sweeps (int): Number of input consecutive frames. Default: 10
	"""
	if load_augmented is None:
		# otherwise, infos must have been created, we just skip.
		nuscenes_converter.create_nuscenes_infos(
			root_path,
			info_prefix,
			version=version,
			max_sweeps=max_sweeps,
		)

	# if version == "v1.0-test":
	#     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
	#     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
	#     return

	# info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
	# info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
	# nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
	# nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

	create_groundtruth_database(
		dataset_name,
		root_path,
		info_prefix,
		f'{out_dir}/{info_prefix}_infos_train.pkl',
		load_augmented=load_augmented,
	)


def main(
	dataset: str,
	root_path: str = './data/nuscenes',
	version: str = 'v1.0',
	max_sweeps: int = 10,
	out_dir: str = './data/nuscenes',
	extra_tag: str = 'nuscenes',
	painted: bool = False,
	virtual: bool = False,
	# workers: int = 4, # not used
):

	load_augmented = None
	if virtual:
		if painted:
			load_augmented = 'mvp'
		else:
			load_augmented = 'pointpainting'

	if dataset == 'nuscenes' and version != 'v1.0-mini':
		train_version = f'{version}-trainval'
		nuscenes_data_prep(
			root_path=root_path,
			info_prefix=extra_tag,
			version=train_version,
			dataset_name='NuScenesDataset',
			out_dir=out_dir,
			max_sweeps=max_sweeps,
			load_augmented=load_augmented,
		)
		test_version = f'{version}-test'
		nuscenes_data_prep(
			root_path=root_path,
			info_prefix=extra_tag,
			version=test_version,
			dataset_name='NuScenesDataset',
			out_dir=out_dir,
			max_sweeps=max_sweeps,
			load_augmented=load_augmented,
		)
	elif dataset == 'nuscenes' and version == 'v1.0-mini':
		train_version = f'{version}'
		nuscenes_data_prep(
			root_path=root_path,
			info_prefix=extra_tag,
			version=train_version,
			dataset_name='NuScenesDataset',
			out_dir=out_dir,
			max_sweeps=max_sweeps,
			load_augmented=load_augmented,
		)


if __name__ == '__main__':
	fire.Fire(main)
