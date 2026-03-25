from __future__ import annotations

from pathlib import Path

import fire

from visualization_toolkit.core import (
	build_standalone_payload,
	save_detection_visuals,
	save_segmentation_visuals,
	show_detection_pointcloud_window,
	show_segmentation_pointcloud_window,
)


class VisualizeCLI:
	def standalone(
		self,
		config: str = 'configs/experiment/det/swint_v0p075.yaml',
		dataset_root: str = 'data/nuscenes-mini',
		split: str = 'val',
		index: int = 0,
		checkpoint: str | None = None,
		out_dir: str = 'outputs/visualize',
		topk: int = 30,
		score_thresh: float = 0.3,
		show_gt: bool = True,
		show_pred: bool = True,
		point_mode: str = 'default',
		point_size: float = 1.0,
		box_line_width: float = 1.0,
		show_window: bool = True,
		save_images: bool = True,
	) -> dict[str, str]:
		payload = build_standalone_payload(
			config,
			split=split,
			index=index,
			dataset_root=dataset_root,
			checkpoint=checkpoint,
			topk=topk,
			score_thresh=score_thresh,
		)
		if show_window:
			if payload['task'] == 'det':
				show_detection_pointcloud_window(
					payload,
					show_gt=show_gt,
					show_pred=show_pred,
					point_mode=point_mode,
					point_size=point_size,
					box_line_width=box_line_width,
					topk=topk,
					score_thresh=score_thresh,
				)
			else:
				show_segmentation_pointcloud_window(
					payload,
					show_gt=show_gt,
					point_mode=point_mode,
					point_size=point_size,
					box_line_width=box_line_width,
				)
		if not save_images:
			return {}
		if payload['task'] == 'det':
			result = save_detection_visuals(
				payload,
				out_dir,
				show_gt=show_gt,
				show_pred=show_pred,
				point_mode=point_mode,
				point_size=point_size,
				box_line_width=box_line_width,
				topk=topk,
				score_thresh=score_thresh,
			)
		else:
			result = save_segmentation_visuals(
				payload,
				out_dir,
				show_gt=show_gt,
				point_mode=point_mode,
				point_size=point_size,
				box_line_width=box_line_width,
			)
		return {key: str(value) for key, value in result.items()}


def main():
	fire.Fire(VisualizeCLI)


if __name__ == '__main__':
	main()
