import collections
from importlib import import_module


class Compose:
	"""Compose multiple transforms sequentially.

	Args:
	    transforms (Sequence[dict | callable]): Sequence of transform object or
	        config dict to be composed.
	"""

	def __init__(self, transforms):
		assert isinstance(transforms, collections.abc.Sequence)
		self.transforms = []
		for transform in transforms:
			if callable(transform):
				self.transforms.append(transform)
			elif isinstance(transform, collections.abc.Mapping):
				transform_cfg = dict(transform)
				transform_type = transform_cfg.pop('type')
				pipelines_module = import_module('datasets.pipelines')
				registry = getattr(pipelines_module, 'PIPELINE_REGISTRY')
				if transform_type not in registry:
					raise KeyError(f'Unsupported pipeline type: {transform_type}')
				self.transforms.append(registry[transform_type](**transform_cfg))
			else:
				raise TypeError('transform must be callable or a dict')

	def __call__(self, data):
		"""Call function to apply transforms sequentially.

		Args:
		    data (dict): A result dict contains the data to transform.

		Returns:
		   dict: Transformed data.
		"""

		for t in self.transforms:
			data = t(data)
			if data is None:
				return None
		return data

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		for t in self.transforms:
			format_string += '\n'
			format_string += f'    {t}'
		format_string += '\n)'
		return format_string
