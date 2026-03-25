def extract_result_dict(results, key):
	"""Extract and return the data corresponding to key in result dict.

	``results`` is a dict output from `pipeline(input_dict)`, which is the
	    loaded data from ``Dataset`` class.
	The data terms inside may be wrapped in list, tuple and DataContainer, so
	    this function essentially extracts data from these wrappers.

	Args:
	    results (dict): Data loaded using pipeline.
	    key (str): Key of the desired data.

	Returns:
	    np.ndarray | torch.Tensor | None: Data term.
	"""
	if key not in results.keys():
		return None
	# results[key] may be data or list[data] or tuple[data]
	# data may be wrapped inside a container with a `_data` payload
	data = results[key]
	if isinstance(data, (list, tuple)):
		data = data[0]
	if hasattr(data, '_data'):
		data = data._data
	return data
