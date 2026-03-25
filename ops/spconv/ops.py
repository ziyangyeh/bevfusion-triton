from __future__ import annotations

import importlib

try:
	from .src import spconv_triton
except Exception:  # pragma: no cover
	spconv_triton = None

# Keep the historical extension available as a pytest reference backend.
# Runtime sparse ops should go through the local Triton implementation.
_legacy_module = None


def _get_legacy_module():
	global _legacy_module
	if _legacy_module is None:
		_legacy_module = importlib.import_module('.ops_legacy', package=__package__)
	return _legacy_module


# Expose the legacy extension for parity tests and low-level reference checks.
sparse_conv_ext = _get_legacy_module().sparse_conv_ext


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
	ndim = len(input_size)
	output_size = []
	for i in range(ndim):
		size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[
			i
		] + 1
		if kernel_size[i] == -1:
			output_size.append(1)
		else:
			output_size.append(size)
	return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
	ndim = len(input_size)
	output_size = []
	for i in range(ndim):
		if kernel_size[i] == -1:
			raise ValueError("deconv don't support kernel_size < 0")
		size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
		output_size.append(size)
	return output_size


def get_indice_pairs(
	indices,
	batch_size,
	spatial_shape,
	ksize=3,
	stride=1,
	padding=0,
	dilation=1,
	out_padding=0,
	subm=False,
	transpose=False,
	grid=None,
):
	if spconv_triton is not None:
		return spconv_triton.get_indice_pairs(
			indices,
			batch_size,
			spatial_shape,
			ksize=ksize,
			stride=stride,
			padding=padding,
			dilation=dilation,
			out_padding=out_padding,
			subm=subm,
			transpose=transpose,
			grid=grid,
		)
	return _get_legacy_module().get_indice_pairs(
		indices,
		batch_size,
		spatial_shape,
		ksize=ksize,
		stride=stride,
		padding=padding,
		dilation=dilation,
		out_padding=out_padding,
		subm=subm,
		transpose=transpose,
		grid=grid,
	)


def indice_conv(
	features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False
):
	if spconv_triton is not None:
		return spconv_triton.indice_conv(
			features,
			filters,
			indice_pairs,
			indice_pair_num,
			num_activate_out,
			inverse=inverse,
			subm=subm,
		)
	return _get_legacy_module().indice_conv(
		features,
		filters,
		indice_pairs,
		indice_pair_num,
		num_activate_out,
		inverse=inverse,
		subm=subm,
	)


def fused_indice_conv(
	features, filters, bias, indice_pairs, indice_pair_num, num_activate_out, inverse, subm
):
	if spconv_triton is not None:
		return spconv_triton.fused_indice_conv(
			features,
			filters,
			bias,
			indice_pairs,
			indice_pair_num,
			num_activate_out,
			inverse,
			subm,
		)
	return _get_legacy_module().fused_indice_conv(
		features,
		filters,
		bias,
		indice_pairs,
		indice_pair_num,
		num_activate_out,
		inverse,
		subm,
	)


def indice_conv_backward(
	features, filters, out_bp, indice_pairs, indice_pair_num, inverse=False, subm=False
):
	if spconv_triton is not None:
		return spconv_triton.indice_conv_backward(
			features,
			filters,
			out_bp,
			indice_pairs,
			indice_pair_num,
			inverse=inverse,
			subm=subm,
		)
	return _get_legacy_module().indice_conv_backward(
		features,
		filters,
		out_bp,
		indice_pairs,
		indice_pair_num,
		inverse=inverse,
		subm=subm,
	)


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
	if spconv_triton is not None:
		return spconv_triton.indice_maxpool(
			features, indice_pairs, indice_pair_num, num_activate_out
		)
	return _get_legacy_module().indice_maxpool(
		features, indice_pairs, indice_pair_num, num_activate_out
	)


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs, indice_pair_num):
	if spconv_triton is not None:
		return spconv_triton.indice_maxpool_backward(
			features,
			out_features,
			out_bp,
			indice_pairs,
			indice_pair_num,
		)
	return _get_legacy_module().indice_maxpool_backward(
		features, out_features, out_bp, indice_pairs, indice_pair_num
	)
