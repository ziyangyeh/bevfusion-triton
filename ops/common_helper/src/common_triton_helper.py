from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _bilinear_interpolate_geometry(
	y,
	x,
	height,
	width,
):
	invalid = (y < -1.0) | (y > height) | (x < -1.0) | (x > width)

	y = tl.maximum(y, 0)
	x = tl.maximum(x, 0)

	height_last = height - 1
	width_last = width - 1
	y_low = tl.minimum(y.to(tl.int32), height_last)
	x_low = tl.minimum(x.to(tl.int32), width_last)
	y_high = tl.minimum(y_low + 1, height_last)
	x_high = tl.minimum(x_low + 1, width_last)
	y_is_edge = y_low == height_last
	x_is_edge = x_low == width_last

	y_low_cast = y_low.to(y.dtype)
	x_low_cast = x_low.to(x.dtype)
	y = tl.where(y_is_edge, y_low_cast, y)
	x = tl.where(x_is_edge, x_low_cast, x)

	ly = y - y_low_cast
	lx = x - x_low_cast
	one = tl.full(ly.shape, 1, dtype=ly.dtype)
	hy = one - ly
	hx = one - lx
	return invalid, y_low, x_low, y_high, x_high, hy, hx, ly, lx


@triton.jit
def bilinear_interpolate_kernel(
	plane_ptr,
	y,
	x,
	height,
	width,
):
	invalid, y_low, x_low, y_high, x_high, hy, hx, ly, lx = _bilinear_interpolate_geometry(
		y, x, height, width
	)

	pos1 = y_low * width + x_low
	pos2 = y_low * width + x_high
	pos3 = y_high * width + x_low
	pos4 = y_high * width + x_high

	v1 = tl.load(plane_ptr + pos1, mask=~invalid, other=0)
	v2 = tl.load(plane_ptr + pos2, mask=~invalid, other=0)
	v3 = tl.load(plane_ptr + pos3, mask=~invalid, other=0)
	v4 = tl.load(plane_ptr + pos4, mask=~invalid, other=0)
	w1 = (hy * hx).to(v1.dtype)
	w2 = (hy * lx).to(v1.dtype)
	w3 = (ly * hx).to(v1.dtype)
	w4 = (ly * lx).to(v1.dtype)

	val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4).to(v1.dtype)
	val = tl.where(invalid, 0, val)
	return val


@triton.jit
def bilinear_interpolate_gradient_kernel(
	y,
	x,
	height,
	width,
):
	invalid, y_low, x_low, y_high, x_high, hy, hx, ly, lx = _bilinear_interpolate_geometry(
		y, x, height, width
	)

	w1 = (hy * hx).to(y.dtype)
	w2 = (hy * lx).to(y.dtype)
	w3 = (ly * hx).to(y.dtype)
	w4 = (ly * lx).to(y.dtype)

	neg_one = tl.zeros_like(y_low) - 1
	y_low = tl.where(invalid, neg_one, y_low)
	x_low = tl.where(invalid, neg_one, x_low)
	y_high = tl.where(invalid, neg_one, y_high)
	x_high = tl.where(invalid, neg_one, x_high)

	w1 = tl.where(invalid, 0, w1)
	w2 = tl.where(invalid, 0, w2)
	w3 = tl.where(invalid, 0, w3)
	w4 = tl.where(invalid, 0, w4)
	return w1, w2, w3, w4, y_low, x_low, y_high, x_high


__all__ = [
	'bilinear_interpolate_kernel',
	'bilinear_interpolate_gradient_kernel',
]
