from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def encode_coors_to_keys_kernel(
	coors_ptr,
	keys_ptr,
	num_rows,
	num_dims,
	size0,
	size1,
	size2,
	size3,
):
	row = tl.program_id(axis=0)
	if row >= num_rows:
		return
	base = row * num_dims
	c0 = tl.load(coors_ptr + base + 0).to(tl.int64)
	key = tl.full((), -1, tl.int64)
	if c0 >= 0:
		key = c0
		if num_dims >= 2:
			c1 = tl.load(coors_ptr + base + 1).to(tl.int64)
			key = key * size1 + c1
		if num_dims >= 3:
			c2 = tl.load(coors_ptr + base + 2).to(tl.int64)
			key = key * size2 + c2
		if num_dims >= 4:
			c3 = tl.load(coors_ptr + base + 3).to(tl.int64)
			key = key * size3 + c3
	tl.store(keys_ptr + row, key)


@triton.jit
def decode_keys_to_coors_kernel(
	keys_ptr,
	coors_ptr,
	num_rows,
	num_dims,
	size0,
	size1,
	size2,
	size3,
):
	row = tl.program_id(axis=0)
	if row >= num_rows:
		return
	key = tl.load(keys_ptr + row).to(tl.int64)
	base = row * num_dims
	if key < 0:
		i = 0
		while i < num_dims:
			tl.store(coors_ptr + base + i, -1)
			i += 1
		return
	if num_dims >= 4:
		c3 = key % size3
		key = key // size3
		tl.store(coors_ptr + base + 3, c3.to(tl.int32))
	if num_dims >= 3:
		c2 = key % size2
		key = key // size2
		tl.store(coors_ptr + base + 2, c2.to(tl.int32))
	if num_dims >= 2:
		c1 = key % size1
		key = key // size1
		tl.store(coors_ptr + base + 1, c1.to(tl.int32))
	c0 = key % size0
	tl.store(coors_ptr + base + 0, c0.to(tl.int32))


__all__ = [
	'decode_keys_to_coors_kernel',
	'encode_coors_to_keys_kernel',
]
