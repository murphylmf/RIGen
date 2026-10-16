from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

import torch


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    ''
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


def compute_rectangular_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
) -> torch.Tensor:
    ''
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    mask = torch.ones(length)
    if left_ramp > 0:
        mask[:left_ramp] = 0
    if right_ramp > 0:
        mask[-right_ramp:] = 0
    return mask


@dataclass(frozen=True)
class DimensionInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


@dataclass(frozen=True)
class DimensionIntervals:
    ''

    intervals: list[DimensionInterval]


@dataclass(frozen=True)
class LatentIntervals:
    ''

    original_shape: torch.Size
    dimension_intervals: tuple[DimensionIntervals, ...]


                                                                                                             
SplitOperation = Callable[[int], DimensionIntervals]
                                                                                                               
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor | None]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(intervals=[DimensionInterval(start=0, end=length, left_ramp=0, right_ramp=0)])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[torch.Tensor | None]]:
    return [slice(0, None)], [None]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


                                                                             
                 
                                                                             


def split_by_size(size: int, overlap: int) -> SplitOperation:
    ''
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}")

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        intervals = [
            DimensionInterval(start=0, end=size, left_ramp=0, right_ramp=overlap),
            *(
                DimensionInterval(
                    start=i * (size - overlap),
                    end=i * (size - overlap) + size,
                    left_ramp=overlap,
                    right_ramp=overlap,
                )
                for i in range(1, amount - 1)
            ),
            DimensionInterval(
                start=(amount - 1) * (size - overlap), end=dimension_size, left_ramp=overlap, right_ramp=0
            ),
        ]
        return DimensionIntervals(intervals=intervals)

    return split


def split_temporal_causal(size: int, overlap: int) -> SplitOperation:
    ''
    non_causal_split = split_by_size(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        dim_intervals = non_causal_split(dimension_size)
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_temporal(tile_size_frames: int, overlap_frames: int) -> SplitOperation:
    ''
    non_causal_split = split_by_size(tile_size_frames, overlap_frames)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= tile_size_frames:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        dim_intervals = non_causal_split(dimension_size)
        modified_intervals = [
            replace(interval, end=interval.end + 1, right_ramp=0) for interval in dim_intervals.intervals[:-1]
        ] + [replace(dim_intervals.intervals[-1], right_ramp=0)]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_by_count_temporal_causal(num_tiles: int, overlap: int = 0) -> SplitOperation:
    ''
    non_causal_split = split_by_count(num_tiles, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        dim_intervals = non_causal_split(dimension_size)
        if len(dim_intervals.intervals) <= 1:
            return dim_intervals
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_by_count(num_tiles: int, overlap: int = 0) -> SplitOperation:
    ''
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")

    def split(dim_size: int) -> DimensionIntervals:
        if num_tiles > dim_size:
            raise ValueError(
                f"num_tiles ({num_tiles}) exceeds dim_size ({dim_size}). Cannot assign at least 1 unit per tile."
            )
        if num_tiles == 1:
            return DEFAULT_SPLIT_OPERATION(dim_size)

        total = dim_size + overlap * (num_tiles - 1)
        tile_size = total // num_tiles
        remainder = total % num_tiles

        base_intervals = split_by_size(tile_size, overlap)(dim_size - remainder).intervals

                                                                                        
        intervals: list[DimensionInterval] = []
        for i, iv in enumerate(base_intervals):
            shift = min(i, remainder)
            grow = 1 if i < remainder else 0
            intervals.append(replace(iv, start=iv.start + shift, end=iv.end + shift + grow))

        return DimensionIntervals(intervals=intervals)

    return split


                                                                             
                    
                                                                             


def identity_mapping_operation(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor | None]]:
    ''
    out_slices: list[slice] = []
    masks: list[torch.Tensor | None] = []
    for iv in intervals.intervals:
        out_slices.append(slice(iv.start, iv.end))
        masks.append(compute_trapezoidal_mask_1d(iv.end - iv.start, iv.left_ramp, iv.right_ramp))
    return out_slices, masks


class Tile(NamedTuple):
    ''

    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[torch.Tensor | None, ...]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: list[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                                                                 
                one = torch.ones(1)

                view_shape[dim_idx] = 1
                per_dimension_masks.append(one.view(*view_shape))
                continue

                                                                                                
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

                                                                                             
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def create_tiles_from_intervals_and_mappers(
    intervals: LatentIntervals,
    mappers: list[MappingOperation],
) -> list[Tile]:
    full_dim_input_slices: list[list[slice]] = []
    full_dim_output_slices: list[list[slice]] = []
    full_dim_masks_1d: list[list[torch.Tensor | None]] = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        input_slices = [slice(interval.start, interval.end) for interval in dimension_intervals.intervals]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        n_intervals = len(input_slices)
        if len(output_slices) != n_intervals or len(masks_1d) != n_intervals:
            raise ValueError(
                f"Axis {axis_index}: mapper produced {len(output_slices)} output slices and "
                f"{len(masks_1d)} masks for {n_intervals} input intervals"
            )
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    return [
        Tile(in_coords=in_coord, out_coords=out_coord, masks_1d=mask_1d)
        for in_coord, out_coord, mask_1d in zip(
            itertools.product(*full_dim_input_slices),
            itertools.product(*full_dim_output_slices),
            itertools.product(*full_dim_masks_1d),
            strict=True,
        )
    ]


def create_tiles(
    latent_shape: torch.Size,
    splitters: list[SplitOperation],
    mappers: list[MappingOperation],
) -> list[Tile]:
    if len(splitters) != len(latent_shape):
        raise ValueError(
            f"Number of splitters must be equal to number of dimensions in latent shape, "
            f"got {len(splitters)} and {len(latent_shape)}"
        )
    if len(mappers) != len(latent_shape):
        raise ValueError(
            f"Number of mappers must be equal to number of dimensions in latent shape, "
            f"got {len(mappers)} and {len(latent_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape, strict=True)]
    latent_intervals = LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(latent_intervals, mappers)


                                                                             
                           
                                                                             


@dataclass(frozen=True)
class DimensionTilingConfig:
    ''

    num_tiles: int
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")

    @classmethod
    def from_tile_size(cls, dim_size: int, tile_size: int, overlap: int = 0) -> DimensionTilingConfig:
        ''
        split_op = split_by_size(tile_size, overlap)
        intervals = split_op(dim_size)
        return cls(num_tiles=len(intervals.intervals), overlap=overlap)


@dataclass(frozen=True)
class TileCountConfig:
    ''

    frames: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0)
    height: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0)
    width: DimensionTilingConfig = DimensionTilingConfig(num_tiles=1, overlap=0)
