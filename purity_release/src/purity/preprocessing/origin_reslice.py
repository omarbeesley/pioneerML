"""Partition ATAR slices by simulation mixing origin and recompute mean times.

This preserves origin_reslice.py's grouping, including its dense sorted labels.
It requires simulation origin bookkeeping and is not detector-only reslicing.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._io import ensure_output, record_run

REQUIRED = ("atar_slice", "atar_origin", "atar_t", "atar_E", "atar_slice_mean_t")


def reslice_arrays(slices, origins, times, energies):
    slices = np.asarray(slices)
    n = len(slices)
    origins = np.asarray(origins)
    times = np.asarray(times, float)
    energies = np.asarray(energies, float)
    if not (len(origins) == len(times) == len(energies) == n):
        raise ValueError("ATAR slice, origin, time, and energy arrays must have equal lengths")
    if n == 0:
        return [], []
    key = slices.astype(np.int64) * 100000 + (origins.astype(np.int64) + 10)
    _, new = np.unique(key, return_inverse=True)
    mean_times = np.zeros(n)
    for label in np.unique(new):
        mask = new == label
        weight = energies[mask].sum()
        mean_times[mask] = ((times[mask] * energies[mask]).sum() / weight
                            if weight > 0 else times[mask].mean())
    return new.tolist(), mean_times.tolist()


def reslice(input_path, output_path, batch_size=10_000):
    output_path = ensure_output(output_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    parquet = pq.ParquetFile(input_path)
    missing = set(REQUIRED) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing reslicing columns: {sorted(missing)}")
    schema = parquet.schema_arrow
    # The original pandas writer infers list<int64> / list<float64> for these outputs.
    schema = schema.set(schema.get_field_index("atar_slice"), pa.field("atar_slice", pa.list_(pa.int64())))
    schema = schema.set(schema.get_field_index("atar_slice_mean_t"), pa.field("atar_slice_mean_t", pa.list_(pa.float64())))
    rows = split_events = 0
    with pq.ParquetWriter(output_path, schema) as writer:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            data = {name: table.column(name).to_pylist() for name in REQUIRED}
            new_slices, new_times = [], []
            for values in zip(*(data[name] for name in REQUIRED[:4])):
                labels, times = reslice_arrays(*values)
                split_events += len(np.unique(labels)) != len(np.unique(values[0]))
                new_slices.append(labels)
                new_times.append(times)
            for name, values in (("atar_slice", new_slices), ("atar_slice_mean_t", new_times)):
                field = schema.field(name)
                table = table.set_column(schema.get_field_index(name), field, pa.array(values, type=field.type))
            writer.write_table(table.cast(schema))
            rows += table.num_rows
    record_run(output_path, [input_path], __file__, {
        "rows": rows, "split_events": int(split_events), "batch_size": batch_size,
        "grouping": "dense sorted ids of original_slice * 100000 + origin + 10",
        "requires_simulation_origin": True,
    })
    print(f"Resliced {rows:,} events ({split_events:,} split) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    reslice(args.input, args.output, args.batch_size)


if __name__ == "__main__":
    main()
