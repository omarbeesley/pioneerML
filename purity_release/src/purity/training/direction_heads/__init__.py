"""Direction-head ensemble: extra positron-direction heads trained on a frozen PURITY network.

The network's own direction head (dir_attn_pool + positron_dir_head) is retrained many times on the
frozen network's cached inputs, each copy from its own random start, on its own random half of
the events and in its own batch order; the polar angles of the copies are averaged at inference
(models/main.py, PURITY_DIR_HEADS). Single heads scatter in the acceptance bias as much as
successive training epochs do; the average of 25 removes that scatter (docs/training.md).

    dump_inputs        run the frozen network and save what its direction head reads
    train              train sets of heads on those inputs (variants.json: the sets trained)
    build_checkpoint   write a weights-only checkpoint carrying the network and N heads
    dirhead            the stand-alone head and the cached-input reader
"""
