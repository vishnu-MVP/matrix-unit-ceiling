# ggml-cpu/amx: reload tile configuration when the palette changes

## What this fixes

`ggml_tile_config_init()` installs the AMX tile palette behind a one-shot
`thread_local` flag:

```cpp
inline void ggml_tile_config_init(void) {
    static thread_local bool done = false;

    if (done) {
        return;
    }
    ...
    _tile_loadconfig(&tc);
    done = true;
}
```

AMX tile geometry is **process state**, not an instruction operand: the shapes are
installed with `_tile_loadconfig` and every subsequent `_tile_dpbssd` /`_tile_loadd`
interprets its operands according to whatever palette was most recently loaded on that
thread. The `done` latch therefore makes the *first* palette installed on a thread
permanent for the lifetime of that thread.

Today `ggml` ships exactly one palette, so the latch is harmless and the bug is latent.
It becomes a live, silent correctness bug the moment a second tile shape is introduced -
which is the natural way to add support for a quantisation type whose block size is not
`TILE_K` or `8 * TILE_K`.

## Why it is worth fixing now

The failure mode is unusually hard to diagnose, because nothing crashes and nothing warns.
When a second palette is latched out, the tile instructions still execute; they simply
interpret their operands with the wrong row/column geometry. The result is numerically
wrong output from *the other* quantisation type in the same model, while the type that
installed the palette keeps working correctly.

I encountered this while experimenting with an additional tile shape locally. A model
containing two different quantisation types produced correct output for the type whose
palette had been installed first and silently corrupt output for the other, with no error
reported anywhere.

Because the corruption is silent and type-dependent, it is very easy to misattribute it to
the arithmetic of whichever kernel is under development rather than to global palette
state.

## The change

Track the configuration that is actually loaded instead of latching on first use, and
reload only when it differs:

```cpp
inline void ggml_tile_config_init(void) {
    static thread_local tile_config_t loaded = {};

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.start_row  = 0;
    tc.rows[0] = 8;   tc.colsb[0] = 64;
    /* ... unchanged ... */

    if (memcmp(&loaded, &tc, sizeof(tc)) == 0) {
        return;
    }

    _tile_loadconfig(&tc);
    loaded = tc;
}
```

Behaviour is unchanged for the current single-palette code: the first call loads the
palette, every later call compares 64 bytes and returns. The comparison is negligible
against the GEMM it guards, and it removes the hazard for any future tile shape.

The `memcmp` is well defined here because `tile_config_t` has no padding: `palette_id`,
`start_row` and `reserved_0[14]` occupy the first 16 bytes, `colsb[16]` (`uint16_t`) is
naturally aligned at offset 16, and `rows[16]` follows at offset 48, for exactly the
64 bytes the palette descriptor requires. The largest member alignment is 2, so there are
no interior or trailing padding bytes to compare.

## Testing

Tested on a Sapphire Rapids Xeon with `amx_tile` / `amx_int8` / `avx512_vnni`, built with
`GGML_NATIVE=ON`, GCC 11.5, Linux, against master at `c06f841`.

- `test-backend-ops -o MUL_MAT -b CPU`: **1917/1917 tests passed**, `Backend CPU: OK`.
- `test-backend-ops perf -o MUL_MAT`: no measurable throughput cost. Three runs each,
  mean over ~203 cases: patched 1.890 GFLOPS vs unpatched 1.897 GFLOPS, a 0.4%
  difference against a 1.6% spread across the patched runs themselves. The added
  comparison is 64 bytes against a tile GEMM.
- Verified against a local tree that installs a second palette: with the latch present the
  secondary quantisation type is silently corrupted; with this change both types produce
  correct output.
## Notes

This is a standalone correctness fix and does not add any new quantisation type. I am
happy to split the `memcmp` into an explicit field comparison, or to gate the reload
behind a palette id, if maintainers prefer.
