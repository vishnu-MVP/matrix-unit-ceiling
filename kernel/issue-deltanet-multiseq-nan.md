# Gated DeltaNet produces non-finite activations when a batch contains more than one sequence

### Name and Version

```
$ ./build/bin/llama-cli --version
version: 6b790a9 (vulkan: handle misalignment in conv_2d and conv_3d (#29365))
built with cc (GCC) 11.5.0 for x86_64-amazon-linux
```

Built from a clean checkout, no local modifications (`git status` empty), 0 build errors.

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

libllama (core)

### Command line

```shell
./build/bin/llama-imatrix -m qwen3-next-Q8_0.gguf -f calib.txt \
    --chunks 2 -c 512 -b 2048 -ub 512 -t 16
```

### Problem description

On models using Gated DeltaNet (Qwen3-Next family - hybrid Gated-DeltaNet + attention),
the DeltaNet layers emit non-finite values as soon as a single batch contains more than
one sequence, i.e. whenever `n_seqs > 1`.

This is trivially reproducible through `llama-imatrix`, because the logical batch size
(`-b`) being larger than the micro-batch (`-ub`) is what causes more than one sequence to
be packed into a batch. The condition is not specific to imatrix - it is a property of the
DeltaNet op, so it also affects parallel serving and speculative decoding, both of which
routinely place several sequences in one batch.

The distinguishing factor is **only** the batch geometry. Changing `-b` while holding every
other argument constant flips the failure on and off deterministically:

| run | `-b` | `-ub` | sequences per batch | result |
|---|---|---|---|---|
| A | 2048 | 512 | > 1 | **`non-finite values detected in blk.0.ssm_out.weight`** |
| B | 512 | 512 | 1 | completes normally |

The first DeltaNet layer (`blk.0`) is already affected, so the corruption originates inside
the layer rather than accumulating across depth.

**Impact**

- `llama-imatrix` cannot produce an importance matrix for this architecture at default
  settings, which blocks any imatrix-guided quantisation of these models.
- Parallel serving (`--parallel N`) and speculative decoding both batch multiple sequences
  together and are affected by the same code path.

**Workaround**

Force one sequence per batch by setting `-b` equal to `-ub`:

```shell
./build/bin/llama-imatrix -m model.gguf -f calib.txt --chunks 2 -c 512 -b 512 -ub 512
```

**Note on scope.** I originally found this while testing a custom CPU kernel, so I
re-verified it on a clean upstream tree with no local patches to rule out my own code.
Both runs above are from that clean build. I have not attempted a fix, since the correct
handling of per-sequence recurrent state in this op is better judged by someone familiar
with the DeltaNet implementation.

### First Bad Commit

Not bisected. Present at `6b790a9`.

### Relevant log output

```
########## A) -b 2048 -ub 512  (n_seqs > 1) ##########
system_info: n_threads = 16 (n_threads_batch = 16) / 16 | CPU : SSE3 = 1 | SSSE3 = 1 |
AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 |
AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | AMX_INT8 = 1 | ...

compute_imatrix: computing over 2 chunks with batch_size 512
E collect_imatrix: non-finite values detected in blk.0.ssm_out.weight

########## B) -b 512 -ub 512   (n_seqs == 1) ##########
system_info: n_threads = 16 (n_threads_batch = 16) / 16 | CPU : SSE3 = 1 | SSSE3 = 1 |
AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 |
AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | AMX_INT8 = 1 | ...

compute_imatrix: computing over 2 chunks with batch_size 512
(no error; run completes)
```

Hardware: Intel Xeon (Sapphire Rapids), 8 physical cores / 16 threads, 123 GiB RAM.
