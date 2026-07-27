# Universal Benchmark

This walkthrough explains how the `universal_benchmark` executable operates and
what each flag means.

## Step-by-Step Operation

1. Generate the mixture of operations you will be running, which consists of a
certain percentage of reads, inserts, erases, updates, and upserts. We
pre-generate the mixture to avoid having to compute which operation to run
while timing the workload
2. Pre-generate all keys we will be inserting into the table. We can calculate
an upper bound on the number of keys being inserted based on the prefill
percentage (`--prefill`) and the number of inserts and upserts we’ll be
performing. Again, pre-generating the keys avoids doing it while timing the
workload. We don’t need to pre-generate the values, since the values are the
same for all operations
3. Initialize the table, pre-sized to a specified capacity (`--initial-capacity`)
4. Fill up the table in advance to the specified prefill percentage
5. Run the pre-generated mixture of operations (`--total-ops`) and time how long
it takes to complete all of them
6. Report the details of the benchmark configuration and the quantities
measured, including time elapsed, throughput, and (optionally) memory usage
samples.

## Flags

These flags are passed to CMake and set various compile-time parameters for the
benchmark:

`-DUNIVERSAL_KEY`
: sets the type of the table Key (by default, this is `uint64_t`). Support for
new keys can be added in `universal_gen.hh`

`-DUNIVERSAL_VALUE`
: sets the type of the table Value.  Support for new values can be added in
`universal_gen.hh`

`-DUNIVERSAL_TABLE`
: sets the type of the hashmap being benchmarked (by default, this is
`LIBCUCKOO`).  Support for new maps can be added in
`universal_table_wrapper.hh`

`-DUNIVERSAL_TRACKING_ALLOCATOR`
: enables memory usage sampling

These flags control the mixture of operations that will be run in the
benchmark. They are interpreted as whole number percentages, and must sum to
100:

`--reads`
: percentage of operations that are reads

`--inserts`
: percentage of operations that are inserts

`--erases`
: percentage of operations that are erases

`--updates`
: percentage of operations that are updates

`--upserts`
: percentage of operations that are upserts

These flags control some parameters about what the table will look like before
the operation mixture is run:

`--initial-capacity`
: sets the initial number of elements the table is pre-sized
to hold, as a power of 2. So if you pass 25 as the value of this flag, the
table will be pre-sized to hold 2^25 elements

`-–prefill`
: sets the percentage to fill the pre-sized table to before running
the operations.
    
These flags control a few other details of the benchmark:
    
`--total-ops`
: the total number of operations to run (in the timed phase), as a percentage
of the initial table capacity. So specifying `--initial-capacity 25 --total-ops
90` means run `90%` of `2^25`, or about `30198988`, operations. Specifying it
as a percentage makes it easy to specify the final table capacity without
having to do too much calculation or write a large number.

`--num-threads`
: the number of threads to use in all phases of the benchmark

`--seed`
: the seed to use for the rng, or 0 if you want to use a randomly
generated seed. If `--num-threads` is 1 and you specify a specific seed, the test
should be repeatable.
