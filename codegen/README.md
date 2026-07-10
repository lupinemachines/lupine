Codegen works via a human-in-the-loop system. It's quite challenging to build a codegen engine that can correctly
infer what parameters should be sent and received so we instead have a two-step process.

First, `annotationgen.py` reads in all the `nvml.h`, `cuda.h`, et al headers and copies the function signatures
into `annotations.h`. This file is intended to be modified by humans. In particular, the `@param` annotations
have significant meanings.

Specifically, the order of `@param` annotations indicates the order in which the parameters are sent or received.
`SEND_ONLY`, `RECV_ONLY`, and `SEND_RECV` indicate the directions the parameter is transferred in. Other arguments
available are `NULL_TERMINATED` (to indicate that this is a null-terminated string), or `LENGTH:<param>` and
`SIZE:<value>` to specify the size (aka width) of the parameter. If `LENGTH:<param>` is specified, `<param>` must
be placed in front of the parameter referencing it, otherwise the generated code will not compile.

Client routing can also be annotated for handles that belong to a specific LUPINE
server connection. `@routingkey <kind> <param>` selects the connection for the
generated client wrapper before it writes the RPC. Supported kinds are
`DEVICE`, `CONTEXT`, `MODULE`, `FUNCTION`, `STREAM`, `EVENT`, `MEMORY_POOL`,
`GRAPH`, `GRAPH_NODE`, `GRAPH_EXEC`, and `DEVICEPTR`.
`@routingkey CURRENT_CONTEXT` routes through the client's current CUDA context
owner. `DEVICE` and `CONTEXT` routing is inferred from the first non-pointer
`CUdevice` or `CUcontext` parameter, so those annotations are only needed when
the routing key is not the first matching parameter.

NVML wrappers use the same mechanism. A by-value `nvmlDevice_t` parameter
infers `NVML_DEVICE` routing: the generated client resolves its owning server
and substitutes the remote handle before marshalling the request. Device lookup
APIs without an input handle use `@routingkey ALL` together with
`@recordowner NVML_DEVICE <output>` to search every server and translate the
returned remote handle back to the client's virtual device handle.

Inverse CUDA device lookups use `@routingkey ALL <output>`. The generated
client tries each local and remote route, then translates the successful
route-local device back to its virtual client ordinal before returning it.

`@disabled client` leaves server/RPC generation enabled while requiring a
manual client implementation with the original API name. These manual symbols
remain part of the generated client function map.

Generated wrappers can record ownership for handles returned by an API with
`@recordowner <kind> <param>`. Supported owner kinds are `CONTEXT`, `MODULE`,
`FUNCTION`, `STREAM`, `EVENT`, `MEMORY_POOL`, `GRAPH`, `GRAPH_NODE`,
`GRAPH_EXEC`, and `DEVICEPTR`. Ownership is recorded only after the CUDA call
returns `CUDA_SUCCESS`.

Some APIs need small, reusable behaviors beyond plain parameter send/receive
layout. These should be expressed as annotations when the behavior is generic
enough for more than one API or can be described without API-specific C++.
Currently, `@crossservercopy <dst> <src> <bytes> [STREAM:<param>] [ASYNC]` adds
a generated client fallback for device-to-device copies whose source and
destination pointers are owned by different server connections. The fallback
routes through the client-side cross-server copy helper.

Keep function-specific code in manual files when the behavior cannot be
described by annotations without embedding C++ for that exact API. Typical
manual cases include callback forwarding, CUDA graph capture bookkeeping,
stdout capture, local host allocation, local file loading, cross-server event
waits, manual client/server wire framing, and server-side deferred-copy queue
management.

With the annotations in place, `codegen.py` reads in the annotations and generates the RPC server and client.

The motivation for this approach is grounded in codegen being very good at ensuring that the RPC server and client
match in behavior but not very good at determining the specifics of what should be sent and received. Humans, on
the other hand, are very good at the latter but not the former. Therefore, this specification file allows humans
to edit and build out the "layout" of the RPC call without having to worry about ensuring the server and client
remain at parity.

## Why a custom wire format?

One might ask why not use something like a protobuf instead of this custom wire format? The main reason is we would
end up needing to generate the proto file anyways since the volume of requests and responses is way too high.
Ultimately, that would end up with substantially more code and another layer of abstraction.

Additionally, using protobuf makes it challenging, but not impossible, to transfer opaque blocks of memory and cast
them as protobuf has its own opinion on memory layout. One advantage of a custom wire format is we can read from
the RPC directly into the destination, saving us a byte shuffling layer.

Lastly, doing something very custom in protobuf is not nearly as easy. For example, `cudaMemcpy` requires a completely
custom implementation as the wire format is dependent on the `kind` parameter. This is straightforward to do as
we can stub out the implementation with our custom codegen but if we were to tie ourselves to protobuf we would
need an escape hatch.

## Future Improvements

Some improvements that can be made:

- [ ] Currently, the RPC ID is not deterministic. This is fine for now as we are still in demo-phase but this won't work for backwards compatibility.
- [ ] We could use C++ annotations to make the processing a little more "C++"-y. Worth investigating for a bit.
- [ ] Generate `cublas` and other friends.
