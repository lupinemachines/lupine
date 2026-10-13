"""Tensor operations through the converted cuDNN library."""

from contextlib import ExitStack, nullcontext
import ctypes as ct

from .runtime import F, I, P, S


class Cudnn:
    def __init__(self, runtime):
        self.runtime = runtime
        self.functions = {}
        self.handle = P()
        self.call("cudnnCreate", [ct.POINTER(P)], ct.byref(self.handle))

    def call(self, name, types, *arguments):
        if name not in self.functions:
            self.functions[name] = self.runtime.bind(self.runtime.runtime, name, *types)
        status = self.functions[name](*arguments)
        if status:
            error = self.runtime.runtime.cudnnGetErrorString
            error.argtypes, error.restype = [I], ct.c_char_p
            raise RuntimeError(f"{name}: cuDNN {status}: {error(status).decode()}")

    def close(self):
        self.call("cudnnDestroy", [P], self.handle)

    def descriptor(self, resources, kind):
        value = P()
        self.call(f"cudnnCreate{kind}Descriptor", [ct.POINTER(P)], ct.byref(value))
        resources.callback(self.call, f"cudnnDestroy{kind}Descriptor", [P], value)
        return value

    def tensor(self, resources, shape):
        value = self.descriptor(resources, "Tensor")
        # Contiguous NCHW, float32.
        self.call(
            "cudnnSetTensor4dDescriptor", [P, I, I, I, I, I, I], value, 0, 0, *shape
        )
        return value

    def softmax(self, shape, input, output, *, logarithmic=False, grad=None):
        with ExitStack() as resources:
            desc = self.tensor(resources, shape)
            out_desc = self.tensor(resources, shape)
            alpha, beta = F(1), F(0)
            # Accurate/log algorithm, instance mode over contiguous rows.
            common = (
                self.handle,
                2 if logarithmic else 1,
                0,
                ct.byref(alpha),
                desc,
                input,
            )
            if grad is None:
                self.call(
                    "cudnnSoftmaxForward",
                    [P, I, I] + [P] * 6,
                    *common,
                    ct.byref(beta),
                    out_desc,
                    output,
                )
            else:
                self.call(
                    "cudnnSoftmaxBackward",
                    [P, I, I] + [P] * 8,
                    *common,
                    self.tensor(resources, shape),
                    grad,
                    ct.byref(beta),
                    out_desc,
                    output,
                )
            self.runtime.launches += 1

    def execute(self, operation, a_desc, a, b_desc, b, conv, out_desc, out):
        # Forward implicit GEMM; backward algorithm 1 avoids atomic accumulation.
        algorithm = 0 if operation == "Forward" else 1
        size = S()
        self.call(
            f"cudnnGetConvolution{operation}WorkspaceSize",
            [P, P, P, P, P, I, ct.POINTER(S)],
            self.handle,
            a_desc,
            b_desc,
            conv,
            out_desc,
            algorithm,
            ct.byref(size),
        )
        alpha, beta = F(1), F(0)
        allocation = (
            self.runtime.allocated(size.value) if size.value else nullcontext(P())
        )
        with allocation as workspace:
            self.call(
                f"cudnnConvolution{operation}",
                [P, P, P, P, P, P, P, I, P, S, P, P, P],
                self.handle,
                ct.byref(alpha),
                a_desc,
                a,
                b_desc,
                b,
                conv,
                algorithm,
                workspace,
                size,
                ct.byref(beta),
                out_desc,
                out,
            )
            self.runtime.launches += 1

    def convolution(
        self,
        input_shape,
        weight_shape,
        output_shape,
        *,
        stride,
        padding,
        dilation,
        groups,
        input,
        weight,
        output=None,
        bias=None,
        grad=None,
        dx=None,
        dw=None,
        db=None,
    ):
        with ExitStack() as resources:
            x_desc = self.tensor(resources, input_shape)
            y_desc = self.tensor(resources, output_shape)
            w_desc = self.descriptor(resources, "Filter")
            self.call(
                "cudnnSetFilter4dDescriptor",
                [P, I, I, I, I, I, I],
                w_desc,
                0,
                0,
                *weight_shape,
            )
            conv = self.descriptor(resources, "Convolution")
            self.call(
                "cudnnSetConvolution2dDescriptor",
                [P, I, I, I, I, I, I, I, I],
                conv,
                *padding,
                *stride,
                *dilation,
                1,
                0,
            )
            self.call("cudnnSetConvolutionGroupCount", [P, I], conv, groups)
            # FMA math preserves float32 precision instead of permitting TF32.
            self.call("cudnnSetConvolutionMathType", [P, I], conv, 3)
            if output is not None:
                self.execute(
                    "Forward", x_desc, input, w_desc, weight, conv, y_desc, output
                )
            if dx is not None:
                self.execute(
                    "BackwardData", w_desc, weight, y_desc, grad, conv, x_desc, dx
                )
            if dw is not None:
                self.execute(
                    "BackwardFilter", x_desc, input, y_desc, grad, conv, w_desc, dw
                )
            if bias is not None or db is not None:
                b_desc = self.tensor(resources, (1, output_shape[1], 1, 1))
                alpha, beta = F(1), F(0)
                if bias is not None:
                    self.call(
                        "cudnnAddTensor",
                        [P, P, P, P, P, P, P],
                        self.handle,
                        ct.byref(alpha),
                        b_desc,
                        bias,
                        ct.byref(alpha),
                        y_desc,
                        output,
                    )
                if db is not None:
                    self.call(
                        "cudnnConvolutionBackwardBias",
                        [P, P, P, P, P, P, P],
                        self.handle,
                        ct.byref(alpha),
                        y_desc,
                        grad,
                        ct.byref(beta),
                        b_desc,
                        db,
                    )
                self.runtime.launches += 1
