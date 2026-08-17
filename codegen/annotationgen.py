# /// script
# requires-python = ">=3.10"
# dependencies = ["cxxheaderparser"]
# ///
import os

from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Array


def main():
    options = ParserOptions(preprocessor=make_gcc_preprocessor(defines=["CUBLASAPI="]))

    nvml_ast: ParsedData = parse_file("/usr/include/nvml.h", options=options)
    cudnn_graph_ast: ParsedData = parse_file("/usr/include/cudnn_graph.h", options=options)
    cudnn_ops_ast: ParsedData = parse_file("/usr/include/cudnn_ops.h", options=options)
    cuda_ast: ParsedData = parse_file("/usr/include/cuda.h", options=options)
    cublas_ast: ParsedData = parse_file("/usr/include/cublas_api.h", options=options)
    cudart_ast: ParsedData = parse_file(
        "/usr/include/cuda_runtime_api.h", options=options
    )
    annotations: ParsedData = parse_file("annotations.h", options=options)

    functions = (
        nvml_ast.namespace.functions
        + cuda_ast.namespace.functions
        + cudart_ast.namespace.functions
        + cudnn_graph_ast.namespace.functions
        + cudnn_ops_ast.namespace.functions
        + cublas_ast.namespace.functions
    )

    with open("annotations.h", "a") as f:
        for function in functions:
            if any(f.name == function.name for f in annotations.namespace.functions):
                continue
            # produce some best-guess annotations
            f.write("/**\n")
            for param in function.parameters:
                if isinstance(param.type, Type):
                    f.write(
                        " * @param {name} SEND_ONLY\n".format(
                            name=param.name, type=param.type.format()
                        )
                    )
                elif isinstance(param.type, Pointer):
                    f.write(
                        " * @param {name} SEND_RECV\n".format(
                            name=param.name, type=param.type.format()
                        )
                    )
                elif isinstance(param.type, Array):
                    f.write(
                        " * @param {name} SEND_ONLY\n".format(
                            name=param.name, type=param.type.format()
                        )
                    )
            f.write(" */\n")

            params = []

            for param in function.parameters:
                if param.name and "[]" in param.type.format():
                    params.append(
                        "{type} {name}".format(
                            type=param.type.format().replace("[]", ""),
                            name=param.name + "[]",
                        )
                    )
                elif param.name:
                    params.append(
                        "{type} {name}".format(
                            type=param.type.format(),
                            name=param.name,
                        )
                    )
                else:
                    params.append(param.type.format())

            joined_params = ", ".join(params)

            f.write(
                "{return_type} {name}({params});\n".format(
                    return_type=function.return_type.format(),
                    name=function.name.format(),
                    params=joined_params,
                )
            )


if __name__ == "__main__":
    # annotations.h is read and appended to CWD-relative.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
