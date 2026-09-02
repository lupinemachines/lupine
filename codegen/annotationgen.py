# /// script
# requires-python = ">=3.10"
# dependencies = ["cxxheaderparser"]
# ///
import argparse
import glob
import os

from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.simple import ParsedData, ParserOptions, parse_file
from cxxheaderparser.types import Array, Pointer, Type


PARTITIONS = {
    "cuda": ("cuda.h", "annotations_cuda.h"),
    "nvml": ("nvml.h", "annotations_nvml.h"),
    "cublas": ("cublas_api.h", "annotations_cublas.h"),
}

INCLUDE_DIRS = (
    "/usr/local/cuda/include",
    "/opt/cuda/include",
    "/usr/local/include",
    "/usr/include",
    "/usr/include/nvidia",
)


def find_header(filename: str) -> str:
    for include_dir in INCLUDE_DIRS:
        matches = glob.glob(
            os.path.join(include_dir, "**", filename), recursive=True
        )
        if matches:
            return matches[0]
    raise FileNotFoundError(filename)


def format_parameters(function) -> str:
    parameters = []
    for parameter in function.parameters:
        formatted_type = parameter.type.format()
        if parameter.name and "[]" in formatted_type:
            parameters.append(
                f"{formatted_type.replace('[]', '')} {parameter.name}[]"
            )
        elif parameter.name:
            parameters.append(f"{formatted_type} {parameter.name}")
        else:
            parameters.append(formatted_type)
    return ", ".join(parameters)


def append_missing_functions(partition: str) -> None:
    header_name, annotations_name = PARTITIONS[partition]
    header_path = find_header(header_name)
    include_paths = list(INCLUDE_DIRS) + [os.path.dirname(header_path)]
    options = ParserOptions(
        preprocessor=make_gcc_preprocessor(
            defines=["CUBLASAPI="], include_paths=include_paths
        )
    )
    header: ParsedData = parse_file(header_path, options=options)
    annotations: ParsedData = parse_file(annotations_name, options=options)
    annotated_names = {
        function.name.format() for function in annotations.namespace.functions
    }

    with open(annotations_name, "a", encoding="utf-8") as output:
        for function in header.namespace.functions:
            name = function.name.format()
            if name in annotated_names:
                continue
            annotated_names.add(name)
            output.write("/**\n")
            for parameter in function.parameters:
                if isinstance(parameter.type, Type):
                    direction = "SEND_ONLY"
                elif isinstance(parameter.type, Pointer):
                    direction = "SEND_RECV"
                elif isinstance(parameter.type, Array):
                    direction = "SEND_ONLY"
                else:
                    continue
                output.write(f" * @param {parameter.name} {direction}\n")
            output.write(" */\n")
            output.write(
                f"{function.return_type.format()} {name}"
                f"({format_parameters(function)});\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append missing declarations to one annotation partition"
    )
    parser.add_argument("partition", choices=PARTITIONS)
    args = parser.parse_args()
    append_missing_functions(args.partition)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
