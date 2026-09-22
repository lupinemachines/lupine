# /// script
# requires-python = ">=3.10"
# dependencies = ["cxxheaderparser"]
# ///
import argparse
import os

from cxxheaderparser.simple import parse_file, ParsedData, ParserOptions
from cxxheaderparser.preprocessor import make_gcc_preprocessor
from cxxheaderparser.types import Type, Pointer, Array

from codegen import ANNOTATION_FILES, find_header_file

# The SDK header whose declarations seed each target's annotation file.
TARGET_HEADERS = {
    "cuda": "cuda.h",
    "nvml": "nvml.h",
    "hip": "hip_runtime_api.h",
}


def main(target: str):
    header = find_header_file(TARGET_HEADERS[target])
    annotations_path = ANNOTATION_FILES[target]
    options = ParserOptions(
        preprocessor=make_gcc_preprocessor(
            defines=["__HIP_PLATFORM_AMD__"],
            # HIP declarations are included as <hip/...>, so search the
            # header's parent directory as well as its own.
            include_paths=[
                os.path.dirname(header),
                os.path.dirname(os.path.dirname(header)),
            ],
        )
    )

    header_ast: ParsedData = parse_file(header, options=options)
    annotations: ParsedData = parse_file(annotations_path, options=options)
    annotated = {f.name.format() for f in annotations.namespace.functions}

    with open(annotations_path, "a") as f:
        for function in header_ast.namespace.functions:
            if function.name.format() in annotated:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=sorted(TARGET_HEADERS))
    args = parser.parse_args()
    # Annotation files are read and appended to CWD-relative.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args.target)
