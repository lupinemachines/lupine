import textwrap
from pathlib import Path

from cxxheaderparser.lexer import LexerTokenStream

from ops import ClientCallTemplate


GENERATED_CALL_MARKER = "LUPINE_GENERATED_CALL"


def _mask_preprocessor_directives(source: str) -> str:
    masked_lines = []
    continued = False
    for line in source.splitlines(keepends=True):
        is_directive = continued or line.lstrip().startswith("#")
        continued = is_directive and line.rstrip("\r\n").endswith("\\")
        if is_directive:
            masked_lines.append(
                "".join(
                    character if character in "\r\n" else " " for character in line
                )
            )
        else:
            masked_lines.append(line)
    return "".join(masked_lines)


def _lex(source: str) -> list:
    stream = LexerTokenStream(None, _mask_preprocessor_directives(source))
    tokens = []
    try:
        while True:
            tokens.append(stream.token())
    except EOFError:
        return tokens


def _token_values(source: str) -> list[str]:
    return [token.value for token in _lex(source)]


def _matching_open(tokens, close_index: int, opening: str, closing: str) -> int:
    depth = 1
    for index in range(close_index - 1, -1, -1):
        value = tokens[index].value
        if value == closing:
            depth += 1
        elif value == opening:
            depth -= 1
            if depth == 0:
                return index
    raise RuntimeError(f"unmatched {closing} in client call template")


def _function_name(tokens, body_open: int) -> str:
    parameters_close = body_open - 1
    if parameters_close < 0 or tokens[parameters_close].value != ")":
        raise RuntimeError("client call template must be a plain function definition")
    parameters_open = _matching_open(tokens, parameters_close, "(", ")")
    name_index = parameters_open - 1
    if name_index < 0 or tokens[name_index].type != "NAME":
        raise RuntimeError("could not identify client call template function")
    return tokens[name_index].value


def _line_start(source: str, position: int) -> int:
    return source.rfind("\n", 0, position) + 1


def _template_section(source: str) -> str:
    stripped = source.strip("\r\n")
    if not stripped.strip():
        return ""
    return textwrap.dedent(stripped) + "\n"


def collect_client_call_templates(
    path: str, definition_return_types: dict[str, str]
) -> dict[str, ClientCallTemplate]:
    source = Path(path).read_text(encoding="utf-8")
    tokens = _lex(source)

    brace_stack = []
    brace_pairs = {}
    marker_bodies = {}
    for index, token in enumerate(tokens):
        if token.value == "{":
            brace_stack.append(index)
        elif token.value == "}":
            if not brace_stack:
                raise RuntimeError("unmatched } while reading client call templates")
            brace_pairs[brace_stack.pop()] = index
        elif token.value == GENERATED_CALL_MARKER:
            if not brace_stack:
                raise RuntimeError(f"{GENERATED_CALL_MARKER} must appear in a function")
            body_open = brace_stack[-1]
            if body_open in marker_bodies:
                raise RuntimeError("client call template must contain exactly one marker")
            marker_bodies[body_open] = index
    if brace_stack:
        raise RuntimeError("unmatched { while reading client call templates")

    templates = {}
    for body_open, marker_index in marker_bodies.items():
        body_close = brace_pairs[body_open]
        name = _function_name(tokens, body_open)
        if name in templates:
            raise RuntimeError(f"duplicate client call template for {name}")
        expected_return_type = definition_return_types.get(name)
        if expected_return_type is None:
            raise RuntimeError(f"{name}: client call template has no parsed definition")

        expected_marker = [
            "return_value",
            "=",
            GENERATED_CALL_MARKER,
            "(",
            ")",
            ";",
        ]
        marker_start = marker_index - 2
        marker_end = marker_index + 4
        if (
            marker_start <= body_open
            or marker_end > body_close
            or [token.value for token in tokens[marker_start:marker_end]]
            != expected_marker
        ):
            raise RuntimeError(
                f"{name}: expected `return_value = {GENERATED_CALL_MARKER}();`"
            )

        declaration_start = marker_start
        while (
            declaration_start > body_open + 1
            and tokens[declaration_start - 1].value not in {";", "{", "}"}
        ):
            declaration_start -= 1
        declared_return_type = [
            token.value for token in tokens[declaration_start:marker_start]
        ]
        if declared_return_type != _token_values(expected_return_type):
            raise RuntimeError(
                f"{name}: generated call result must be declared as "
                f"`{expected_return_type} return_value`"
            )

        final_return_start = body_close - 3
        if [
            token.value for token in tokens[final_return_start:body_close]
        ] != ["return", "return_value", ";"]:
            raise RuntimeError(
                f"{name}: client call template must end with `return return_value;`"
            )

        body_start_position = tokens[body_open].lexpos + 1
        declaration_position = _line_start(source, tokens[declaration_start].lexpos)
        if source[declaration_position : tokens[declaration_start].lexpos].strip():
            raise RuntimeError(
                f"{name}: generated call declaration must start on its own line"
            )
        marker_end_position = tokens[marker_end - 1].lexpos + len(
            tokens[marker_end - 1].value
        )
        final_return_position = _line_start(source, tokens[final_return_start].lexpos)
        if source[final_return_position : tokens[final_return_start].lexpos].strip():
            raise RuntimeError(
                f"{name}: final return must start on its own line"
            )
        templates[name] = ClientCallTemplate(
            return_type=expected_return_type,
            before_call=_template_section(
                source[body_start_position:declaration_position]
            ),
            after_call=_template_section(
                source[marker_end_position:final_return_position]
            ),
        )

    missing_markers = set(definition_return_types) - set(templates)
    if missing_markers:
        raise RuntimeError(
            "function definitions in annotations.h require "
            f"{GENERATED_CALL_MARKER}: " + ", ".join(sorted(missing_markers))
        )
    return templates
