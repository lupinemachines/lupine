import textwrap
from pathlib import Path

from ops import ClientCallTemplate


GENERATED_CALL_MARKER = "LUPINE_GENERATED_CALL"


def _template_section(source: str) -> str:
    stripped = source.strip("\r\n")
    if not stripped.strip():
        return ""
    return textwrap.dedent(stripped) + "\n"


def _definition_body(source: str, name: str, return_type: str) -> str:
    signature = f"{return_type} {name}("
    if source.count(signature) != 1:
        raise RuntimeError(f"{name}: could not locate definition in annotations.h")

    signature_start = source.index(signature)
    body_start = source.index("{", signature_start) + 1
    body_end = source.index("\n}", body_start)
    return source[body_start:body_end]


def collect_client_call_templates(
    path: str, definition_return_types: dict[str, str]
) -> dict[str, ClientCallTemplate]:
    source = Path(path).read_text(encoding="utf-8")
    templates = {}
    for name, return_type in definition_return_types.items():
        body = _definition_body(source, name, return_type)
        marker = f"  {return_type} return_value = {GENERATED_CALL_MARKER}();"
        if body.count(marker) != 1:
            raise RuntimeError(
                f"{name}: expected exactly one `{marker.strip()}`"
            )
        before_call, after_call = body.split(marker)

        final_return = "  return return_value;"
        if after_call.count(final_return) != 1:
            raise RuntimeError(
                f"{name}: client call template must end with `{final_return.strip()}`"
            )
        after_call, trailing = after_call.split(final_return)
        if trailing.strip():
            raise RuntimeError(
                f"{name}: `{final_return.strip()}` must be the final statement"
            )

        templates[name] = ClientCallTemplate(
            return_type=return_type,
            before_call=_template_section(before_call),
            after_call=_template_section(after_call),
        )
    return templates
