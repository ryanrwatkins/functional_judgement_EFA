"""
Utility functions for rendering and handling prompt templates.
"""


def render_prompt(template: str, context: dict) -> str:
    """
    Render a simple template string with Python str.format as fallback.

    :param template: Template string with format placeholders.
    :param context: Mapping for placeholders.
    :return: Rendered string.
    """
    try:
        return template.format(**context)
    except Exception:
        return template


def render_jinja_template(template_path: Path, context: dict) -> str:
    """
    Render a Jinja2 template file with the given context.

    :param template_path: Path to the .jinja template file.
    :param context: Mapping for template rendering.
    :return: Rendered template string.
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        autoescape=select_autoescape(['jinja']),
        keep_trailing_newline=True,
    )
    template = env.get_template(template_path.name)
    return template.render(**context)
