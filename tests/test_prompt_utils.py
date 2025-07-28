"""
Tests for prompt rendering utilities.
"""
import json
from pathlib import Path

import pytest

from utils.prompt_utils import render_jinja_template


def test_render_jinja_template(tmp_path):
    template = tmp_path / 'tmpl.jinja'
    template.write_text("Hello {{ name }}!\n")
    out = render_jinja_template(template, {'name': 'Test'})
    assert out.strip() == "Hello Test!"
