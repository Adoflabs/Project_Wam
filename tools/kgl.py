#!/usr/bin/env python3
"""Kaggle CLI wrapper for this cloud environment.

The Kaggle API token is not stored here: the environment's network proxy adds it
as an `Authorization: Bearer` header on requests to api.kaggle.com. The stock CLI
validates a token by sending it in a request *body* (IntrospectToken), which the
proxy cannot fill, so we skip that one check and take the username from
KAGGLE_USERNAME instead. Every other call is authenticated normally.

Usage: python tools/kgl.py <any kaggle CLI args>   e.g.  python tools/kgl.py quota
"""
import contextlib
import io
import os
import sys

os.environ.setdefault("KAGGLE_API_TOKEN", "proxy-injected")  # placeholder; proxy replaces the header

with contextlib.redirect_stdout(io.StringIO()):  # silence the import-time auth attempt
    import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

KaggleApi._introspect_token = lambda self, token: os.environ.get("KAGGLE_USERNAME", "unknown-user")
kaggle.api.authenticate()

from kaggle.cli import main  # noqa: E402

sys.argv[0] = "kaggle"
sys.exit(main())
