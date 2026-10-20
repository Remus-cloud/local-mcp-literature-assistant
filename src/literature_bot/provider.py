"""Optional provider extensions shared by all Chat Completions call paths."""
from copy import deepcopy

from openai import OpenAI


class ProviderClient(OpenAI):
    def __init__(self, *, request_extra_body=None, **kwargs):
        super().__init__(**kwargs)
        self._literature_extra_body = deepcopy(request_extra_body or {})

    def copy(self, **kwargs):
        client = super().copy(**kwargs)
        client._literature_extra_body = deepcopy(self._literature_extra_body)
        return client

    with_options = copy


def request_options(client) -> dict:
    """Plain SDK clients/mocks default to portable requests, without extensions."""
    body = getattr(client, "_literature_extra_body", {})
    return {"extra_body": deepcopy(body)} if body else {}
