"""Regression coverage for init-time fallbacks that change API wire mode (#150445)."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def test_init_fallback_rebuilds_anthropic_client_and_snapshot():
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = "https://api.anthropic.com"
    anthropic_client = MagicMock(name="anthropic-client")
    resolutions = []

    def resolve(provider, **kwargs):
        resolutions.append((provider, kwargs.get("model")))
        if provider == "openai-codex":
            return None, None
        assert provider == "anthropic"
        return fallback_client, "claude-sonnet-5-5"

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve),
        patch("agent.process_bootstrap.OpenAI", side_effect=AssertionError("wrong wire client")),
        patch(
            "agent.context_compressor.get_model_context_length",
            return_value=200_000,
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=anthropic_client,
        ),
    ):
        agent = AIAgent(
            provider="openai-codex",
            model="gpt-5.3-codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-5-5",
                }
            ],
        )

    assert resolutions[:2] == [
        ("openai-codex", "gpt-5.3-codex"),
        ("anthropic", "claude-sonnet-5-5"),
    ]
    assert agent.provider == "anthropic"
    assert agent.model == "claude-sonnet-5-5"
    assert agent.api_mode == "anthropic_messages"
    assert agent.base_url == "https://api.anthropic.com"
    assert agent.client is None
    assert agent._anthropic_client is anthropic_client
    assert agent._primary_runtime["anthropic_api_key"] == "fallback-key"
    assert agent._primary_runtime["anthropic_base_url"] == "https://api.anthropic.com"
