import pytest

from ducky import registry
from ducky.registry import RegistryError


def write_user_models(ducky_home, text):
    path = ducky_home / "config" / "models.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_builtin_registry_is_valid():
    models = registry.load_registry()
    assert set(models) == {"claude", "chatgpt", "gemini", "ollama"}
    assert models["claude"].provider == "anthropic"
    assert models["gemini"].provider == "gemini"
    assert models["chatgpt"].token_param == "max_completion_tokens"
    ollama = models["ollama"]
    assert (
        ollama.provider == "ollama"
        and ollama.api_key_env is None
        and ollama.num_ctx == 8192
    )


def test_user_file_overrides_single_field(ducky_home):
    write_user_models(ducky_home, '[models.claude]\nmodel = "claude-sonnet-5-5"\n')
    spec = registry.load_registry()["claude"]
    assert spec.model == "claude-sonnet-5-5"
    assert spec.api_key_env == "ANTHROPIC_API_KEY"  # untouched fields survive the merge


def test_user_file_adds_keyless_local_model(ducky_home):
    write_user_models(
        ducky_home,
        '[models.local]\nprovider = "openai_compat"\nbase_url = "http://localhost:11434/v1"\nmodel = "llama3.1"\n',
    )
    spec = registry.get_spec("local")
    assert spec.api_key_env is None and spec.base_url.startswith("http://localhost")


@pytest.mark.parametrize(
    "body,fragment",
    [
        ('[models.x]\nprovider = "openai_compat"\nmodel = "m"\n', "base_url"),
        ('[models.x]\nprovider = "anthropic"\nmodel = "m"\n', "api_key_env"),
        ('[models.x]\nprovider = "carrier-pigeon"\nmodel = "m"\n', "provider"),
        ("[models.claude]\nmaxtokens = 5\n", "maxtokens"),  # typo -> loud failure
        ("[models.claude]\nmax_tokens = 0\n", "max_tokens"),
        ('[models.claude]\ntier = "hard"\n', "tier"),  # field from the old sample
        (
            '[models.MyModel]\nprovider = "gemini"\nmodel = "m"\napi_key_env = "K"\n',
            "Invalid model name",
        ),
        ("[models.x\n", "invalid TOML"),
        ("models = 3\n", "expected [models"),
    ],
)
def test_invalid_user_entries(ducky_home, body, fragment):
    path = write_user_models(ducky_home, body)
    with pytest.raises(RegistryError) as exc:
        registry.load_registry()
    assert fragment in str(exc.value)
    assert (
        str(path) in str(exc.value)
        or "invalid TOML" in str(exc.value)
        or "expected" in str(exc.value)
    )


def test_get_spec_case_insensitive_and_unknown():
    assert registry.get_spec("CLAUDE").provider == "anthropic"
    with pytest.raises(RegistryError, match="Options: claude"):
        registry.get_spec("llama")


def test_ollama_entry_needs_neither_key_nor_base_url(ducky_home):
    write_user_models(
        ducky_home, '[models.mine]\nprovider = "ollama"\nmodel = "llama3.2"\n'
    )
    spec = registry.get_spec("mine")
    assert spec.api_key_env is None and spec.base_url is None and spec.num_ctx is None


def test_ollama_fields_accept_expected_types(ducky_home):
    write_user_models(
        ducky_home,
        '[models.a]\nprovider = "ollama"\nmodel = "m"\nthink = false\nkeep_alive = "30m"\nnum_ctx = 4096\n'
        '[models.b]\nprovider = "ollama"\nmodel = "m"\nthink = "low"\nkeep_alive = 0\n',
    )
    a, b = registry.get_spec("a"), registry.get_spec("b")
    assert (a.think, a.keep_alive, a.num_ctx) == (False, "30m", 4096)
    assert (b.think, b.keep_alive) == ("low", 0)


@pytest.mark.parametrize(
    "field", ["num_ctx = 4096", 'keep_alive = "5m"', "think = false"]
)
def test_ollama_only_fields_rejected_elsewhere(ducky_home, field):
    write_user_models(ducky_home, f"[models.claude]\n{field}\n")
    with pytest.raises(RegistryError, match="only apply to provider"):
        registry.load_registry()


def test_ollama_num_ctx_must_be_positive(ducky_home):
    write_user_models(
        ducky_home, '[models.x]\nprovider = "ollama"\nmodel = "m"\nnum_ctx = 0\n'
    )
    with pytest.raises(RegistryError, match="num_ctx"):
        registry.load_registry()
