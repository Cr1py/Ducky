import io
import zipfile

import httpx2
import pytest

from ducky.stt import models
from ducky.stt.models import ModelError, download_model, is_installed, resolve_model

NAME = "vosk-model-small-en-us-0.15"


def make_zip(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for path, data in files.items():
            z.writestr(path, data)
    return buffer.getvalue()


GOOD = make_zip({f"{NAME}/am/final.mdl": b"x" * 100, f"{NAME}/conf/model.conf": b"c"})


def client(handler):
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def serving(payload, status=200):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(status, content=payload)

    return handler, calls


def leftovers(root):
    return [p.name for p in root.iterdir() if p.name != NAME]


def test_download_installs_and_cleans_up(tmp_path):
    handler, calls = serving(GOOD)
    seen = []
    path = download_model(NAME, root=tmp_path, http_client=client(handler),
                          on_progress=lambda got, total: seen.append((got, total)))
    assert path == tmp_path / NAME and is_installed(path)
    assert (path / "am" / "final.mdl").read_bytes() == b"x" * 100
    assert calls[0].url.path.endswith(f"/{NAME}.zip")
    assert seen and seen[-1] == (len(GOOD), len(GOOD)) and seen == sorted(seen)
    assert leftovers(tmp_path) == []          # no .part file, no staging folder


def test_installs_into_ducklys_models_dir_by_default_and_resolves_by_name():
    handler, _ = serving(GOOD)
    path = download_model(NAME, http_client=client(handler))
    assert path == models.models_dir() / NAME
    assert resolve_model(NAME) == path


def test_already_installed_makes_no_request(tmp_path):
    (tmp_path / NAME / "am").mkdir(parents=True)

    def handler(request):
        raise AssertionError("must not hit the network")

    assert download_model(NAME, root=tmp_path, http_client=client(handler)) == tmp_path / NAME


def test_archive_without_a_wrapper_folder_is_accepted(tmp_path):
    handler, _ = serving(make_zip({"am/final.mdl": b"x", "conf/model.conf": b"c"}))
    assert is_installed(download_model(NAME, root=tmp_path, http_client=client(handler)))


def test_unknown_model_name(tmp_path):
    handler, _ = serving(b"", status=404)
    with pytest.raises(ModelError, match="No Vosk model named"):
        download_model("vosk-model-nope", root=tmp_path, http_client=client(handler))
    assert leftovers(tmp_path) == []


def test_server_error(tmp_path):
    handler, _ = serving(b"", status=503)
    with pytest.raises(ModelError, match="HTTP 503"):
        download_model(NAME, root=tmp_path, http_client=client(handler))


def test_no_internet(tmp_path):
    def handler(request):
        raise httpx2.ConnectError("offline")

    with pytest.raises(ModelError, match="internet connection"):
        download_model(NAME, root=tmp_path, http_client=client(handler))
    assert leftovers(tmp_path) == []


def test_corrupt_download(tmp_path):
    handler, _ = serving(b"this is not a zip file")
    with pytest.raises(ModelError, match="corrupt"):
        download_model(NAME, root=tmp_path, http_client=client(handler))
    assert leftovers(tmp_path) == []


def test_archive_that_is_not_a_vosk_model(tmp_path):
    handler, _ = serving(make_zip({f"{NAME}/readme.txt": b"hi"}))
    with pytest.raises(ModelError, match="doesn't look like a Vosk model"):
        download_model(NAME, root=tmp_path, http_client=client(handler))
    assert leftovers(tmp_path) == []


def test_zip_slip_is_refused(tmp_path):
    root = tmp_path / "models"
    handler, _ = serving(make_zip({"../evil.txt": b"gotcha", f"{NAME}/am/x": b"x"}))
    with pytest.raises(ModelError, match="unsafe path"):
        download_model(NAME, root=root, http_client=client(handler))
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize("bad", ["../x", "a/b", "", "-leading", "with space", "C:\\models\\x"])
def test_invalid_names_are_rejected(bad):
    with pytest.raises(ModelError, match="isn't a valid model name"):
        download_model(bad, http_client=client(lambda r: httpx2.Response(200)))


def test_resolve_model_accepts_a_folder_path(tmp_path):
    folder = tmp_path / "my-model"
    (folder / "am").mkdir(parents=True)
    assert resolve_model(str(folder)) == folder


def test_resolve_model_returns_none_when_missing_or_not_a_model(tmp_path):
    assert resolve_model(NAME) is None
    (tmp_path / "empty").mkdir()
    assert resolve_model(str(tmp_path / "empty")) is None
    assert resolve_model(str(tmp_path / "nope")) is None


def test_mirror_env_var_redirects_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv(models.MIRROR_ENV, "http://mirror.internal:8080/vosk/")
    handler, calls = serving(GOOD)
    download_model(NAME, root=tmp_path, http_client=client(handler))
    assert str(calls[0].url) == f"http://mirror.internal:8080/vosk/{NAME}.zip"


def test_explicit_base_url_beats_the_mirror(monkeypatch, tmp_path):
    monkeypatch.setenv(models.MIRROR_ENV, "http://mirror.internal")
    handler, calls = serving(GOOD)
    download_model(NAME, root=tmp_path, http_client=client(handler), base_url="http://explicit.test/m")
    assert calls[0].url.host == "explicit.test"
