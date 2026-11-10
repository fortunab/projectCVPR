"""Access, checkpoint identity and relocation checks; no pretrained weights."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_pathology_checkpoints import fetch
from prepare_pathology_comparison import image_location, prepare, runtime_verification_metadata
from compare_pathology_encoders import validate_comparison
from review_protocol import sha256


def registry(tmp_path, data=b"explicit-test-fixture-not-model-weights"):
    path = tmp_path / "registry.json"
    model = {"repository": "fixture/model", "filename": "pytorch_model.bin", "revision": "1234",
             "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path.write_text(json.dumps({"models": {"uni": model}}))
    return path, data


def test_runtime_check_is_explicit_and_rejects_a_stale_extractor(tmp_path):
    assert runtime_verification_metadata(None) == {}
    path = tmp_path / 'runtime.json'
    extractor = Path(__file__).resolve().parents[1] / 'scripts/extract_pathology_review_features.py'
    path.write_text(json.dumps({'extractor_sha256': sha256(extractor)}))
    assert runtime_verification_metadata(path)['untrained_runtime_verification_sha256'] == sha256(path)
    path.write_text(json.dumps({'extractor_sha256': 'obsolete-extractor'}))
    with pytest.raises(ValueError, match='another extractor revision'):
        runtime_verification_metadata(path)


def test_invalid_runtime_fails_before_creating_a_partial_plan(tmp_path):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps({'extractor_sha256': 'obsolete-extractor'}))
    output = tmp_path / 'new-plan'
    with pytest.raises(ValueError, match='another extractor revision'):
        prepare(output, tmp_path / 'registry.json', tmp_path / 'weights', path)
    assert not output.exists()


def test_download_requires_local_login_before_contacting_provider(tmp_path):
    path, _ = registry(tmp_path)
    def forbidden(**kwargs):
        pytest.fail("The downloader must not be called before local authentication")
    with pytest.raises(RuntimeError, match="Local Hugging Face login is missing"):
        fetch(path, tmp_path / "weights", ["uni"], downloader=forbidden, token_reader=lambda: None)
    assert not (tmp_path / "weights").exists()


def test_download_pins_revision_and_keeps_credentials_out_of_records(tmp_path):
    path, content = registry(tmp_path)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["local_dir"]) / kwargs["filename"]
        output.write_bytes(content)
        return str(output)
    result = fetch(path, tmp_path / "weights", ["uni"], downloader=download, token_reader=lambda: "fixture-secret")
    assert calls[0]["revision"] == "1234"
    assert "fixture-secret" not in json.dumps(result)
    assert "fixture-secret" not in (tmp_path / "weights/verified_checkpoints.json").read_text()


def test_offline_verified_weights_are_reused_and_corruption_is_not_overwritten(tmp_path):
    path, content = registry(tmp_path)
    checkpoint = tmp_path / "weights/uni/pytorch_model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(content)
    def forbidden(*args, **kwargs):
        pytest.fail("No authentication or download is needed for an already verified checkpoint")
    fetch(path, tmp_path / "weights", ["uni"], downloader=forbidden, token_reader=forbidden)
    changed = bytes([content[0] ^ 1]) + content[1:]
    checkpoint.write_bytes(changed)
    with pytest.raises(ValueError, match="size/SHA256"):
        fetch(path, tmp_path / "weights", ["uni"], downloader=forbidden, token_reader=forbidden)
    assert checkpoint.read_bytes() == changed


def test_download_error_does_not_expose_request_or_token(tmp_path):
    path, _ = registry(tmp_path)
    def failed(**kwargs):
        raise RuntimeError("Authorization: fixture-secret")
    with pytest.raises(RuntimeError) as error:
        fetch(path, tmp_path / "weights", ["uni"], downloader=failed, token_reader=lambda: "fixture-secret")
    assert "fixture-secret" not in str(error.value)
    assert error.value.__suppress_context__


def test_relocated_images_must_preserve_the_actual_file_hash(tmp_path):
    marker = "isolated_datasets/polypgen_extensions_2026-10-20/exp5_camelyon/images/"
    image = tmp_path / marker / "patient_001/patch.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"raw-image-fixture")
    row = {"sample_id": "s1", "image_path": "Z:/old-host/" + marker + "patient_001/patch.png",
           "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    assert image_location(row, tmp_path) == image.resolve()
    image.write_bytes(b"altered")
    with pytest.raises(ValueError, match="altered"):
        image_location(row, tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        image_location(dict(row, image_path="Z:/old/" + marker + "../outside.png"), tmp_path)


@pytest.mark.parametrize('change', ['none', 'target_support', 'checkpoint', 'grid'])
def test_encoder_comparison_requires_same_support_and_pinned_inputs(tmp_path, change):
    plan = {'seeds': list(range(10)), 'protocols': ['matched4'],
            'original_manifest_sha256': 'original', 'portable_manifest_sha256': 'portable',
            'reference': {'feature_sha256': 'reference-features'},
            'models': {name: {'sha256': name+'-weights', 'revision': name+'-revision'} for name in ('uni', 'conch')}}
    ref_run, new_run, ref_report, new_report = [tmp_path / name for name in ('ref-run', 'new-run', 'ref-report', 'new-report')]
    for folder in (ref_run, new_run, ref_report, new_report):
        folder.mkdir()
    reference = {'tasks': ['camelyon_resnet18'], 'seeds': plan['seeds'], 'protocols': plan['protocols'],
                 'candidate_grids': {'matched4': 'same-prespecified-grid'},
                 'inputs': {'camelyon_resnet18': {'manifest_sha256': 'original', 'feature_sha256': 'reference-features'}}}
    updated = {**reference, 'tasks': ['camelyon_custom_uni', 'camelyon_custom_conch'], 'inputs': {
        'camelyon_custom_'+name: {'manifest_sha256': 'portable', 'feature_metadata': {
            'checkpoint_sha256': name+'-weights', 'model_revision': name+'-revision'}} for name in ('uni', 'conch')}}
    if change == 'checkpoint':
        updated['inputs']['camelyon_custom_conch']['feature_metadata']['checkpoint_sha256'] = 'different-weights'
    if change == 'grid':
        updated['candidate_grids'] = {'matched4': 'different-budget'}
    def write(path, data):
        path.write_text(json.dumps(data))
    for folder, data in ((ref_run, reference), (new_run, updated)):
        write(folder / 'run_manifest.json', data)
        write(folder / 'COMPLETE.json', {'tasks': data['tasks'], 'seeds': data['seeds']})
    for report, run, tasks in ((ref_report, ref_run, reference['tasks']), (new_report, new_run, updated['tasks'])):
        supports = {f'{task}/{center}': 'same-target-samples-labels-and-patients' for task in tasks for center in ('1', '2')}
        if report == new_report and change == 'target_support':
            supports['camelyon_custom_uni/2'] = 'unpaired-target'
        write(report / 'provenance.json', {'run_manifest_sha256': sha256(run / 'run_manifest.json'), 'target_support_sha256': supports})
    if change == 'none':
        validate_comparison(plan, ref_run, ref_report, new_run, new_report)
    else:
        with pytest.raises(ValueError):
            validate_comparison(plan, ref_run, ref_report, new_run, new_report)
