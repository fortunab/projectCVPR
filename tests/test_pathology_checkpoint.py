"""Optional provider-runtime tests for the actual CONCH release's missing decoder."""
import sys
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from extract_pathology_review_features import load_complete_state


@unittest.skipUnless(torch is not None, 'Run in the separate provider environment')
class CheckpointCompleteness(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'tmp/pathology_checkpoint_tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        assert Path(self.temporary.name).resolve().is_relative_to(parent.resolve())
        self.path = Path(self.temporary.name) / 'explicit_fixture.pt'
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Linear(3, 2)
        self.model.text_decoder = torch.nn.Linear(2, 2)
        self.state = {key: value for key, value in self.model.state_dict().items() if key.startswith('visual.')}

    def tearDown(self):
        self.temporary.cleanup()

    def test_only_unused_caption_decoder_may_be_missing(self):
        torch.save(self.state, self.path)
        load_complete_state(self.model, self.path, allowed_missing_prefixes=('text_decoder.',))
        self.assertEqual(self.model._comparison_checkpoint_audit['missing_required_keys'], 0)
        self.assertEqual(self.model._comparison_checkpoint_audit['unused_missing_keys'], 2)

    def test_missing_visual_weight_is_always_rejected(self):
        del self.state['visual.weight']
        torch.save(self.state, self.path)
        with self.assertRaises(RuntimeError):
            load_complete_state(self.model, self.path, allowed_missing_prefixes=('text_decoder.',))

    def test_unexpected_keys_and_wrong_shapes_are_rejected(self):
        self.state['unexpected.weight'] = torch.zeros(1)
        torch.save(self.state, self.path)
        with self.assertRaises(RuntimeError):
            load_complete_state(self.model, self.path, allowed_missing_prefixes=('text_decoder.',))
        del self.state['unexpected.weight']
        self.state['visual.weight'] = torch.zeros(7, 3)
        torch.save(self.state, self.path)
        with self.assertRaises(RuntimeError):
            load_complete_state(self.model, self.path, allowed_missing_prefixes=('text_decoder.',))


if __name__ == '__main__':
    unittest.main()
