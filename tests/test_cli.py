import contextlib
import io
import unittest

from jiuge_training.cli import main


class CliTests(unittest.TestCase):
    def test_dry_run_and_bad_override(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['dry-run']), 0)
        self.assertIn('torchrun', output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        'validate',
                        '--set',
                        'task.lora_rank=0',
                        '--task',
                        'lora-smoke',
                    ]
                ),
                2,
            )
            self.assertEqual(main(['validate', '--set', 'task.typo=1']), 2)


if __name__ == '__main__':
    unittest.main()
