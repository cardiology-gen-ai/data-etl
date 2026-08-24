import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main_graph


class MainGraphLoggingTests(unittest.TestCase):
    def setUp(self):
        self.root_logger = logging.getLogger()
        self.original_handlers = list(self.root_logger.handlers)

    def tearDown(self):
        for handler in list(self.root_logger.handlers):
            if handler not in self.original_handlers:
                self.root_logger.removeHandler(handler)
                handler.close()
        main_graph._CURRENT_RUN_LOG_CONTEXT = None

    def test_configure_run_logging_uses_work_root_logs_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            work_root = project_root / "first_prototype"
            work_root.mkdir(parents=True, exist_ok=True)

            with patch.dict("os.environ", {}, clear=False):
                log_path = main_graph.configure_run_logging(
                    work_root=work_root,
                    phase="entities",
                    project_root=project_root,
                    log_to_file=True,
                )
                repeated_path = main_graph.configure_run_logging(
                    work_root=work_root,
                    phase="entities",
                    project_root=project_root,
                    log_to_file=True,
                )

            self.assertIsNotNone(log_path)
            self.assertEqual(log_path, repeated_path)
            self.assertEqual(log_path.parent, (work_root / "logs").resolve())
            self.assertTrue(log_path.exists())
            self.assertRegex(log_path.name, r"^entities_\d{8}_\d{6}\.log$")

        run_handlers = [
            handler
            for handler in self.root_logger.handlers
            if isinstance(handler, logging.FileHandler)
            and getattr(handler, "name", "") == main_graph._RUN_LOG_HANDLER_NAME
        ]
        self.assertEqual(len(run_handlers), 1)

    def test_configure_run_logging_uses_custom_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            work_root = project_root / "first_prototype"
            custom_log_dir = project_root / "custom_logs"
            work_root.mkdir(parents=True, exist_ok=True)

            with patch.dict("os.environ", {}, clear=False):
                log_path = main_graph.configure_run_logging(
                    work_root=work_root,
                    phase="normalization",
                    project_root=project_root,
                    log_to_file=True,
                    configured_log_dir=custom_log_dir,
                )

            self.assertIsNotNone(log_path)
            self.assertEqual(log_path.parent, custom_log_dir.resolve())
            self.assertTrue(log_path.exists())
            self.assertTrue(re.match(r"^normalization_\d{8}_\d{6}\.log$", log_path.name))


if __name__ == "__main__":
    unittest.main()
