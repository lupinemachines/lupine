"""Exercise GCE cleanup races without provisioning cloud resources."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from terraform.gcp.cleanup import cleanup, main


class CleanupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tf_dir = Path(directory.name)
        self.config = self.tf_dir / "run.tfvars.json"
        self.config.write_text(json.dumps({"project_id": "test-project", "run_id": "l4-test"}))
        self.now = 0
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(patch("terraform.gcp.cleanup.time.monotonic", side_effect=lambda: self.now))
        self.sleep = self.enterContext(patch("terraform.gcp.cleanup.time.sleep", side_effect=self.advance))
        self.terraform = Mock(return_value=subprocess.CompletedProcess([], 0))
        self.operations = Mock(return_value=[])
        self.instances = Mock(return_value=[])
        self.firewalls = Mock(return_value=[])
        self.delete = Mock(return_value=[])
        self.run = self.enterContext(patch("terraform.gcp.cleanup.subprocess.run", side_effect=self.command))

    def advance(self, seconds):
        self.now += seconds

    def command(self, args, **kwargs):
        self.assertGreater(kwargs["timeout"], 0)
        if args[0] == "terraform":
            self.assertNotIn("force-unlock", args)
            return self.terraform()
        self.assertEqual(args[:2], ["gcloud", "compute"])
        self.assertIn("--project=test-project", args)
        kind, action = args[2:4]
        if action == "delete":
            result = self.delete(args)
        else:
            self.assertEqual(action, "list")
            result = {"instances": self.instances, "firewall-rules": self.firewalls,
                      "operations": self.operations}[kind]()
        if isinstance(result, subprocess.CompletedProcess):
            return result
        return subprocess.CompletedProcess(args, 0, json.dumps(result), "")

    def resource(self, name="l4-test-a", zone="us-central1-a", kind="instances"):
        scope = f"zones/{zone}" if kind == "instances" else "global"
        return {"name": name, "selfLink":
                f"https://www.googleapis.com/compute/v1/projects/test-project/{scope}/{kind}/{name}"}

    def operation(self, resource, kind="insert", status="RUNNING"):
        return {"name": f"operation-{kind}", "operationType": kind, "status": status,
                "targetLink": resource["selfLink"]}

    def test_successful_terraform_still_deletes_untracked_resources_in_multiple_zones(self):
        a = self.resource()
        b = self.resource("l4-test-b", "us-west1-b")
        firewall = self.resource("l4-test-ssh", kind="firewalls")
        self.instances.side_effect = [[a, b], []]
        self.firewalls.side_effect = [[firewall], []]
        cleanup(self.tf_dir)
        commands = [call.args[0] for call in self.delete.call_args_list]
        self.assertEqual([cmd[4] for cmd in commands], [a["name"], b["name"], firewall["name"]])
        self.assertIn("--zone=us-central1-a", commands[0])
        self.assertIn("--zone=us-west1-b", commands[1])
        self.assertEqual(self.run.call_args_list[0].args[0][0], "terraform")
        self.assertIn("Cleanup verified", self.output.getvalue())

    def test_terraform_failure_or_timeout_does_not_skip_cloud_cleanup(self):
        for failure in (subprocess.CompletedProcess([], 1), subprocess.TimeoutExpired("terraform", 300)):
            with self.subTest(failure=failure):
                self.terraform.side_effect = [failure]
                self.instances.side_effect = [[self.resource()], []]
                self.delete.reset_mock()
                cleanup(self.tf_dir)
                self.delete.assert_called_once()

    def test_pending_failed_create_is_waited_for_even_before_instance_appears(self):
        resource = self.resource()
        pending = self.operation(resource)
        pending["error"] = {"errors": [{"code": "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS"}]}
        self.operations.side_effect = [[pending], [self.operation(resource, status="DONE")], []]
        self.instances.side_effect = [[], [resource], []]
        cleanup(self.tf_dir)
        self.delete.assert_called_once()
        self.assertEqual(self.sleep.call_count, 2)
        self.assertIn("ZONE_RESOURCE_POOL_EXHAUSTED", self.output.getvalue())

    def test_pending_delete_is_waited_for_after_instance_disappears(self):
        resource = self.resource()
        self.operations.side_effect = [[self.operation(resource, "delete", "PENDING")],
                                       [self.operation(resource, "delete")], []]
        self.instances.side_effect = [[resource], [], []]
        cleanup(self.tf_dir)
        self.delete.assert_not_called()
        self.assertEqual(self.sleep.call_count, 2)

    def test_delete_command_timeout_polls_operation_without_resubmitting(self):
        resource = self.resource()
        self.operations.side_effect = [[], [self.operation(resource, "delete")], []]
        self.instances.side_effect = [[resource], [resource], []]
        self.delete.side_effect = subprocess.TimeoutExpired("gcloud", 30)
        cleanup(self.tf_dir)
        self.delete.assert_called_once()
        self.assertIn("Delete still pending", self.output.getvalue())

    def test_unresolved_operation_reaches_deadline_and_fails_cli(self):
        self.operations.return_value = [self.operation(self.resource())]
        errors = io.StringIO()
        with patch("sys.argv", ["cleanup.py", str(self.tf_dir), "--timeout", "20"]), redirect_stderr(errors):
            self.assertEqual(main(), 1)
        self.assertEqual(self.now, 20)
        self.assertIn("deadline exceeded", errors.getvalue())
        self.assertIn("operation-insert", self.output.getvalue())
        self.assertNotIn("Cleanup verified", self.output.getvalue())

    def test_cloud_errors_are_not_treated_as_empty_inventory(self):
        self.instances.return_value = subprocess.CompletedProcess([], 1, "", "Permission denied")
        with self.assertRaisesRegex(RuntimeError, "Permission denied"):
            cleanup(self.tf_dir)
        self.delete.assert_not_called()
        self.assertNotIn("Cleanup verified", self.output.getvalue())

    def test_partial_cloud_list_failure_is_not_treated_as_empty_inventory(self):
        self.operations.return_value = subprocess.CompletedProcess(
            [], 0, "[]", "WARNING: Some requests did not succeed. Permission denied")
        with self.assertRaisesRegex(RuntimeError, "Some requests did not succeed"):
            cleanup(self.tf_dir)
        self.delete.assert_not_called()
        self.assertNotIn("Cleanup verified", self.output.getvalue())

    def test_other_runs_resource_types_and_projects_are_ignored(self):
        self.instances.return_value = [self.resource("l4-test-a-extra"), self.resource("l4-other-a")]
        self.firewalls.return_value = [self.resource("l4-test-ssh-extra", kind="firewalls")]
        other_project = self.operation(self.resource())
        other_project["targetLink"] = other_project["targetLink"].replace("/test-project/", "/other-project/")
        self.operations.return_value = [other_project, self.operation(self.resource(kind="disks")),
                                       self.operation(self.resource("l4-test-a-extra"))]
        cleanup(self.tf_dir)
        self.delete.assert_not_called()
        self.sleep.assert_not_called()

    def test_invalid_run_id_cannot_broaden_deletion_scope(self):
        self.config.write_text(json.dumps({"project_id": "test-project", "run_id": "l4-.*"}))
        with self.assertRaises(ValueError):
            cleanup(self.tf_dir)
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
