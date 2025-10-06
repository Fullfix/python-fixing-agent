import abc
import logging
from dataclasses import dataclass
import tempfile
import os
import subprocess
import sys


@dataclass(frozen=True)
class ScriptExecutionResult:
    stdout: str
    stderr: str
    exit_code: int

    @staticmethod
    def from_timeout(stdout: str, stderr: str):
        return ScriptExecutionResult(
            stdout=stdout,
            stderr=stderr + '\nTIMEOUT',
            exit_code=124
        )


class ScriptExecutorInterface(abc.ABC):
    @abc.abstractmethod
    def execute(self, code: str) -> ScriptExecutionResult:
        pass


class LocalScriptExecutor(ScriptExecutorInterface):
    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str) -> ScriptExecutionResult:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                script_path = os.path.join(temp_dir, "run.py")
                os.makedirs(os.path.dirname(script_path), exist_ok=True)
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(code)

                cmd = [sys.executable, script_path]

                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )

                return ScriptExecutionResult(
                    stdout=process.stdout,
                    stderr=process.stderr,
                    exit_code=process.returncode,
                )
        except subprocess.TimeoutExpired as e:
            return ScriptExecutionResult.from_timeout(
                stdout=str(e.stdout),
                stderr=str(e.stderr)
            )


class DockerScriptExecutor(ScriptExecutorInterface):
    def __init__(self, image: str = "python:3.11-slim", timeout_seconds: float = 6.0):
        self.image = image
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str) -> ScriptExecutionResult:
        inspect_cmd = ["docker", "image", "inspect", self.image]
        image_exists = subprocess.run(inspect_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0

        if not image_exists:
            logging.debug(f"Docker image {self.image} not found. Pulling")
            pull_cmd = ["docker", "pull", self.image]
            try:
                subprocess.run(pull_cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logging.warning("Failed to pull docker image")
                return ScriptExecutionResult(
                    stdout=e.stdout,
                    stderr=f"Failed to pull image {self.image}\n{e.stderr}",
                    exit_code=e.returncode
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = os.path.join(temp_dir, "run.py")
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = [
                "docker", "run", "--rm",
                "-v", f"{temp_dir}:/work:ro",
                self.image,
                "python", "/work/run.py"
            ]
            try:
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                return ScriptExecutionResult(
                    stdout=process.stdout,
                    stderr=process.stderr,
                    exit_code=process.returncode
                )
            except subprocess.TimeoutExpired as e:
                return ScriptExecutionResult.from_timeout(
                    stdout=str(e.stdout),
                    stderr=str(e.stderr)
                )
