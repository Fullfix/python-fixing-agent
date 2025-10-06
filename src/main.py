import argparse
import logging
from tqdm import tqdm
import os
from datasets import load_dataset
from agent import HumanEvalFixAgent
from adapter import HuggingFacePipelineChatAdapter
from script_executor import ScriptExecutorInterface, LocalScriptExecutor, DockerScriptExecutor
from utils import create_test_script, validate_test_script


def evaluate_sample(agent: HumanEvalFixAgent, script_executor: ScriptExecutorInterface, sample: dict) -> bool:
    logging.debug(f"Running agent on sample: {sample['signature']}")
    function_definition = agent.run(
        entry_point=sample["entry_point"],
        declaration=sample["declaration"],
        docstring=sample["docstring"],
        instruction=sample["instruction"],
        buggy_solution=sample["buggy_solution"],
        test=sample["test"]
    )
    logging.debug(f"Finished running agent")
    if function_definition is None:
        return False

    script = create_test_script(
        declaration=sample["declaration"],
        definition=function_definition,
        test_script=sample["test"]
    )
    logging.debug(f"Testing")
    execution_result = script_executor.execute(script)
    passed = validate_test_script(
        exit_code=execution_result.exit_code,
        stdout=execution_result.stdout
    )
    if passed:
        logging.debug("Tests passed")
    else:
        logging.debug("Tests failed")
    return passed


def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import torch
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to evaluate LangGraph agent for fixing buggy Python code on HumanEvalFix benchmark."
    )

    parser.add_argument(
        dest="model",
        help="Hugging Face model name (e.g., 'Qwen2.5-Coder-0.5B-Instruct')."
    )

    parser.add_argument(
        "--i", "--iterations",
        dest="iterations",
        type=int,
        default=3,
        help="Maximum number of reasoning iterations."
    )

    parser.add_argument(
        "--l", "--local",
        dest="local",
        action="store_true",
        help="Run locally instead of in docker (use with caution, as the executed code is LLM-generated!). By default, runs in docker container."
    )

    parser.add_argument(
        '--t', '--timeout',
        dest='timeout',
        type=float,
        default=6.0,
        help='Testing script timeout in seconds. By default, 6 seconds.'
    )

    parser.add_argument(
        '--s', '--seed',
        dest='seed',
        type=int,
        default=69,
        help='Seed for reproducibility.'
    )

    parser.add_argument(
        '--ts', '--test_samples',
        dest='test_samples',
        type=int,
        default=None,
        help='Number of test samples to evaluate. If not specified, evaluate all samples.'
    )

    parser.add_argument(
        "-v", "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose debug logging."
    )

    args = parser.parse_args()

    model_name: str = args.model
    local: bool = args.local
    max_iterations: int = args.iterations
    timeout_seconds: float = args.timeout
    seed: int = args.seed
    verbose: bool = args.verbose
    test_samples: int | None = args.test_samples

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    logging.info(f"Testing {model_name} with {max_iterations} reasoning iterations {'locally' if local else 'docker'}.")
    logging.info(f"Seed: {seed}, Timeout: {timeout_seconds:.3f} seconds.")

    logging.debug("Loading dataset HumanEvalFix")
    ds = load_dataset("bigcode/humanevalpack", "python")["test"]
    logging.debug("Loaded dataset")

    llm = HuggingFacePipelineChatAdapter.from_model_name(model_name=model_name)

    if local:
        script_executor = LocalScriptExecutor(timeout_seconds=timeout_seconds)
    else:
        script_executor = DockerScriptExecutor(timeout_seconds=timeout_seconds)

    def init_agent():
        return HumanEvalFixAgent(
            llm=llm,
            script_executor=script_executor,
            max_iterations=max_iterations
        )

    correct_sum = 0
    total_sum = 0

    t = tqdm(range(test_samples or len(ds)), desc='0/0')
    for i in t:
        sample = ds[i]
        agent = init_agent()
        seed_all(seed)
        correct = evaluate_sample(agent, script_executor, sample)
        correct_sum += int(correct)
        total_sum += 1
        t.set_description(f"{correct_sum}/{total_sum}")

    accuracy = correct_sum / total_sum
    logging.info("Finished evaluating")
    logging.info(f"Results: {correct_sum}/{total_sum}.\nAccuracy: {accuracy:.3f}")
    return correct_sum / total_sum


if __name__ == '__main__':
    main()