from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseLLM
from langgraph.graph import StateGraph, END
import logging
import textwrap
from typing import TypedDict
from script_executor import ScriptExecutorInterface
from utils import (
    extract_failed_assert,
    extract_error_summary,
    extract_function_definition,
    wrap_code_in_block,
    create_test_script,
    validate_test_script,
    get_sampling_params
)
from prompts import (
    SYSTEM_PROMPT,
    DRAFT_PROMPT_TEMPLATE,
    REFLECT_PROMPT_TEMPLATE,
    REPEAT_PROMPT,
    SYNTAX_PROMPT
)


class AgentStateInputData(TypedDict):
    entry_point: str
    declaration: str
    docstring: str
    instruction: str
    buggy_solution: str
    test_script: str


class AgentStateConfig(TypedDict):
    max_iterations: int


class AgentStateExecutionResult(TypedDict):
    stdout: str
    stderr: str
    exit_code: int


class AgentState(TypedDict):
    input_data: AgentStateInputData
    config: AgentStateConfig
    messages: list[BaseMessage]
    candidate_solutions: list[str]
    execution_result: AgentStateExecutionResult | None
    iteration_number: int


def node_draft(state: AgentState, llm: BaseLLM) -> AgentState:
    buggy_implementation = state["input_data"]["declaration"] + textwrap.indent(state["input_data"]["buggy_solution"], "    ")
    tests = state["input_data"]["test_script"]
    docstring = state["input_data"]["docstring"]

    prompt = DRAFT_PROMPT_TEMPLATE.format(
        buggy_implementation=buggy_implementation,
        tests=tests,
        docstring=docstring
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    logging.debug("DRAFT: Invoking LLM")

    code = llm.invoke(messages)
    candidate_solution = extract_function_definition(code, state["input_data"]["entry_point"])

    logging.debug("DRAFT: Received candidate solution:")
    logging.debug(candidate_solution)

    state["messages"] += messages + [AIMessage(content=wrap_code_in_block(candidate_solution))]
    state["candidate_solutions"].append(candidate_solution)
    state["iteration_number"] = 0
    return state


def node_test(state: AgentState, script_executor: ScriptExecutorInterface) -> AgentState:
    code = state["candidate_solutions"][-1]
    script = create_test_script(
        declaration=state["input_data"]["declaration"],
        definition=code,
        test_script=state["input_data"]["test_script"]
    )

    logging.debug("TEST: Executing tests")
    response = script_executor.execute(script)
    logging.debug(f"TEST: Tests finished with exit code: {response.exit_code}")

    execution_result: AgentStateExecutionResult = {
        "stdout": response.stdout,
        "stderr": response.stderr,
        "exit_code": response.exit_code
    }
    state["execution_result"] = execution_result
    return state


def node_reflect(state: AgentState, llm: BaseLLM) -> AgentState:
    execution_result = state["execution_result"]
    if execution_result is not None:
        docstring = state["input_data"]["docstring"]
        stderr = execution_result["stderr"]
        assert_statement = extract_failed_assert(stderr)

        assert_block = ""
        error_block = ""

        if assert_statement is not None:
            assert_block = f"Failed assertion:\n```\n{assert_statement}\n```\n"
        else:
            error_summary = extract_error_summary(stderr)
            error_block = f"Error summary:\n```\n{error_summary}\n```\n"

        prompt = REFLECT_PROMPT_TEMPLATE.format(
            stderr=execution_result["stderr"],
            docstring=docstring,
            error_block=error_block,
            assert_block=assert_block
        )

        human_message = HumanMessage(content=prompt)
        messages = list(state["messages"]) + [human_message]

        logging.debug("REFLECT: Invoking LLM")
        code = llm.invoke(messages, config=get_sampling_params(iteration_number=state["iteration_number"]))

        candidate_solution = extract_function_definition(code, state["input_data"]["entry_point"])

        logging.debug("REFLECT: Received candidate function")
        logging.debug(candidate_solution)

        state["messages"] += [human_message, AIMessage(content=wrap_code_in_block(candidate_solution))]
        state["candidate_solutions"].append(candidate_solution)
    return state


def node_repeat(state: AgentState, llm: BaseLLM) -> AgentState:
    human_message = HumanMessage(content=REPEAT_PROMPT)
    messages = list(state["messages"]) + [human_message]

    logging.debug("REPEAT: Invoking LLM")
    code = llm.invoke(messages, config=get_sampling_params(iteration_number=state["iteration_number"]))
    candidate_solution = extract_function_definition(code, state["input_data"]["entry_point"])

    logging.debug("REPEAT: Received candidate function")
    logging.debug(candidate_solution)

    state["messages"] += [human_message, AIMessage(content=wrap_code_in_block(candidate_solution))]
    state["candidate_solutions"].append(candidate_solution)

    return state


def node_syntax(state: AgentState, llm: BaseLLM) -> AgentState:
    human_message = HumanMessage(content=SYNTAX_PROMPT)
    messages = list(state["messages"]) + [human_message]

    logging.debug("SYNTAX: Invoking LLM")
    code = llm.invoke(messages, config=get_sampling_params(iteration_number=state["iteration_number"]))
    candidate_solution = extract_function_definition(code, state["input_data"]["entry_point"])
    logging.debug("SYNTAX: Received candidate function")
    logging.debug(candidate_solution)

    state["messages"] += [human_message, AIMessage(content=wrap_code_in_block(candidate_solution))]
    state["candidate_solutions"].append(candidate_solution)

    return state


def node_iteration(state: AgentState) -> AgentState:
    state["iteration_number"] += 1
    logging.debug(f"Iteration {state['iteration_number']}")
    return state


class HumanEvalFixAgent:
    def __init__(self, llm: BaseLLM, script_executor: ScriptExecutorInterface, max_iterations: int = 5, recursion_limit: int = 50):
        self.llm = llm
        self.script_executor = script_executor
        self.max_iterations = max_iterations
        self.recursion_limit = recursion_limit

        self.compiled_graph = self.build_graph()
        logging.debug("Compiled graph")

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("draft", lambda state: node_draft(state, self.llm))
        graph.add_node("iteration", lambda state: node_iteration(state))
        graph.add_node("pretest-valid-check", lambda state: state)
        graph.add_node("pretest-repeat-check", lambda state: state)
        graph.add_node("test", lambda state: node_test(state, self.script_executor))
        graph.add_node("reflect", lambda state: node_reflect(state, self.llm))
        graph.add_node("syntax", lambda state: node_syntax(state, self.llm))
        graph.add_node("repeat", lambda state: node_repeat(state, self.llm))

        graph.add_node("guard-reflect", lambda state: state)
        graph.add_node("guard-syntax", lambda state: state)
        graph.add_node("guard-repeat", lambda state: state)

        def max_iterations_reached(state: AgentState) -> bool:
            return state["iteration_number"] >= state["config"]["max_iterations"]

        def test_success(state: AgentState) -> bool:
            execution_result = state["execution_result"]
            if execution_result is not None:
                passed = validate_test_script(
                    exit_code=execution_result["exit_code"],
                    stdout=execution_result["stdout"]
                )
                if passed:
                    return True
            return False

        def is_candidate_code_valid(state: AgentState) -> bool:
            return state["candidate_solutions"][-1] is not None

        def is_repeated(state: AgentState) -> bool:
            solutions = state["candidate_solutions"]
            if len(solutions) < 2:
                return False
            last_solution = solutions[-1].strip()
            for i in range(len(solutions) - 1):
                solution = solutions[i].strip()
                if last_solution == solution:
                    return True
            return False

        graph.set_entry_point("draft")
        graph.add_edge("draft", "pretest-valid-check")
        graph.add_edge("reflect", "iteration")
        graph.add_edge("repeat", "iteration")
        graph.add_edge("syntax", "iteration")
        graph.add_edge("iteration", "pretest-valid-check")
        graph.add_conditional_edges(
            "guard-reflect",
            max_iterations_reached,
            {
                False: "reflect",
                True: END
            }
        )
        graph.add_conditional_edges(
            "guard-repeat",
            max_iterations_reached,
            {
                False: "repeat",
                True: END
            }
        )
        graph.add_conditional_edges(
            "guard-syntax",
            max_iterations_reached,
            {
                False: "syntax",
                True: END
            }
        )
        graph.add_conditional_edges(
            "pretest-valid-check",
            is_candidate_code_valid,
            {
                True: "pretest-repeat-check",
                False: "guard-syntax"
            }
        )
        graph.add_conditional_edges(
            "pretest-repeat-check",
            is_repeated,
            {
                False: "test",
                True: "guard-repeat"
            }
        )
        graph.add_conditional_edges(
            "test",
            test_success,
            {
                False: "guard-reflect",
                True: END,
            }
        )

        return graph.compile()

    def run(self,
            entry_point: str,
            declaration: str,
            docstring: str,
            instruction: str,
            buggy_solution: str,
            test: str
    ) -> str:
        input_data: AgentStateInputData = {
            "entry_point": entry_point,
            "declaration": declaration,
            "docstring": docstring,
            "instruction": instruction,
            "buggy_solution": buggy_solution,
            "test_script": test
        }
        config: AgentStateConfig = {
            "max_iterations": self.max_iterations,
        }
        initial_state: AgentState = {
            "messages": [],
            "input_data": input_data,
            "config": config,
            "candidate_solutions": [],
            "execution_result": None,
            "iteration_number": 0
        }
        final_state = self.compiled_graph.invoke(initial_state, { "recursion_limit": self.recursion_limit })
        return final_state["candidate_solutions"][-1]
