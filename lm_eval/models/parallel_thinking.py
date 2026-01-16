import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import cast, override

from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are an expert problem solver using parallel thinking. "
        "Your goal is to solve the user's request by breaking it down into independent sub-goals.\n"
        "1. **Analyze**: Understand the problem and identify the key components.\n"
        "2. **Decompose**: Call the `think` tool with a list of *independent* sub-goals. "
        "Each sub-goal should be a self-contained task that contributes to the final solution. "
        "Avoid making sub-goals that depend on each other if possible.\n"
        "3. **Solve**: The system will provide the sub-goal as a tool output. "
        "You must strictly provide the solution for THAT specific sub-goal.\n"
        "4. **Synthesize**: After all sub-goals are solved, combine the results to determine the final answer.\n"
        "IMPORTANT: The final answer must be clear and direct."
    ),
}

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Invoke thinkers to think about sub-goals.",
        "parameters": {
            "type": "object",
            "properties": {
                "sub_goals": {
                    "type": "array",
                    "description": "A list of sub-goals to think about.",
                    "items": {
                        "type": "string",
                    },
                },
            },
            "required": ["sub_goals"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class ParallelThinkingLM(LocalChatCompletion):
    def __init__(self, num_concurrent: int = 1, max_depth: int = 1, **kwargs):
        super().__init__(**kwargs)

        self._num_concurrent = num_concurrent  # This is not self._concurrent.
        self._max_depth = max_depth

    @staticmethod
    def parse_generations(outputs: list[dict] | dict, **kwargs) -> list[dict]:
        if not isinstance(outputs, list):
            outputs = [outputs]

        res = []
        for out in outputs:
            message: dict = out["choices"][0]["message"]

            result = {
                "role": message["role"],
                "reasoning_content": message.get("reasoning_content"),
                "content": message["content"],
                "tool_calls": message.get("tool_calls"),
            }

            res.append(result)

        return res

    @override
    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        concurrent_semaphore = Semaphore(self._num_concurrent)

        def process_request(request: Instance) -> str:
            prompt = request.args[0]
            assert isinstance(prompt, JsonChatStr)
            history = [SYSTEM_MESSAGE] + json.loads(prompt.prompt)

            gen_kwargs: dict = request.args[1] if len(request.args) > 1 else {}
            gen_kwargs = gen_kwargs.copy()
            gen_kwargs["tools"] = [THINK_TOOL]

            messages = self._invoke_thinker(
                history=history,
                gen_kwargs=gen_kwargs,
                concurrent_semaphore=concurrent_semaphore,
            )

            return messages[-1]["content"]

        with ThreadPoolExecutor(max_workers=self._num_concurrent) as executor:
            futures = [executor.submit(process_request, req) for req in requests]

            # To make progress bar more responsive.
            for _ in tqdm(
                as_completed(futures), total=len(requests), disable=disable_tqdm
            ):
                pass

            results = [f.result() for f in futures]

        return results

    def _invoke_thinker(
        self,
        history: list[dict],
        gen_kwargs: dict,
        concurrent_semaphore: Semaphore,
        depth: int = 0,
        goal: str | None = None,
        tool_call_id: str | None = None,
    ) -> list[dict]:
        """Thinks about a goal.

        Args:
            history: The message history.
            gen_kwargs: Generation keyword arguments.
            depth: The current depth of thinking.
            concurrent_semaphore: Semaphore to control concurrency.
            goal: The goal to think about.
            tool_call_id: The tool call ID if this is invoked via a tool call.

        Note:
            `goal` and `tool_call_id` must be provided together, or both be None.

        Returns:
            New messages without the initial history.
        """

        # Step 1: Think.
        if history[-1]["role"] == "assistant":
            assert goal is not None and tool_call_id is not None

            step1_history = history + [
                {
                    "role": "tool",
                    "content": goal,
                    "tool_call_id": tool_call_id,
                }
            ]

        else:
            step1_history = history

        with concurrent_semaphore:
            step1_response = cast(
                "dict",
                super().generate_until(
                    [
                        Instance(
                            request_type="generate_until",
                            arguments=(step1_history, gen_kwargs),
                            idx=0,
                            doc={},
                        )
                    ],
                    disable_tqdm=True,  # pyright: ignore[reportCallIssue]
                    disable_len_check_warn=True,  # pyright: ignore[reportCallIssue]
                )[0],
            )

        # Step 2: Process sub-goals.
        sub_goals = [
            sub_goal
            for tool_call in step1_response["tool_calls"] or []
            if tool_call["function"]["name"] == "think"
            for sub_goal in json.loads(tool_call["function"]["arguments"])["sub_goals"]
        ]

        if sub_goals == []:
            # No sub-goals generated, return the response as is.
            return (step1_history + [step1_response])[len(history) :]

        tool_call_id = next(
            (
                tool_call["id"]
                for tool_call in step1_response["tool_calls"]
                if tool_call["function"]["name"] == "think"
            ),
            None,
        )

        step2_history = step1_history + [step1_response]

        if depth < self._max_depth:

            def process_sub_goal(sub_goal: str) -> list[dict]:
                return self._invoke_thinker(
                    history=step2_history,
                    gen_kwargs=gen_kwargs,
                    concurrent_semaphore=concurrent_semaphore,
                    depth=depth + 1,
                    goal=sub_goal,
                    tool_call_id=tool_call_id,
                )

            with ThreadPoolExecutor() as executor:
                step2_responses = [
                    msg
                    for msgs in executor.map(process_sub_goal, sub_goals)
                    for msg in msgs
                ]

        else:
            step2_responses = []

        # Step 3: Tail invocation.
        step3_history = (
            step2_history
            + step2_responses
            + [
                {
                    "role": "tool",
                    "content": ""
                    if depth < self._max_depth
                    else "ERROR: Max depth reached.",
                    "tool_call_id": tool_call_id,
                }
            ]
        )

        step3_resp_msgs = self._invoke_thinker(
            history=step3_history,
            gen_kwargs=gen_kwargs,
            concurrent_semaphore=concurrent_semaphore,
            depth=depth,
        )

        return (step3_history + step3_resp_msgs)[len(history) :]
