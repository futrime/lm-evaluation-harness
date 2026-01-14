from typing import cast, override

from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalChatCompletion


SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a thoughtful thinker. When you are given a goal, in your "
        "thinking process, you should first think of an outline to break down "
        "the goal into smaller sub-goals, and then invoke other thinkers to "
        "think about those sub-goals. Adapt the number of sub-goals to the "
        "complexity of the original goal. For complex goals, you should break "
        "them down into as many independent sub-goals as possible to maximize "
        "parallel thinking. Unless the goal is too simple to break down, you "
        "should not think deeply about it by yourself."
    ),
}

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Invoke a thinker to think about a specific goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The specific goal to think about.",
                },
            },
            "required": ["goal"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class ParallelThinkingLM(LocalChatCompletion):
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
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []

        for request in tqdm(requests):
            prompt: str = request.args[0]
            prompt = (
                prompt
                + " Please reason step by step, and put your final answer within \\boxed{}."
            )

            gen_kwargs: dict = request.args[1] if len(request.args) > 1 else {}
            gen_kwargs = gen_kwargs.copy()
            gen_kwargs["tools"] = [THINK_TOOL]

            messages = self._invoke_thinker(
                goal=prompt,
                history=[SYSTEM_MESSAGE],
                depth=0,
                gen_kwargs=gen_kwargs,
            )

            results.append(messages[-1]["content"])

        return results

    def _invoke_thinker(
        self,
        goal: str,
        history: list[dict],
        depth: int,
        gen_kwargs: dict,
        tool_call_id: str | None = None,
    ) -> list[dict]:
        """Thinks about a goal.

        Args:
            goal: The goal to think about.
            history: The message history. Should ends with an assistant message
                or a system message.
            depth: The current depth of thinking.
            gen_kwargs: Generation keyword arguments.
            tool_call_id: If provided, indicates that this thinker is invoked.

        Returns:
            New messages without the initial history.
        """

        # Step 1: Think.
        if tool_call_id is not None:
            step1_messages = history + [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Think about: {goal}",
                }
            ]
        else:
            step1_messages = history + [
                {
                    "role": "user",
                    "content": goal,
                }
            ]

        step1_response = cast(
            "dict",
            super().generate_until(
                [
                    Instance(
                        request_type="generate_until",
                        arguments=(step1_messages, gen_kwargs),
                        idx=0,
                        doc={},
                    )
                ]
            )[0],
        )

        think_tool_calls = [
            tc
            for tc in (step1_response["tool_calls"] or [])
            if tc["function"]["name"] == "think"
        ]

        if think_tool_calls == []:
            # No sub-goals generated, return the response as is.
            return (step1_messages + [step1_response])[len(history) :]

        # Step 2: Process sub-goals.
        step2_messages = step1_messages + [step1_response]

        sub_resp_messages: list[list[dict]] = []
        for think_tool_call in think_tool_calls:
            sub_goal: str = think_tool_call["arguments"]["goal"]

            sub_resp_msgs = self._invoke_thinker(
                goal=sub_goal,
                history=step2_messages,
                depth=depth + 1,
                gen_kwargs=gen_kwargs,
                tool_call_id=think_tool_call["id"],
            )

            sub_resp_messages.append(sub_resp_msgs)

        # Step 3: Tail invocation.
        step3_messages = step2_messages + [
            msg for msgs in sub_resp_messages for msg in msgs
        ]

        step3_resp_msgs = self._invoke_thinker(
            goal="Now that you have thought about the sub-goals, please "
            "aggregate them and continue your thinking about the original "
            "goal. If the original goal has been sufficiently addressed by the "
            "sub-goals, you may conclude your thinking",
            history=step3_messages,
            depth=depth,
            gen_kwargs=gen_kwargs,
        )

        return (step3_messages + step3_resp_msgs)[len(history) :]
