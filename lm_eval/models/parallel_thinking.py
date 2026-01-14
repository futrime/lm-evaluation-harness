import concurrent.futures
import json
import re
from typing import Any, override

from openai import Omit, OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.utils import handle_stop_sequences


SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a thoughtful thinker. When you are given a goal, in your thinking process, you should first think of an outline to break down the goal into at most 4 smaller sub-goals, and then invoke other thinkers to think about those sub-goals. Unless the goal is too simple to break down, you should not think deeply about it by yourself.",
}

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What to think next.",
                },
            },
            "required": ["goal"],
        },
    },
}


class ParallelThinkingLM(LM):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        num_concurrent: int = 64,
        max_depth: int = 3,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.num_concurrent = num_concurrent
        self.max_depth = max_depth

        if base_url.endswith("/chat/completions"):
            base_url = base_url.rsplit("/chat/completions", 1)[0]
        self.base_url = base_url
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            use_fast=True,
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_concurrent
        )
        self.client = OpenAI(base_url=self.base_url, api_key="EMPTY")

    def __del__(self):
        self.executor.shutdown(wait=False)

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str | JsonChatStr | list[dict[str, str]]:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.model

    @override
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for request in tqdm(requests):
            results.append(self._generate_single(request))
        return results

    @override
    def loglikelihood(self, *args, **kwargs):
        raise NotImplementedError

    @override
    def loglikelihood_rolling(self, *args, **kwargs):
        raise NotImplementedError

    def _generate_single(self, request: Instance) -> str:
        prompt: str = request.args[0]
        gen_kwargs: dict[str, Any] = request.args[1] if len(request.args) > 1 else {}

        _ = gen_kwargs.pop("do_sample", None)
        max_tokens = gen_kwargs.pop("max_tokens", None) or gen_kwargs.pop(
            "max_gen_toks", None
        )
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos=None)

        # Move extra params to extra_body if provided, or ignore them if they cause issues
        # OpenAI client doesn't accept top_k/min_p directly.
        extra_body = {}
        if "top_k" in gen_kwargs:
            extra_body["top_k"] = gen_kwargs.pop("top_k")
        if "min_p" in gen_kwargs:
            extra_body["min_p"] = gen_kwargs.pop("min_p")

        if extra_body:
            gen_kwargs["extra_body"] = extra_body

        messages = self._invoke_thinker(
            goal=prompt,
            history=[SYSTEM_MESSAGE],
            depth=0,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            gen_kwargs=gen_kwargs,
        )

        if not messages:
            return ""

        # Aggregate all assistant messages to form the context for the final answer
        aggregated_thought = "".join(
            (msg.get("reasoning_content") or "") + (msg.get("content") or "")
            for msg in messages
            if msg["role"] == "assistant"
        )

        # Prepare the final request to synthesize the answer
        final_prompt = (
            "Now, based on all the thinking done, please provide a comprehensive thesis for the final answer to the original goal: "
            + prompt
        )

        final_messages = [
            SYSTEM_MESSAGE,
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": aggregated_thought},
            {"role": "user", "content": final_prompt},
        ]

        # print(f"DEBUG: Final Messages: {final_messages}")

        response = self.client.chat.completions.create(
            messages=final_messages,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            tools=Omit(),
            **gen_kwargs,
        )

        msg = response.choices[0].message
        content = msg.content or ""
        # Check for reasoning content if available (SGLang specific)
        if hasattr(msg, "reasoning_content"):
            content += msg.reasoning_content or ""
        elif msg.model_extra and "reasoning_content" in msg.model_extra:
            content += msg.model_extra["reasoning_content"] or ""

        if not content.strip():
            return aggregated_thought

        return content

    def _invoke_thinker(
        self,
        goal: str,
        history: list[dict],
        depth: int,
        max_tokens: int | None,
        temperature: float,
        stop: list[str] | None,
        gen_kwargs: dict[str, Any],
        tool_call_id: str | None = None,
    ) -> list[dict]:
        enable_tool = depth < self.max_depth
        instruction = (
            "\nYou should directly break it down into smaller sub-goals and invoke others to think about them rather than think deeply about it, unless the goal is too simple to break down. No need to worry about the number of sub-goals, just focus on the quality of the breakdown."
            if depth < self.max_depth
            else "\nYou shouldn't break it down further. You should think deeply about it by yourself."
        )
        user_message = {
            "role": "user",
            "content": goal + instruction,
        }

        response = self.client.chat.completions.create(
            messages=history + [user_message],
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            tools=[THINK_TOOL] if enable_tool else Omit(),
            **gen_kwargs,
        )

        msg = response.choices[0].message
        assistant_message = msg.model_dump()

        # Explicitly capture reasoning_content if present on the object
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            assistant_message["reasoning_content"] = msg.reasoning_content

        # Ensure reasoning_content is preserved in assistant_message from model_extra if needed
        if (
            not assistant_message.get("reasoning_content")
            and msg.model_extra
            and "reasoning_content" in msg.model_extra
        ):
            assistant_message["reasoning_content"] = msg.model_extra[
                "reasoning_content"
            ]

        # Collect sub-goals from tool calls
        sub_goals = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "think":
                    args = json.loads(tc.function.arguments)
                    sub_goal = args.get("goal", "")
                    if sub_goal and sub_goal != goal:
                        sub_goals.append((sub_goal, tc.id))

        # Fallback: Parse tool calls from content if not present
        content_to_parse = (assistant_message.get("reasoning_content") or "") + (
            assistant_message.get("content") or ""
        )
        if not sub_goals and content_to_parse:
            tool_call_regex = r"<tool_call>\s*({.*?})\s*</tool_call>"
            matches = re.findall(tool_call_regex, content_to_parse, re.DOTALL)
            for match in matches:
                try:
                    args = json.loads(match)
                    # Handle both structure formats: direct args or {"name":..., "arguments": ...}
                    if "arguments" in args:
                        args = args["arguments"]

                    sub_goal = args.get("goal", "")
                    if sub_goal and sub_goal != goal:
                        # Use None for tool_call_id to trigger User role for text-based tools
                        sub_goals.append((sub_goal, None))
                except json.JSONDecodeError:
                    pass

        sub_messages = []
        if sub_goals:
            # Create a clean assistant message for history (remove tool calls to avoid API validation errors)
            clean_assistant_message = assistant_message.copy()
            clean_assistant_message.pop("tool_calls", None)
            if clean_assistant_message.get("content") is None:
                clean_assistant_message["content"] = ""

            print(
                f"DEBUG: Processing {len(sub_goals)} sub-goals in parallel: {[sg[0] for sg in sub_goals]}"
            )
            futures = [
                self.executor.submit(
                    self._invoke_thinker,
                    goal=sub_goal,
                    history=history + [user_message, clean_assistant_message],
                    depth=depth + 1,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    gen_kwargs=gen_kwargs,
                    tool_call_id=tool_call_id,
                )
                for sub_goal, tool_call_id in sub_goals
            ]

            # Wait for all immediate futures to complete
            # Recursive calls inside _invoke_thinker will spawn their own futures
            results = [f.result() for f in futures]

            for result in results:
                sub_messages.extend(result)

        return [user_message, assistant_message] + sub_messages
