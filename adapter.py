from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
import logging
from transformers import Pipeline, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline


class HuggingFacePipelineChatAdapter:
    def __init__(self, pipe: Pipeline, tokenizer: PreTrainedTokenizer):
        self.pipe = pipe
        self.tokenizer = tokenizer

    def _call_pipeline(self, prompt: str, config: dict | None = None) -> dict:
        if config is None:
            config = {}

        logging.debug(f"Calling pipeline with config: {config}")
        return self.pipe(
            prompt,
            do_sample=config.get("do_sample", False),
            temperature=config.get("temperature", 0.1),
            top_p=config.get("top_p", 0.95)
        )

    def invoke(
        self,
        messages: list[BaseMessage],
        config: dict | None = None
    ) -> str:
        def get_role(message: BaseMessage):
            if isinstance(message, SystemMessage):
                return "system"
            if isinstance(message, AIMessage):
                return "assistant"
            return "user"

        chat = []
        for message in messages:
            role = get_role(message)
            chat.append({ "role": role, "content": str(message.content) })
        prompt: str = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        output = self._call_pipeline(prompt, config)
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
                return text
        logging.warning("Failed to get generated_text from pipeline. Returning string output")
        return str(output)


    @staticmethod
    def from_model_name(model_name: str, max_new_tokens: int = 512):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            return_full_text=False
        )

        return HuggingFacePipelineChatAdapter(
            pipe=pipe,
            tokenizer=tokenizer
        )
