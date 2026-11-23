import os
import json
import random
from src.core.config import AppConfig
from src.core.exceptions import DataPipelineException, SchemaValidationError
from src.adapters.base_parser import BaseDatasetParser
from pydantic import ValidationError
from src.models.ast_checker import required_params
from src.pipelines.splits import make_splits

SYSTEM_PROMPT = """You are an expert deterministic tool-calling agent. You are provided with a user query and a list of available tools.
You must follow a strict reasoning process inside <thought> tags before deciding whether to call a tool or not.

Your thought process MUST follow this exact structure:
1. Tool Assessment: List the available tools from the system prompt.
2. Intent Matching: State the user's core request and identify which tool (if any) matches it.
3. Parameter Extraction: Explicitly extract the required parameters from the user's prompt.
4. Validation: Check if all required parameters are present. If a parameter is missing, or if no tool matches, output an abort sequence (e.g., [ABORT: Missing parameter '...'] or [ABORT: No matching tool]) and generate a polite conversational refusal instead of a <tool_call>.

If validation passes, output the tool call strictly within <tool_call> tags as a JSON object.
"""

def make_no_tool_negative(record, rate: float, rng: random.Random):
    """With probability `rate`, turn a positive example into a "no matching tool"
    negative by removing the target tool(s) from the tool list."""
    if rate <= 0 or not record.answers or record.answers == "[]" or rng.random() >= rate:
        return record
    try:
        answers = json.loads(record.answers)
    except json.JSONDecodeError:
        return record
    answers = answers if isinstance(answers, list) else [answers]
    targets = {a.get("name") for a in answers if isinstance(a, dict)}
    remaining = [
        t for t in record.tools
        if (t["function"].get("name") if isinstance(t.get("function"), dict) else t.get("name")) not in targets
    ]
    return record.model_copy(update={"tools": remaining, "answers": "[]"})


class DataPipeline:
    def __init__(self, config: AppConfig, parser: BaseDatasetParser, tokenizer):
        self.config = config
        self.parser = parser
        self.tokenizer = tokenizer
        
    def generate_prompt_for_record(self, record):
        query = record.query
        tools_json = record.tools

        answers_json = []
        if record.answers and record.answers != "[]":
            try:
                answers_json = json.loads(record.answers)
            except json.JSONDecodeError:
                pass
        if isinstance(answers_json, dict):
            answers_json = [answers_json]

        tools_str = json.dumps(tools_json, indent=2)
        user_message = f"Available tools:\n{tools_str}\n\nUser Query: {query}"

        available_tool_names = [
            n for n in (
                (t["function"].get("name") if isinstance(t.get("function"), dict) else t.get("name"))
                for t in tools_json if isinstance(t, dict)
            ) if n
        ] if isinstance(tools_json, list) else []
        tool_list_str = ", ".join(available_tool_names) if available_tool_names else "None"

        if not answers_json:
            thought = (
                "<thought>\n"
                f"1. Tool Assessment: The available tools are: {tool_list_str}.\n"
                f"2. Intent Matching: The user's query is '{query}'. None of the available tools match this intent.\n"
                "3. Parameter Extraction: N/A\n"
                "4. Validation: [ABORT: No matching tool]\n"
                "</thought>"
            )
            assistant_response = f"{thought}\nI'm sorry, but I don't have the tools to help you with that request."
        else:
            # Describe every call (xLAM often has parallel calls), with required
            # parameters read from each tool's schema.
            intents, extractions = [], []
            for call in answers_json:
                name = call.get("name", "unknown") if isinstance(call, dict) else "unknown"
                args = call.get("arguments", {}) if isinstance(call, dict) else {}
                req = required_params(tools_json if isinstance(tools_json, list) else [], name)
                intents.append(f"'{name}'")
                extractions.append(
                    f"{name}: required {req if req else 'none'}; values {json.dumps(args, ensure_ascii=False)}"
                )
            n = len(answers_json)
            thought = (
                "<thought>\n"
                f"1. Tool Assessment: The available tools are: {tool_list_str}.\n"
                f"2. Intent Matching: The user's query is '{query}'. "
                f"{'The tool' if n == 1 else f'{n} calls are needed:'} {', '.join(intents)} "
                f"{'matches' if n == 1 else 'match'} this intent.\n"
                f"3. Parameter Extraction: {' | '.join(extractions)}.\n"
                "4. Validation: All required parameters are present. Proceeding with tool call.\n"
                "</thought>"
            )
            tool_call_str = json.dumps(answers_json, indent=2, ensure_ascii=False)
            assistant_response = f"{thought}\n<tool_call>\n{tool_call_str}\n</tool_call>"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def process(self):
        dataset_name = self.config.data.dataset_name

        try:
            # Train rows only: the held-out test split is created here and never trained on.
            dataset, test_split = make_splits(self.config)
        except Exception as e:
            raise DataPipelineException(f"Failed to load dataset '{dataset_name}': {e}")
        print(f"Training rows: {len(dataset)} | held-out test rows: {len(test_split)}")
            
        if len(dataset) > 0:
            first_row = dataset[0]
            try:
                self.parser.transform(first_row)
            except SchemaValidationError as e:
                raise e
            except ValidationError as e:
                raise SchemaValidationError(f"Pydantic schema validation failed on first row: {e}")
            except Exception as e:
                raise SchemaValidationError(f"Unexpected error validating first row: {e}")
                
        rng = random.Random(self.config.data.split_seed)

        def format_func(examples):
            texts = []
            keys = list(examples.keys())
            num_samples = len(examples[keys[0]])
            
            for i in range(num_samples):
                raw_record = {k: examples[k][i] for k in keys}
                try:
                    record = self.parser.transform(raw_record)
                    record = make_no_tool_negative(record, self.config.data.no_tool_negative_rate, rng)
                    text = self.generate_prompt_for_record(record)
                    texts.append(text)
                except Exception as e:
                    raise DataPipelineException(f"Failed to process record: {e}")
            return {"text": texts}
            
        # num_proc=1 keeps negative sampling deterministic for a given seed
        processed_dataset = dataset.map(format_func, batched=True, num_proc=1)
        
        out_dir = "./data/processed_dataset"
        os.makedirs(out_dir, exist_ok=True)
        processed_dataset.save_to_disk(out_dir)
        print(f"Dataset successfully mapped and saved to {out_dir}")
        return processed_dataset
