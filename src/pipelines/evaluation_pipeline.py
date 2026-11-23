import os
import json
import time
import torch
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.core.config import AppConfig
from src.models.router_agent import RouterAgent
from src.models.validator import Validator
from src.models.ast_checker import check_sample, summarise
from src.pipelines.splits import load_test_split
from src.adapters.base_parser import BaseDatasetParser

class EvaluationPipeline:
    def __init__(self, config: AppConfig, parser: BaseDatasetParser):
        self.config = config
        self.parser = parser
        
    def execute(self, num_samples: int = 2000):
        agent = RouterAgent(self.config)
        lora_path = os.path.join(self.config.training.output_dir, "lora_model")

        if not os.path.exists(lora_path):
            print(f"Model not found at {lora_path}. Please train first.")
            return

        model, tokenizer = agent.load_for_inference(lora_path)

        # Held-out split only (created before training by the data pipeline).
        dataset = load_test_split()
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        print(f"Evaluating on {len(dataset)} held-out samples (never seen in training)")

        reports_dir = os.path.join(self.config.training.output_dir, "evaluation_reports")
        os.makedirs(reports_dir, exist_ok=True)
        trace_path = os.path.join(reports_dir, "evaluation_debug_trace.jsonl")

        from src.pipelines.data_pipeline import SYSTEM_PROMPT

        results, json_flags = [], []
        total_latency_ms, total_tokens = 0.0, 0

        with open(trace_path, "w") as f_trace:
            for item in tqdm(dataset, desc="Evaluating"):
                try:
                    record = self.parser.transform(item)
                except Exception:
                    continue

                tools_json = record.tools if isinstance(record.tools, list) else []
                try:
                    expected = json.loads(record.answers) if record.answers and record.answers != "[]" else []
                    gt_error = False
                except json.JSONDecodeError:
                    expected, gt_error = [], True
                if isinstance(expected, dict):
                    expected = [expected]

                user_message = f"Available tools:\n{json.dumps(tools_json, indent=2)}\n\nUser Query: {record.query}"
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message}]
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                ).to("cuda")

                start_time = time.time()
                outputs = model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=512, max_length=None, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                latency_ms = (time.time() - start_time) * 1000
                response_tokens = outputs[0][inputs.shape[1]:]
                total_latency_ms += latency_ms
                total_tokens += len(response_tokens)

                response = tokenizer.decode(response_tokens, skip_special_tokens=True)
                parsed = Validator.parse_generated_text(response)
                predicted = parsed["parsed_json"]
                if isinstance(predicted, dict):
                    predicted = [predicted]

                result = check_sample(
                    predicted_calls=predicted if parsed["is_valid_json"] else None,
                    expected_calls=expected,
                    tools=tools_json,
                    is_abort=parsed["is_abort"],
                    has_tool_call_block=parsed["has_tool_call"],
                    is_valid_json=parsed["is_valid_json"],
                )
                results.append(result)
                json_flags.append(parsed["is_valid_json"] if parsed["has_tool_call"] else None)

                f_trace.write(json.dumps({
                    "sample_id": len(results),
                    "query": record.query,
                    "available_tools": [t.get("name") for t in tools_json if isinstance(t, dict)],
                    "expected_calls": expected,
                    "ground_truth_parsing_error": gt_error,
                    "raw_response": response,
                    "validator_output": parsed,
                    "outcome": result.outcome,
                    "error_category": result.error,
                    "per_call": [{"passed": c.passed, "error": c.error, "detail": c.detail} for c in result.per_call],
                    "latency_ms": latency_ms,
                    "generated_tokens": len(response_tokens),
                }, default=str) + "\n")

        final_report = summarise(results, json_flags)
        final_report["held_out"] = True
        final_report["performance"] = {
            "avg_latency_ms": total_latency_ms / max(1, len(results)),
            "tokens_per_second": total_tokens / (total_latency_ms / 1000) if total_latency_ms > 0 else 0,
        }

        report_path = os.path.join(reports_dir, "bfcl_advanced_metrics.json")
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=4)

        cm, errs = final_report["confusion_matrix"], final_report["error_distribution"]
        self._generate_plots({**cm, **{f"error_{k}": v for k, v in errs.items()}}, reports_dir)

        print(f"Accuracy: {final_report['accuracy']:.3f} | valid JSON: {final_report['valid_json_rate']:.3f}")
        print(f"Evaluation complete. Metrics saved to {report_path}; trace at {trace_path}")
        return final_report

    def _generate_plots(self, metrics, reports_dir):
        try:
            # Confusion Matrix
            plt.figure(figsize=(6,5))
            matrix = [[metrics["true_positives"], metrics["false_positives"]],
                      [metrics["false_negatives"], metrics["true_negatives"]]]
            
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Positive", "Negative"],
                        yticklabels=["Positive", "Negative"])
            plt.ylabel('Actual (Ground Truth)')
            plt.xlabel('Predicted')
            plt.title('Agentic Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"))
            plt.close()
            
            # Error Breakdown
            plt.figure(figsize=(10,6))
            error_types = ['Syntax', 'Missing Tool Call', 'Hallucinated Tool', 'Wrong Tool',
                           'Hallucinated Param', 'Missing Param', 'Type Mismatch', 'Value Mismatch', 'Call Count']
            error_counts = [
                metrics.get(k, 0) for k in (
                    "error_syntax", "error_missing_tool", "error_hallucinated_tool", "error_wrong_tool",
                    "error_hallucinated_param", "error_missing_param", "error_type_mismatch",
                    "error_value_mismatch", "error_call_count")
            ]
            
            sns.barplot(x=error_counts, y=error_types, palette="Reds_r")
            plt.title('Failure Breakdown Analysis')
            plt.xlabel('Number of Samples')
            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, "failure_breakdown.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to generate plots. Ensure matplotlib/seaborn are installed. {e}")
