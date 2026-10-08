"""Query an open-weight model on the puzzle set with vLLM.

    python -m chesscheck.query.local --model Qwen/Qwen3-8B --temperature 0.6 --top-p 0.95 \
        --max-tokens 16384 --max-model-len 17408 --batch-size 16 --num-rows 3000

scripts/local_model_configs.sh has each model's parameters. Any vLLM engine
argument is accepted. Output rows are {PuzzleId: text}, appended; resume with
--start-row.
"""
import json
import os

from vllm import LLM, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ..paths import LLM_PROMPT, RAW_LOCAL_DIR
from ..puzzles import SYSTEM_INSTRUCTION, iter_requests, user_content

SAMPLING_ARGS = ("max_tokens", "temperature", "top_p", "top_k", "min_p",
                 "presence_penalty", "repetition_penalty")


def create_parser():
    parser = FlexibleArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="mistralai/Mistral-7B-Instruct-v0.3")
    g = parser.add_argument_group("Sampling parameters")
    g.add_argument("--max-tokens", type=int)
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--min-p", type=float)
    g.add_argument("--presence-penalty", type=float)
    g.add_argument("--repetition-penalty", type=float)
    g.add_argument("--guided-json", type=str,
                   help="JSON Schema file; constrains decoding to schema-valid JSON (vLLM structured outputs)")
    parser.add_argument("--prompt-file", type=str, default=str(LLM_PROMPT))
    parser.add_argument("--output-dir", type=str, default=str(RAW_LOCAL_DIR))
    parser.add_argument("--test-only", action="store_true", help="print outputs, write nothing")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-rows", type=int, default=5000, help="puzzle rows to process")
    parser.add_argument("--start-row", type=int, default=0, help="first puzzle row (0-based)")
    return parser


def main(args: dict):
    sampling = {k: args.pop(k) for k in SAMPLING_ARGS}
    guided_json = args.pop("guided_json")
    test_only = args.pop("test_only")
    batch_size = args.pop("batch_size")
    num_rows = args.pop("num_rows")
    start_row = args.pop("start_row")
    prompt_file = args.pop("prompt_file")
    output_dir = args.pop("output_dir")

    llm = LLM(**args)
    params = llm.get_default_sampling_params()
    for k, v in sampling.items():
        if v is not None:
            setattr(params, k, v)
    if guided_json is not None:
        from vllm.sampling_params import StructuredOutputsParams
        with open(guided_json) as f:
            params.structured_outputs = StructuredOutputsParams(json=json.load(f))
        print(f"Structured outputs enabled from schema: {guided_json}")

    prompt_text = open(prompt_file, encoding="utf-8").read()
    prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]
    model_name = args["model"].split("/")[-1]
    output_file = os.path.join(output_dir, f"outputs_{prompt_name}_{model_name}.jsonl")
    if not test_only:
        os.makedirs(output_dir, exist_ok=True)

    def flush(batch):
        outputs = llm.chat([conv for _, conv in batch], params)
        assert len(outputs) == len(batch)
        texts = [o.outputs[0].text for o in outputs]
        if test_only:
            for (key, _), text in zip(batch, texts):
                print(f"Puzzle ID: {key}, Output: {text}")
        else:
            with open(output_file, "a", encoding="utf-8") as out:
                for (key, _), text in zip(batch, texts):
                    out.write(json.dumps({key: text}) + "\n")

    end_row = None if num_rows is None else start_row + num_rows
    batch, processed = [], 0
    print("=" * 80)
    for req in iter_requests(start_row, end_row):
        conversation = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content(req.fen, prompt_text)},
        ]
        if test_only:
            print("-" * 80)
            print(f"Puzzle ID: {req.key}")
            print(f"System instruction: {conversation[0]['content']}")
            print(f"Prompt: {conversation[1]['content']}")
        batch.append((req.key, conversation))
        if len(batch) == batch_size:
            flush(batch)
            processed += len(batch)
            print(f"Processed {processed} puzzles")
            batch = []
    if batch:
        flush(batch)
        processed += len(batch)
        print(f"Processed {processed} puzzles")


if __name__ == "__main__":
    main(vars(create_parser().parse_args()))
