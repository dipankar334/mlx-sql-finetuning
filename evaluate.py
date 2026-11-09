import json
import sys
from mlx_lm import load, generate

model_path = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
adapter_path = sys.argv[2] if len(sys.argv) > 2 else None

print(f"\nLoading model: {model_path} (Adapter: {adapter_path or 'None'})...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

with open("data/test.jsonl") as f:
    test_rows = [json.loads(line) for line in f]

exact_matches = 0
print(f"Running evaluation on {len(test_rows)} unseen test queries...\n" + "-"*60)

for idx, row in enumerate(test_rows, 1):
    messages = row["messages"][:2]  # system + user prompt only
    expected_sql = row["messages"][2]["content"].strip()

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prediction = generate(model, tokenizer, prompt=prompt, max_tokens=150, verbose=False).strip()

    is_match = (prediction == expected_sql)
    if is_match:
        exact_matches += 1

    print(f"[{idx}/{len(test_rows)}] Question: {messages[1]['content']}")
    print(f"   Expected : {expected_sql}")
    print(f"   Predicted: {prediction}")
    print(f"   Result   : {'✅ EXACT MATCH' if is_match else '❌ MISMATCH'}\n")

accuracy = (exact_matches / len(test_rows)) * 100
print("="*60)
print(f"FINAL EXACT-MATCH ACCURACY: {exact_matches}/{len(test_rows)} ({accuracy:.1f}%)")
print("="*60)
