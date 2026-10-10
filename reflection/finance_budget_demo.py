#!/usr/bin/env python3
"""
Reflection pattern, finance edition: an AI drafts a monthly household budget.

Same idea as reflection_demo.py, in a setting everyone relates to:
  1. The model drafts a budget as JSON.
  2. Cheap, rule-based checks run first. No LLM needed for these:
     does it add up to income? are fixed bills right? is enough being saved?
  3. All checks pass -> stop, no extra cost.
     Any check fails -> the model is told exactly what's wrong ("over by ₹1,200"),
     critiques its own budget and fixes it. Up to --max-rounds times.
  4. Feedback is specific: Python works out the exact numbers to change
     ("savings ₹13,750 → ₹40,750"), because models are bad at checking their own maths.

Usage:
  python finance_budget_demo.py --model llama3.2 --runs 5
  python finance_budget_demo.py --model llama3.2 --runs 5 --always-reflect
  python finance_budget_demo.py --model qwen2.5-coder:7b --runs 5

Change the numbers in the SCENARIO section to try your own budget.
"""
import argparse
import json
import os
import re

from reflection_demo import chat  # same Ollama call, tokens and timing

# ---------------- SCENARIO ------------------
INCOME = 100000                                   # monthly take-home, ₹
FIXED = {"rent": 22000, "car_emi": 9500}         # must be kept exactly
CATEGORIES = ["rent", "car_emi", "groceries", "utilities", "transport",
              "insurance", "savings", "emergency_fund",
              "dining_out", "shopping", "entertainment"]
MIN_SAVINGS_PCT = 20     # savings + emergency_fund >= 20% of income
MAX_WANTS_PCT = 15       # dining_out + shopping + entertainment <= 15% of income
MIN_GROCERIES = 8000
# ---------------------------------------------------------

CAT_LIST = ", ".join(CATEGORIES)

TASK = f"""You are helping plan a monthly household budget in Indian rupees.
Monthly take-home income: ₹{INCOME:,}
Rules:
- Use exactly these categories and no others: {CAT_LIST}
- rent must be exactly ₹{FIXED['rent']:,} and car_emi exactly ₹{FIXED['car_emi']:,}
- Every amount is a whole number of rupees greater than 0
- All categories together must add up to exactly ₹{INCOME:,}
- savings + emergency_fund must be at least {MIN_SAVINGS_PCT}% of income
- dining_out + shopping + entertainment must be at most {MAX_WANTS_PCT}% of income
- groceries must be at least ₹{MIN_GROCERIES:,}
Return only a JSON object mapping each category to its amount, in a single ```json block."""

REFLECT = """Review your budget critically.
{feedback}
First list what is wrong in 1-3 short bullets.
Then give the full corrected budget as a JSON object in a single ```json block."""

CHECKS = ["valid JSON", "exact categories", "whole amounts above 0", "rent kept",
          "car EMI kept", "adds up to income", "savings rate", "wants cap",
          "groceries minimum"]


def extract_budget(text):
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def is_amount(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and float(v).is_integer() and v > 0)


def check_budget(reply):
    """The cheap check: plain Python rules, no LLM. Returns (passed, failures)."""
    data = extract_budget(reply)
    if data is None:
        return 0, ["The reply was not a valid JSON object of category -> amount."]

    failures = []
    amt = lambda k: data[k] if k in data and is_amount(data[k]) else 0

    missing = [c for c in CATEGORIES if c not in data]
    extra = [k for k in data if k not in CATEGORIES]
    if missing or extra:
        failures.append(f"Categories are wrong. Missing: {missing or 'none'}. "
                        f"Not allowed: {extra or 'none'}.")

    bad = [k for k, v in data.items() if not is_amount(v)]
    if bad:
        failures.append(f"These amounts are not whole rupees above 0: {', '.join(bad)}.")

    for key, label in [("rent", "rent"), ("car_emi", "car_emi")]:
        if amt(key) != FIXED[key]:
            failures.append(f"{label} is ₹{amt(key):,.0f}, but must be exactly ₹{FIXED[key]:,}.")

    total = sum(v for v in data.values() if is_amount(v))
    if total != INCOME:
        side = "over" if total > INCOME else "short"
        failures.append(f"The budget adds up to ₹{total:,.0f}, but income is ₹{INCOME:,} "
                        f"({side} by ₹{abs(total - INCOME):,.0f}).")

    saved = amt("savings") + amt("emergency_fund")
    need = INCOME * MIN_SAVINGS_PCT / 100
    if saved < need:
        failures.append(f"savings + emergency_fund is ₹{saved:,.0f} ({saved / INCOME:.1%}), "
                        f"but must be at least {MIN_SAVINGS_PCT}% (₹{need:,.0f}).")

    wants = amt("dining_out") + amt("shopping") + amt("entertainment")
    cap = INCOME * MAX_WANTS_PCT / 100
    if wants > cap:
        failures.append(f"dining_out + shopping + entertainment is ₹{wants:,.0f} "
                        f"({wants / INCOME:.1%}), but must be at most {MAX_WANTS_PCT}% (₹{cap:,.0f}).")

    if amt("groceries") < MIN_GROCERIES:
        failures.append(f"groceries is ₹{amt('groceries'):,.0f}, "
                        f"but must be at least ₹{MIN_GROCERIES:,}.")

    return len(CHECKS) - len(failures), failures


def suggest_fix(reply):
    """Exact changes that would make the budget pass every rule.
    Python does the arithmetic, so the model only has to copy the numbers."""
    data = extract_budget(reply)
    if (data is None or set(data) != set(CATEGORIES)
            or not all(is_amount(v) for v in data.values())):
        return []  # the failure messages already explain these problems
    new = dict(data)
    new.update(FIXED)
    new["groceries"] = max(new["groceries"], MIN_GROCERIES)
    need = INCOME * MIN_SAVINGS_PCT // 100

    # Wants over the cap: scale them all down evenly, rounded to ₹100
    wants = ["dining_out", "shopping", "entertainment"]
    total_wants, cap = sum(new[k] for k in wants), INCOME * MAX_WANTS_PCT // 100
    if total_wants > cap:
        for k in wants:
            new[k] = max(100, new[k] * cap // total_wants // 100 * 100)
        new[max(wants, key=new.get)] -= max(sum(new[k] for k in wants) - cap, 0)

    # Saving too little: top up savings
    new["savings"] += max(need - new["savings"] - new["emergency_fund"], 0)

    # Total doesn't match income
    diff = INCOME - sum(new.values())
    if diff > 0:
        new["savings"] += diff  # spare money goes to savings, which breaks no rule
    for k in ["shopping", "dining_out", "entertainment", "transport",
              "utilities", "insurance", "groceries", "savings", "emergency_fund"]:
        if diff >= 0:
            break
        other = {"savings": "emergency_fund", "emergency_fund": "savings"}.get(k)
        floor = (MIN_GROCERIES if k == "groceries"
                 else max(500, need - new[other]) if other else 500)
        cut = min(-diff, max(new[k] - floor, 0))
        new[k] -= cut
        diff += cut

    return [f"{k}: ₹{data[k]:,.0f} → ₹{new[k]:,.0f}" for k in CATEGORIES if new[k] != data[k]]


# ---------------- Terminal display ----------------
GREEN, RED, YELLOW, CYAN, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[1m", "\033[0m")


def snapshot(reply):
    """One line that shows the budget at a glance: total, savings rate, wants share."""
    data = extract_budget(reply)
    if data is None:
        return f"{RED}(no valid budget in the reply){RESET}"
    amt = lambda k: data[k] if k in data and is_amount(data[k]) else 0
    total = sum(v for v in data.values() if is_amount(v))
    saved = (amt("savings") + amt("emergency_fund")) / INCOME
    wants = (amt("dining_out") + amt("shopping") + amt("entertainment")) / INCOME
    return f"Total ₹{total:,.0f} of ₹{INCOME:,} · Saving {saved:.1%} · Wants {wants:.1%}"


def critique(reply):
    """The model's own list of what was wrong: the bullet lines before the JSON."""
    text = reply.split("```")[0]
    lines = [l.strip(" -*•\t") for l in text.splitlines()
             if re.match(r"\s*([-*•]|\d+[.)])\s+", l)]
    return [l if len(l) <= 90 else l[:87] + "..." for l in lines[:3]]


def show_round(i, reply, passed, failures, tokens, secs):
    title = "First draft" if i == 0 else f"Round {i} · after reflection"
    print(f"{BOLD}{title}{RESET}  {DIM}{tokens:,} tokens · {secs:.1f}s{RESET}")
    print(f"  {snapshot(reply)}")
    if failures:
        print(f"  {RED}✗ {passed}/{len(CHECKS)} rules passed{RESET}")
        for f in failures:
            print(f"    {RED}✗{RESET} {f}")
    else:
        print(f"  {GREEN}✓ {passed}/{len(CHECKS)} rules passed{RESET}")


def one_run(model, max_rounds, always_reflect, run_no, save_dir):
    print(f"{DIM}… model is drafting a budget{RESET}")
    messages = [{"role": "user", "content": TASK}]
    reply, tokens, secs = chat(model, messages)
    passed, failures = check_budget(reply)
    rounds = [{"passed": passed, "tokens": tokens, "secs": secs}]
    save(save_dir, run_no, 0, reply, failures)
    show_round(0, reply, passed, failures, tokens, secs)

    for i in range(1, max_rounds + 1):
        if not failures and not always_reflect:
            msg = "no reflection needed" if i == 1 else "fixed by reflection"
            print(f"{GREEN}■ Stop: every rule passed, {msg}{RESET}")
            break  # stop rule: budget passed every check, don't pay for reflection
        reason = "checks failed" if failures else "always-reflect is on"
        print(f"\n{YELLOW}↻ Reflecting ({reason}): the model reviews its own budget{RESET}")
        feedback = ("These checks failed:\n" + "\n".join(failures)) if failures \
            else "All checks passed."
        fix = suggest_fix(reply) if failures else []
        if fix:
            feedback += ("\nSuggested fix, with the maths already checked. Change only these "
                         "and keep every other category the same:\n" + "\n".join(fix))
            print(f"  {YELLOW}Checker's suggested fix:{RESET}")
            for line in fix:
                print(f"    {YELLOW}›{RESET} {line}")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": REFLECT.format(feedback=feedback)}]
        reply, tokens, secs = chat(model, messages)
        passed, failures = check_budget(reply)
        rounds.append({"passed": passed, "tokens": tokens, "secs": secs})
        save(save_dir, run_no, i, reply, failures)
        notes = critique(reply)
        if notes:
            print(f"  {CYAN}Model's own critique:{RESET}")
            for n in notes:
                print(f"    {CYAN}›{RESET} {n}")
        show_round(i, reply, passed, failures, tokens, secs)
    else:
        if failures:
            print(f"{RED}■ Stop: hit the cap of {max_rounds} reflection rounds{RESET}")
        elif max_rounds:
            print(f"{GREEN}■ Stop: every rule passed, fixed by reflection{RESET}")
    return rounds


def save(save_dir, run_no, round_no, reply, failures):
    """Saves the model's full reply plus what the checks found: handy for screenshots."""
    path = os.path.join(save_dir, f"finance_run{run_no}_round{round_no}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(reply.strip() + "\n\n--- checks ---\n")
        f.write("\n".join(failures) if failures else "All checks passed.")
        f.write("\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--always-reflect", action="store_true")
    args = p.parse_args()

    save_dir = "runs"
    os.makedirs(save_dir, exist_ok=True)
    n = len(CHECKS)
    all_runs = []

    for r in range(1, args.runs + 1):
        print(f"\n{BOLD}━━━ Run {r} of {args.runs} {'━' * 40}{RESET}")
        rounds = one_run(args.model, args.max_rounds, args.always_reflect, r, save_dir)
        all_runs.append(rounds)
        path = " → ".join(f"{x['passed']}/{n}" for x in rounds)
        tok = sum(x["tokens"] for x in rounds)
        ratio = tok / rounds[0]["tokens"] if rounds[0]["tokens"] else 0
        print(f"{DIM}Run {r}: rules {path} · tokens {rounds[0]['tokens']:,} → {tok:,} ({ratio:.1f}x){RESET}")

    k = len(all_runs)
    avg = lambda xs: sum(xs) / len(xs)
    draft_ok = sum(run[0]["passed"] == n for run in all_runs)
    final_ok = sum(run[-1]["passed"] == n for run in all_runs)
    d_tok = avg([run[0]["tokens"] for run in all_runs])
    f_tok = avg([sum(x["tokens"] for x in run) for run in all_runs])
    d_sec = avg([run[0]["secs"] for run in all_runs])
    f_sec = avg([sum(x["secs"] for x in run) for run in all_runs])

    mode = "always reflect" if args.always_reflect else "reflect only if checks fail"
    print(f"\n=== AI budget planner + reflection | {args.model} | {k} runs | {mode} ===")
    print(f"Income ₹{INCOME:,} | {n} rules: adds up, fixed bills kept, "
          f"save >= {MIN_SAVINGS_PCT}%, wants <= {MAX_WANTS_PCT}%")
    print(f"First draft followed every rule : {draft_ok}/{k} budgets")
    print(f"After reflection                : {final_ok}/{k} budgets")
    print(f"Avg rules passed                : {avg([r[0]['passed'] for r in all_runs]):.1f}/{n} "
          f"-> {avg([r[-1]['passed'] for r in all_runs]):.1f}/{n}")
    print(f"Avg tokens per budget           : {d_tok:,.0f} -> {f_tok:,.0f}  ({f_tok / d_tok:.1f}x)")
    print(f"Avg time per budget             : {d_sec:.1f}s -> {f_sec:.1f}s  ({f_sec / d_sec:.1f}x)")
    print(f"Every draft and fix saved in ./{save_dir}/finance_run*.txt")


if __name__ == "__main__":
    main()
