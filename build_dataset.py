import json
import os
import random

os.makedirs("data", exist_ok=True)

# Domain-specific templates teaching proprietary business rules + strict formatting
templates = [
    # Rule 1: Net ARR definition
    ("What is the total Net ARR for {region_name}?",
     "SELECT SUM(gross_arr - discount_amount) AS net_arr FROM subscriptions WHERE subscription_status = 'ACTIVE' AND region = '{region_code}';"),
    ("Calculate Net ARR across all active {tier} accounts.",
     "SELECT SUM(gross_arr - discount_amount) AS net_arr FROM subscriptions WHERE subscription_status = 'ACTIVE' AND arr_tier = '{tier_code}';"),
    ("Show Net ARR broken down by plan type for {region_name}.",
     "SELECT plan_type, SUM(gross_arr - discount_amount) AS net_arr FROM subscriptions WHERE subscription_status = 'ACTIVE' AND region = '{region_code}' GROUP BY plan_type;"),
    ("How much Net ARR do we have in {region_name} for {tier} customers?",
     "SELECT SUM(gross_arr - discount_amount) AS net_arr FROM subscriptions WHERE subscription_status = 'ACTIVE' AND region = '{region_code}' AND arr_tier = '{tier_code}';"),

    # Rule 2: Churn Risk definition
    ("List accounts at Churn Risk in {region_name}.",
     "SELECT account_id, account_name, last_login_days, health_score FROM accounts WHERE last_login_days > 45 AND health_score < 60 AND region = '{region_code}' LIMIT 100;"),
    ("How many {tier} customers are currently at Churn Risk?",
     "SELECT COUNT(*) AS churn_risk_count FROM accounts WHERE last_login_days > 45 AND health_score < 60 AND arr_tier = '{tier_code}';"),
    ("Find all Churn Risk accounts sorted by lowest health score.",
     "SELECT account_id, account_name, health_score FROM accounts WHERE last_login_days > 45 AND health_score < 60 ORDER BY health_score ASC LIMIT 100;"),
    ("Count Churn Risk accounts per region.",
     "SELECT region, COUNT(*) AS churn_risk_count FROM accounts WHERE last_login_days > 45 AND health_score < 60 GROUP BY region;"),

    # Rule 3: APAC Tier-1 definition
    ("Show total Net ARR for APAC Tier-1.",
     "SELECT SUM(gross_arr - discount_amount) AS net_arr FROM subscriptions WHERE subscription_status = 'ACTIVE' AND region IN ('IN', 'SG', 'JP', 'AU') AND arr_tier = 'T1';"),
    ("List all APAC Tier-1 accounts that are at Churn Risk.",
     "SELECT account_id, account_name, region, health_score FROM accounts WHERE region IN ('IN', 'SG', 'JP', 'AU') AND arr_tier = 'T1' AND last_login_days > 45 AND health_score < 60 LIMIT 100;"),
    ("Count active subscriptions in APAC Tier-1.",
     "SELECT COUNT(*) AS active_subs FROM subscriptions WHERE subscription_status = 'ACTIVE' AND region IN ('IN', 'SG', 'JP', 'AU') AND arr_tier = 'T1';"),
    ("What is the average discount amount for APAC Tier-1 customers?",
     "SELECT AVG(discount_amount) AS avg_discount FROM subscriptions WHERE subscription_status = 'ACTIVE' AND region IN ('IN', 'SG', 'JP', 'AU') AND arr_tier = 'T1';"),

    # Rule 4: Non-aggregate LIMIT 100 enforcement
    ("Show all active subscriptions in {region_name}.",
     "SELECT subscription_id, account_id, plan_type, gross_arr FROM subscriptions WHERE subscription_status = 'ACTIVE' AND region = '{region_code}' LIMIT 100;"),
    ("Fetch {tier} customer accounts located in {region_name}.",
     "SELECT account_id, account_name, industry FROM accounts WHERE arr_tier = '{tier_code}' AND region = '{region_code}' LIMIT 100;"),
    ("List recent support tickets for {tier} accounts.",
     "SELECT ticket_id, account_id, priority, created_at FROM support_tickets WHERE arr_tier = '{tier_code}' ORDER BY created_at DESC LIMIT 100;"),
    ("Get all high priority tickets in {region_name}.",
     "SELECT ticket_id, account_id, subject, created_at FROM support_tickets WHERE priority = 'HIGH' AND region = '{region_code}' LIMIT 100;")
]

regions = [
    ("India", "IN"), ("Singapore", "SG"), ("Japan", "JP"), ("Australia", "AU"),
    ("North America", "NA"), ("Europe", "EU"), ("United Kingdom", "UK"),
    ("Germany", "DE"), ("Brazil", "BR"), ("Canada", "CA")
]

tiers = [
    ("Tier-1", "T1"), ("Tier-2", "T2"), ("Enterprise", "ENT"), ("Mid-Market", "MM")
]

dataset = []
for q_tmpl, sql_tmpl in templates:
    for r_name, r_code in regions:
        for t_name, t_code in tiers:
            prompt = q_tmpl.format(region_name=r_name, region_code=r_code, tier=t_name, tier_code=t_code)
            completion = sql_tmpl.format(region_name=r_name, region_code=r_code, tier=t_name, tier_code=t_code)
            # MLX chat format
            record = {
                "messages": [
                    {"role": "system", "content": "Translate the business question into raw PostgreSQL for fin_warehouse. Output SQL only."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            }
            dataset.append(record)

# Deduplicate & shuffle deterministically
unique_records = list({r["messages"][1]["content"]: r for r in dataset}.values())
random.seed(42)
random.shuffle(unique_records)

train_data = unique_records[:120]
valid_data = unique_records[120:140]
test_data = unique_records[140:160]

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

write_jsonl("data/train.jsonl", train_data)
write_jsonl("data/valid.jsonl", valid_data)
write_jsonl("data/test.jsonl", test_data)

print(f"Dataset generated! Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
