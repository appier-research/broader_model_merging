"""Bank-77 intent classification.

Customer-service query -> one of 77 intent categories. Ported from
model-merge-transfer/llm_evals/bank77.py; ``return_datasets``/``verify_correctness``
became ``load_split``/``score``, and ``sft_target`` was added for the GD strategies.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from datasets import load_dataset

# in_domain is only 1000 examples across 77 classes (min 4/class, median 13/class),
# but valid_fraction stays at the default 10% (not raised): weight_gd/subspace_gd
# train on the valid pool, so a larger valid share would quietly hand GD-based
# strategies more tuning data than coeff_search needs, biasing the comparison in
# their favor. Keeping the 1:9 valid:test ratio (matching vision_exp's convention)
# means per-class budgets past ~4 degenerate for the sparsest classes -- an
# accepted limitation of this task's data volume, not fixed by resizing the split.
# out_domain (500 examples) can't substitute for either half: every row in it is
# labeled -1 (an out-of-scope catch-all, not per-intent labels).
EVAL_SPLIT_SPEC = {"pool": "in_domain"}

# score() only looks for "<number>. <category>" -- Qwen3's default chat
# template invites the model to open a <think>...</think> block first, which
# has no bearing on this task's answer and would (a) blow past MAX_NEW_TOKENS
# before an answer ever appears and (b) make merged/base-model comparisons
# unfair (one state "thinks" more than another by chance, not by ability).
ENABLE_THINKING = False

# gold "<label>. <label_text>" completions top out at 11 tokens; actual
# base-model generations with ENABLE_THINKING=False above topped out at 10
# tokens over a 20-example sample. 16 leaves headroom without paying for a
# ~256-token decode on every generation call.
MAX_NEW_TOKENS = 16

INSTRUCTION_PROMPT = (
    "Act as a customer service representative and classify the customer query based on the intent.\n\n"
    "        All possible categories for you to choose from are as follows (one category per line, in the format of <number>. <category>):\n"
    "        41. lost_or_stolen_card\n56. top_up_by_bank_transfer_charge\n17. card_payment_wrong_exchange_rate\n21. change_pin\n"
    "22. compromised_card\n38. get_physical_card\n30. edit_personal_details\n60. top_up_limits\n67. transfer_timing\n49. pin_blocked\n"
    "16. card_payment_not_recognised\n20. cash_withdrawal_not_recognised\n33. exchange_via_app\n2. apple_pay_or_google_pay\n"
    "74. why_verify_identity\n76. wrong_exchange_rate_for_cash_withdrawal\n69. verify_my_identity\n0. activate_my_card\n"
    "61. top_up_reverted\n27. declined_transfer\n47. pending_top_up\n28. direct_debit_payment_not_recognised\n"
    "53. reverted_card_payment?\n50. receiving_money\n3. atm_support\n11. card_arrival\n39. getting_spare_card\n"
    "15. card_payment_fee_charged\n43. order_physical_card\n52. request_refund\n55. terminate_account\n1. age_limit\n"
    "25. declined_card_payment\n24. country_support\n62. topping_up_by_card\n34. extra_charge_on_statement\n18. card_swallowed\n"
    "35. failed_transfer\n14. card_not_working\n31. exchange_charge\n12. card_delivery_estimate\n19. cash_withdrawal_charge\n"
    "10. card_acceptance\n44. passcode_forgotten\n4. automatic_top_up\n46. pending_cash_withdrawal\n59. top_up_failed\n"
    "48. pending_transfer\n51. Refund_not_showing_up\n6. balance_not_updated_after_cheque_or_cash_deposit\n"
    "29. disposable_card_limits\n66. transfer_not_received_by_recipient\n5. balance_not_updated_after_bank_transfer\n"
    "65. transfer_into_account\n57. top_up_by_card_charge\n68. unable_to_verify_identity\n42. lost_or_stolen_phone\n"
    "36. fiat_currency_support\n45. pending_card_payment\n70. verify_source_of_funds\n72. virtual_card_not_working\n"
    "75. wrong_amount_of_cash_received\n64. transfer_fee_charged\n9. card_about_to_expire\n7. beneficiary_not_allowed\n"
    "71. verify_top_up\n23. contactless_not_working\n73. visa_or_mastercard\n13. card_linking\n"
    "54. supported_cards_and_currencies\n58. top_up_by_cash_or_cheque\n32. exchange_rate\n26. declined_cash_withdrawal\n"
    "63. transaction_charged_twice\n8. cancel_transfer\n37. get_disposable_virtual_card\n40. getting_virtual_card\n"
    "77. none_of_the_above\n\n        CUSTOMER QUERY:\n%s\n"
    "        Now provide the classification for the query in the following format: <number>. <category>"
)


def load_split(hf_split: str) -> List[Dict]:
    ds = load_dataset("appier-ai-research/bank-77", split=hf_split)
    return [
        {
            "prompt": INSTRUCTION_PROMPT % row["text"],
            "label": row["label"],
            "label_text": row["label_text"],
        }
        for row in ds
    ]


def format_example(row: Dict):
    return row["prompt"]


def score(response_str: str, row: Dict) -> bool:
    pattern = r"^\s*(\d+)\.\s*(.+)$"
    for line in response_str.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            numeric = int(match.group(1))
            category = match.group(2).strip()
            if numeric == int(row["label"]):
                return True
            if category.lower().strip() == row["label_text"].lower().strip():
                return True
    return False


def sft_target(row: Dict) -> Optional[str]:
    return f"{row['label']}. {row['label_text']}"
