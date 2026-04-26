from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


INSTITUTES: Dict[str, str] = {
    "ikni": "ІКНІ",
    "ikta": "ІКТА",
    "iard": "ІАРД",
    "igdg": "ІГДГ",
    "ihsn": "ІГСН",
    "iadu": "ІАДУ",
    "inem": "ІНЕМ",
    "ibib": "ІБІБ",
}


def build_examples() -> List[dict]:
    examples: List[dict] = []
    rector_answer = "Шаховська Юлія Володимирівна"
    rector_queries = [
        "Хто ректор Львівської політехніки?",
        "Назвіть ректора університету Львівська політехніка.",
        "Хто є чинним ректором НУ Львівська політехніка?",
        "Хто очолює університет Львівська політехніка як ректор?",
        "Підкажіть, хто ректор у Львівській політехніці.",
        "Прізвище та ім'я ректора Львівської політехніки.",
        "Хто ректор ЛП?",
        "Ректор університету Львівська політехніка хто?",
    ]
    for query in rector_queries:
        examples.append(
            {
                "query": query,
                "answer": rector_answer,
                "must_contain_text": "ректор",
                "must_contain_type": "lpnu",
            }
        )

    for code, short_name in INSTITUTES.items():
        base_url = f"lpnu.ua/{code}"
        questions = [
            f"Хто директор {short_name}?",
            f"Хто є директором інституту {short_name}?",
            f"Назвіть директора {short_name} у Львівській політехніці.",
            f"Хто очолює {short_name} як директор?",
        ]
        for query in questions:
            examples.append(
                {
                    "query": query,
                    "answer_keywords": ["директор", short_name.lower()],
                    "must_contain_url": base_url,
                    "must_contain_text": "директор",
                    "must_contain_type": "lpnu",
                }
            )

    return examples


def main() -> None:
    out_path = Path("eval_set_structural.jsonl")
    rows = build_examples()
    out_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()
