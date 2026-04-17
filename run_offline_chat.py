"""
HealthBot – Offline chat (no Ollama, no external APIs).
Run: python run_offline_chat.py
Uses only trained models + knowledge base. Fast, local, no network required.
"""
from __future__ import annotations

import sys

from app.nlp.pipeline import get_nlp_pipeline
from app.ml.pipeline import get_ml_pipeline
from app.dialog.manager import DialogueManager

SESSION_ID = "offline_cli"


def main() -> None:
    nlp = get_nlp_pipeline()
    ml = get_ml_pipeline()
    dialog = DialogueManager()

    print("HealthBot (Offline) – Medical info from trained models & knowledge base.")
    print("No Ollama or external APIs. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        nlp_out = nlp.process(user_input)
        ml_out = ml.run(user_input, nlp_out["tokens"], nlp_out["embedding"])

        action = dialog.next_action(
            SESSION_ID,
            ml_out["intent"],
            ml_out["symptoms"],
            ml_out["top3_diseases"],
            ml_out["is_emergency"],
            lang=nlp_out["lang"],
            user_message=user_input,
        )

        response = dialog.build_response(
            SESSION_ID,
            action,
            ml_out["intent"],
            ml_out["symptoms"],
            ml_out["top3_diseases"],
            ml_out["is_emergency"],
            lang=nlp_out["lang"],
            use_ollama_phrasing=False,
            user_message=user_input,
        )

        print("HealthBot:", response, "\n")


if __name__ == "__main__":
    main()
    sys.exit(0)
