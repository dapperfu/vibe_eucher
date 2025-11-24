"""AI-powered summaries for Euchre game hands."""

import os
from typing import List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def generate_hand_summary(hand_log: List[str], api_key: Optional[str] = None) -> Optional[str]:
    """
    Generate an AI-powered verbal summary of a hand using OpenAI.

    Parameters
    ----------
    hand_log : List[str]
        List of log lines from the hand.
    api_key : Optional[str]
        OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.

    Returns
    -------
    Optional[str]
        AI-generated summary of the hand, or None if API is unavailable or fails.
    """
    if not OPENAI_AVAILABLE:
        return None

    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)

        # Combine hand log into a single string
        hand_log_text = "\n".join(hand_log)

        # Create prompt for summary
        prompt = f"""You are a commentator for a game of Euchre. Below is a detailed log of a single hand (round) from a Euchre game. 

Please provide a brief, engaging verbal summary (2-3 sentences) of what happened in this hand. Focus on:
- Who made trump and what suit was chosen
- Key plays or interesting moments
- Which team won the hand and the score impact
- Any notable strategies or decisions

Keep it conversational and entertaining, like a sports commentator.

Hand log:
{hand_log_text}

Summary:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cost-effective model
            messages=[
                {"role": "system", "content": "You are a friendly Euchre game commentator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )

        summary = response.choices[0].message.content
        return summary.strip() if summary else None

    except Exception:
        # Silently fail if API call fails
        return None


def generate_hand_summary_safe(hand_log: List[str], api_key: Optional[str] = None) -> Optional[str]:
    """
    Safely generate a hand summary, handling all errors gracefully.

    Parameters
    ----------
    hand_log : List[str]
        List of log lines from the hand.
    api_key : Optional[str]
        OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.

    Returns
    -------
    Optional[str]
        AI-generated summary of the hand, or None if unavailable.
    """
    try:
        return generate_hand_summary(hand_log, api_key)
    except Exception:
        return None

