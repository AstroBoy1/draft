"""API clients. Keys are read from keys/<name>; if the file is absent the SDK's
own environment variable applies.
"""
import os

from .paths import api_key


def openai_client():
    from openai import OpenAI
    key = api_key("gpt_key")
    if key:
        os.environ["OPENAI_API_KEY"] = key
    return OpenAI()


def gemini_client():
    from google import genai
    key = api_key("gemini_key")
    if key:
        os.environ["GEMINI_API_KEY"] = key
    return genai.Client()


def anthropic_client():
    import anthropic
    key = api_key("claude_key")
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    return anthropic.Anthropic()
