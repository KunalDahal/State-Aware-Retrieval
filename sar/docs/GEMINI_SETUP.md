# Gemini LLM setup

The project can run the MIGRES language-model calls through Gemini using Google's OpenAI-compatible API endpoint. The repository includes a ready-to-use configuration overlay at `configs/gemini.yaml`.

## 1. Install the runtime dependencies

```bash
pip install -r requirements.txt
```

The Gemini overlay uses the project's OpenAI-compatible client path, so the `openai` package is required. `google-genai` is also included in the full requirements for users who want the native Gemini client path.

## 2. Configure the API key and endpoint

Copy `.env.example` to `.env` and set:

```text
GEMINI_API_KEY=your_gemini_api_key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

Do not commit `.env` or the API key.

## 3. Run with the Gemini configuration

Layer the Gemini override after the main configuration:

```bash
python scripts/run_single.py --config configs/default.yaml configs/gemini.yaml
```

For batch or notebook runs, use the same configuration layering approach supported by the project.

## 4. Model

The supplied configuration uses:

```text
gemini-3.5-flash-lite
```

This is a stable Gemini API model. Google currently lists Gemini 3.5 Flash-Lite as a high-throughput, cost-efficient model and recommends the current stable Gemini 3.5 Flash-Lite or Gemini 3.8 Flash models for new projects. Check Google's model documentation before changing the model ID because model availability and lifecycle can change.

The model ID is kept in `configs/gemini.yaml`, rather than hard-coded into the Python source, so it can be changed without modifying the implementation.
