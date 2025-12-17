What is Janus?
Janus is a fully self-hosted, OpenAI-compatible API gateway for Large Language Models that I built in 2025 — months before managed solutions like ngrok AI Gateway entered early access.
It allows you to:

Route requests through a single unified endpoint (/v1/chat/completions compatible)
Automatically failover between cloud providers (Gemini, Grok, OpenAI, Anthropic-ready) and local/self-hosted models (Ollama, vLLM, LM Studio, etc.)
Work completely offline when needed (fallback to local model on private NAS)
Apply dynamic routing policies (cost, latency, accuracy, offline priority)
Serve multiple production AI products from one backend

Janus currently powers several live Telegram-based AI experiences:

Medical diagnostic assistant (Symptoma)
Mystical tarot oracle
Narrative RPG engine (Tabula Rasa style)
Telegram Stars casino with wager system

All running on a mix of Google Gemini and a local Qwen2.5 1.5B model on a QNAP NAS in Ukraine — even during power outages.
Why Janus exists
The future of AI is decentralized and resilient.
We shouldn’t depend on one cloud provider going down, rate limits, or internet blackouts.
Janus proves that you can have:

Production reliability
Hybrid cloud + local inference
Full control over keys and traffic
True offline capability

…all in a single lightweight Python service.
Key Features

OpenAI API compatible — works with official SDKs, LangChain, Vercel AI, etc. (just change base_url)
Automatic failover & smart routing via SpoilManager (key + provider rotation)
Offline-first mode — seamless fallback to local Ollama instance
Multi-product backend — one gateway serves chat, structured generation, RPG, oracle, etc.
Persistent memory via SQLite (player states, chat history, world entities)
Production casino module with Telegram Stars payments, wager requirements, auto-withdraw tickets
CORS enabled — ready for WebApps and frontend integration
Async aiohttp server — high concurrency, low footprint

Quick Start (Local)
Bashgit clone https://github.com/Hawkar-usls/janus-llm-gateway.git
cd janus-llm-gateway

# Create virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install aiohttp aiohttp-cors google-generativeai aiosqlite

# Configure your keys (create janus_keys.json)
# Example structure:
# {
#   "gemini": "AIzaSy...",
#   "openai": "sk-...",
#   "grok": "grok-..."
# }

# Run
python "janus_core (5).py"
Server will start on http://localhost:5000
Test with curl:
Bashcurl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Say hello from Janus"}]
  }'
Configuration

USE_LOCAL_QUEEN = True → forces fallback to local Ollama
LOCAL_AI_URL → your Ollama/vLLM endpoint
janus_keys.json → store API keys securely
All endpoints under /api/ for custom apps (rpg, symptoma, oracle, hrain, casino)

Philosophy
"Iane Bifrons, respiciens et prospiciens."
"Aperi viam initio, claude viam fini."
"Sit initium faustum."
Janus looks both backward (to reliable local inference) and forward (to powerful cloud models).
It opens the path at the beginning and closes it at the end.
May the beginning be fortunate.
Status

Core gateway: Stable, production-used
Casino module: Live with real payments
Offline resilience: Battle-tested in real blackouts
Open-sourcing: Ongoing cleanup for public release

Author
Alexander Agapov 
Built in Ukraine, 2025 — mostly while recovering from illness, using AI as primary development partner.
This project is proof that the future of building is already here: human vision + AI execution.
License
MIT License — use, modify, deploy, fork freely.
