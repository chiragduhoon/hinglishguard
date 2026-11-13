"""Single source of runtime settings shared by the app, agent, CLI and Discord bot.

Values can be overridden with environment variables, but the Ollama endpoint is
always restricted to loopback so the agent can never be pointed at a remote host.
"""
import os

LOOPBACK_OLLAMA_HOSTS = ('http://127.0.0.1:11434', 'http://localhost:11434')

AGENT_MODEL = os.environ.get('HG_AGENT_MODEL', 'qwen3:4b-instruct')
OLLAMA_HOST = os.environ.get('HG_OLLAMA_HOST', LOOPBACK_OLLAMA_HOSTS[0]).rstrip('/')
if OLLAMA_HOST not in LOOPBACK_OLLAMA_HOSTS:
    raise ValueError('HG_OLLAMA_HOST must be a loopback Ollama endpoint: ' + ', '.join(LOOPBACK_OLLAMA_HOSTS))

OLLAMA_CHAT_URL = OLLAMA_HOST + '/api/chat'
OLLAMA_TAGS_URL = OLLAMA_HOST + '/api/tags'
