"""Allowlisted Discord gateway moderation. Run: python -m hinglishguard.discord_bot."""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path

from .runtime import investigator
from .community_policy import review as community_review, warning

LOG = logging.getLogger('hinglishguard.discord')
# Revision cache for edit/delete tracking. Bounded so a long-running bot does not grow forever;
# evicting an old entry only means a very old message edit is investigated again.
MAX_TRACKED_MESSAGES = 10_000


def revision(text, parent):
    return hashlib.sha256(json.dumps([text, parent], ensure_ascii=False).encode()).hexdigest()


def deletion_allowed(result, allow_community_llm=False):
    """Delete only on a complete, evidenced FLAG.

    A FLAG that came solely from the community LLM rule is not benchmark-validated,
    so it deletes only when the operator explicitly opts in.
    """
    if result.get('community_policy') and not allow_community_llm:
        return False
    return (result.get('recommendation') == 'FLAG' and not result.get('error_status')
            and bool(result.get('evidence')) and bool(result.get('run_id')))


class RevisionCache(OrderedDict):
    """Least-recently-updated mapping of message id -> last investigated revision."""

    def __init__(self, limit=MAX_TRACKED_MESSAGES):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.limit:
            self.popitem(last=False)


def load_token(root):
    """Prefer the DISCORD_TOKEN environment variable; fall back to the git-ignored token file."""
    token = os.environ.get('DISCORD_TOKEN', '').strip()
    source = 'DISCORD_TOKEN environment variable'
    if not token:
        path = root / 'config/discord.token'
        if not path.exists():
            raise SystemExit('Set the DISCORD_TOKEN environment variable (preferred) or create config/discord.token.')
        token = path.read_text(encoding='utf-8-sig').strip()
        source = 'config/discord.token'
    if not token or any(c.isspace() for c in token):
        raise SystemExit(f'Token from {source} is empty or contains whitespace; save only the bot token.')
    return token


def build_client(root, config, engine):
    import discord

    allowed = {int(g): {int(c) for c in channels} for g, channels in config['guilds'].items()}
    if not allowed or any(not channels for channels in allowed.values()):
        raise ValueError('An explicit server/channel allowlist is required')
    allow_community_llm = bool(config.get('community_llm_can_delete', False))
    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    intents.message_content = True

    class Moderator(discord.Client):
        def __init__(self):
            super().__init__(intents=intents)
            self.queue = asyncio.Queue(maxsize=int(config.get('max_queue', 500)))
            self.latest = RevisionCache()
            # Writes run in a worker thread (see log) so SQLite never blocks the gateway event loop.
            self.db_lock = threading.Lock()
            self.db = sqlite3.connect(root / 'artifacts/discord.sqlite', check_same_thread=False)
            self.db.execute('CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, message_id TEXT, revision TEXT, status TEXT, result TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
            self.db.commit()

        def audit(self, mid, rev, status, result=None):
            with self.db_lock:
                self.db.execute('INSERT INTO audit(message_id,revision,status,result) VALUES(?,?,?,?)',
                                (str(mid), rev, status, json.dumps(result, ensure_ascii=False)))
                self.db.commit()
            LOG.info('message=%s status=%s', mid, status)

        async def log(self, mid, rev, status, result=None):
            await asyncio.to_thread(self.audit, mid, rev, status, result)

        async def setup_hook(self):
            self.worker_task = asyncio.create_task(self.worker())

        async def close(self):
            if hasattr(self, 'worker_task'):
                self.worker_task.cancel()
            await super().close()
            with self.db_lock:
                self.db.close()

        async def on_ready(self):
            LOG.info('Connected. Monitoring %s configured channel(s).', sum(map(len, allowed.values())))
            if config.get('community_profanity_rule', False) and not allow_community_llm:
                LOG.info('Community LLM rule is advisory: its FLAGs go to human review (community_llm_can_delete=false).')

        def permitted(self, message):
            return (message.guild is not None and message.channel.id in allowed.get(message.guild.id, set())
                    and not message.author.bot and not message.webhook_id
                    and message.type in (discord.MessageType.default, discord.MessageType.reply))

        async def snapshot(self, message):
            parent = None
            ref = message.reference
            if ref and ref.message_id:
                if ref.channel_id != message.channel.id:
                    raise ValueError('Parent context is outside monitored conversation')
                parent_message = await message.channel.fetch_message(ref.message_id)
                parent = parent_message.content
                if not parent:
                    raise ValueError('Parent has no supported text context')
            if not message.content.strip() or len(message.content) > 2000 or (parent and len(parent) > 2000):
                raise ValueError('Unsupported or oversized text')
            return parent, revision(message.content, parent)

        async def submit(self, message):
            if not self.permitted(message):
                return
            try:
                parent, rev = await self.snapshot(message)
                if self.latest.get(message.id) == rev:
                    return
                self.latest[message.id] = rev
                self.queue.put_nowait((message, parent, rev))
                await self.log(message.id, rev, 'queued')
            except asyncio.QueueFull:
                LOG.warning('Queue full; message=%s sent to human review', message.id)
                await self.log(message.id, '', 'HUMAN_REVIEW_queue_full')
            except Exception:
                LOG.exception('Could not snapshot message=%s', message.id)
                self.latest[message.id] = None
                await self.log(message.id, '', 'HUMAN_REVIEW_context_or_input_unavailable')

        async def on_message(self, message):
            await self.submit(message)

        async def on_raw_message_edit(self, payload):
            if payload.channel_id not in allowed.get(payload.guild_id, set()):
                return
            try:
                channel = self.get_channel(payload.channel_id)
                if channel:
                    await self.submit(await channel.fetch_message(payload.message_id))
            except Exception:
                LOG.exception('Could not fetch edited message=%s', payload.message_id)
                self.latest[payload.message_id] = None
                await self.log(payload.message_id, '', 'HUMAN_REVIEW_edit_fetch_failed')

        async def on_raw_message_delete(self, payload):
            self.latest.pop(payload.message_id, None)

        async def worker(self):
            while True:
                message, parent, rev = await self.queue.get()
                try:
                    if self.latest.get(message.id) != rev:
                        continue
                    result = await asyncio.to_thread(engine.run, {
                        'id': f'discord:{message.guild.id}:{message.channel.id}:{message.id}:{rev}',
                        'text': message.content,
                        'parent_id': str(message.reference.message_id) if parent else None,
                        'parent_text': parent,
                    })
                    if config.get('community_profanity_rule', False) and result['recommendation'] == 'HUMAN_REVIEW':
                        try:
                            result = await asyncio.to_thread(community_review, message.content, parent, result)
                        except Exception:
                            LOG.exception('Community check failed for message=%s', message.id)
                            await self.log(message.id, rev, 'HUMAN_REVIEW_community_check_failed')
                    await self.log(message.id, rev, result['recommendation'], result)
                    if result.get('community_policy') and not allow_community_llm:
                        await self.log(message.id, rev, 'HUMAN_REVIEW_community_llm_suggested_removal', result)
                        continue
                    if not deletion_allowed(result, allow_community_llm) or self.latest.get(message.id) != rev:
                        continue
                    current = await message.channel.fetch_message(message.id)
                    _, current_rev = await self.snapshot(current)
                    if current_rev != rev or not self.permitted(current):
                        await self.log(message.id, rev, 'HUMAN_REVIEW_message_changed')
                        await self.submit(current)
                        continue
                    if not current.channel.permissions_for(current.guild.me).manage_messages:
                        await self.log(message.id, rev, 'HUMAN_REVIEW_missing_permission')
                        continue
                    await current.delete()
                    await self.log(message.id, rev, 'deleted')
                    if config.get('send_warnings', False):
                        try:
                            notice = await asyncio.to_thread(warning, result)
                            await current.channel.send(notice, allowed_mentions=discord.AllowedMentions.none())
                            await self.log(message.id, rev, 'warning_sent')
                        except Exception:
                            LOG.exception('Warning could not be sent for message=%s', message.id)
                            await self.log(message.id, rev, 'warning_failed_check_send_messages_permission')
                except discord.NotFound:
                    await self.log(message.id, rev, 'message_already_absent')
                except Exception:
                    LOG.exception('Processing failed for message=%s', message.id)
                    await self.log(message.id, rev, 'HUMAN_REVIEW_processing_or_action_failed')
                finally:
                    self.queue.task_done()

    return Moderator()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / 'config/discord.local.json').read_text(encoding='utf-8-sig'))
    token = load_token(root)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    engine = investigator(root, backend=config.get('backend', 'transformer'))
    if len(engine.predictors) != 2:
        raise SystemExit('Both trained classifiers are required.')
    client = build_client(root, config, engine)
    client.run(token, log_handler=None)


if __name__ == '__main__':
    main()
