import logging
import os
import asyncio
from typing import Optional, Set

import discord
from dotenv import load_dotenv
import fire  # type: ignore
from agent.character_agent import SimpleCharacterAgent
from utils.persona_config import load_persona_config
from config import DISCORD_MESSAGE_LIMIT
from utils.text import chunk_message


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class DiscordAgentClient(discord.Client):
    def __init__(
        self,
        *,
        intents: discord.Intents,
        context_model: str = "qwen3:14b",
        chat_model: str = "gemma3:12b",
        mention_only: bool = True,
        allowed_channels: Optional[Set[int]] = None,
        context_window: int = 16384,
        config_path: str,
        max_voice_lines: int = 10,
        debug: bool = False,
    ):
        super().__init__(intents=intents)
        persona_config = load_persona_config(config_path)
        self.agent = SimpleCharacterAgent(
            context_model=context_model,
            chat_model=chat_model,
            context_window=context_window,
            persona_config=persona_config,
            max_voice_lines=max_voice_lines,
            debug=debug,
        )
        self.graph = self.agent.build_graph()
        self.mention_only = mention_only
        self.allowed_channels = allowed_channels or set()

    async def on_ready(self):
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)
        assert self.user is not None

    async def _create_reply(self, message: discord.Message):
        assert self.user is not None
        content = message.content
        # Remove mention from content for cleaner prompts
        for mention in message.mentions:
            if mention.id == self.user.id:
                content = (
                    content.replace(f"<@{mention.id}>", "")
                    .replace(f"<@!{mention.id}>", "")
                    .strip()
                )

        thread_id = str(message.channel.id)
        # Run the graph synchronously in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        def run_graph_sync(user_input: str):
            # Stream last message only; collect final output text
            final_text = None
            events = self.graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="values",
            )
            for event in events:
                if messages := event.get("messages"):
                    final = messages[-1]
                    try:
                        final_text = (
                            final.content if hasattr(final, "content") else str(final)
                        )
                    except Exception:
                        final_text = str(final)
            # Remove think tokens, <think> and </think>, include newlines
            if final_text and "</think>" in final_text:
                think_start = final_text.find("<think>")
                think_end = final_text.find("</think>")
                final_text = (
                    final_text[:think_start] + final_text[think_end + 8 :]
                ).strip()

            return final_text or "(no response)"

        reply_text: str = await loop.run_in_executor(None, run_graph_sync, content)

        # Discord message limit safety
        chunks = chunk_message(reply_text, DISCORD_MESSAGE_LIMIT)
        for i, chunk in enumerate(chunks):
            await message.reply(chunk, mention_author=(i == 0))

    async def on_message(self, message: discord.Message):
        assert self.user is not None
        logger.debug("Message received: %s", message.content)
        # Ignore messages from ourselves
        if message.author.id == self.user.id:
            return

        # Channel allowlist
        if self.allowed_channels and message.channel.id not in self.allowed_channels:
            return

        # Decide whether to reply
        should_reply = False
        if isinstance(message.channel, discord.DMChannel):
            should_reply = True
        elif self.mention_only:
            logger.debug("Message mentions: %s", message.mentions)
            logger.debug("User name: %s", self.user.name)
            logger.debug("Role mentions: %s", message.role_mentions)
            # Check if the user is mentioned by role
            should_reply = self.user.name.lower() in {
                role.name.lower() for role in message.role_mentions
            }
        else:
            should_reply = True
        logger.debug("Should reply: %s", should_reply)
        if not should_reply:
            return

        async with message.channel.typing():
            await self._create_reply(message)


class DiscordBridge:
    def run(
        self,
        *,
        token: Optional[str] = None,
        context_model: str = "qwen3:14b",
        chat_model: str = "gemma3:12b",
        mention_only: bool = True,
        allowed_channels: Optional[str] = None,
        context_window: int = 16384,
        max_voice_lines: int = 3,
        config_path: str = "personas/democracy_officer/config.yaml",
        debug: bool = False,
    ) -> None:
        """
        Run the Discord bot.

        Args:
            token: Discord bot token. Falls back to DISCORD_BOT_TOKEN env var if not provided.
            model: Model name. Falls back to MODEL env var or 'gpt-oss:20b'.
            mention_only: If True, only reply when mentioned (except DMs if enabled).
            respond_in_dms: If True, reply to direct messages.
            allowed_channels: Comma-separated list of channel IDs to allow. If omitted, all channels are allowed.
        """
        token_value = token or os.environ.get("DISCORD_BOT_TOKEN")
        if not token_value:
            raise RuntimeError(
                "DISCORD_BOT_TOKEN not set in environment or passed as --token"
            )

        intents = discord.Intents.default()
        intents.message_content = True

        allowed_set: Optional[Set[int]] = None
        if allowed_channels:
            try:
                allowed_set = {
                    int(cid.strip())
                    for cid in allowed_channels.split(",")
                    if cid.strip()
                }
            except ValueError:
                raise RuntimeError(
                    "allowed_channels must be a comma-separated list of numeric IDs"
                )

        client = DiscordAgentClient(
            intents=intents,
            context_model=context_model,
            chat_model=chat_model,
            mention_only=mention_only,
            allowed_channels=allowed_set,
            context_window=context_window,
            max_voice_lines=max_voice_lines,
            config_path=config_path,
            debug=debug,
        )
        client.run(token_value)


if __name__ == "__main__":
    fire.Fire(DiscordBridge)
