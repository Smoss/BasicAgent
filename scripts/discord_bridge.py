import os
import asyncio
from typing import Optional, Set

import discord
from dotenv import load_dotenv
import fire  # type: ignore
import yaml
from agent.character_agent import SimpleCharacterAgent


load_dotenv()


class DiscordAgentClient(discord.Client):
    def __init__(
        self,
        *,
        intents: discord.Intents,
        model: str = "gpt-oss:20b",
        mention_only: bool = True,
        allowed_channels: Optional[Set[int]] = None,
        context_window: int = 16384,
        config_path: str,
        max_voice_lines: int = 10,
    ):
        super().__init__(intents=intents)
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        character_name = config["character_name"]
        system_prompt_path = config["system_prompt_path"]
        voice_lines_path = config["voice_lines_path"]
        fandom_wiki = config["fandom_wiki"]
        self.agent = SimpleCharacterAgent(
            model=model,
            context_window=context_window,
            system_prompt_path=system_prompt_path,
            voice_lines_path=voice_lines_path,
            max_voice_lines=max_voice_lines,
            fandom_wiki=fandom_wiki,
            character_name=character_name,
        )
        self.graph = self.agent.build_graph()
        self.mention_only = mention_only
        self.allowed_channels = allowed_channels or set()

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")

    async def _create_reply(self, message: discord.Message):
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
                            final.content
                            if hasattr(final, "content")
                            else str(final)
                        )
                    except Exception:
                        final_text = str(final)
            # Remove think tokens, <think> and </think>, include newlines
            think_start = final_text.find("<think>")
            think_end = final_text.find("</think>")
            final_text = (
                final_text[:think_start] + final_text[think_end + 8 :]
            ).strip()

            return final_text or "(no response)"

        reply_text: str = await loop.run_in_executor(None, run_graph_sync, content)

        # Discord message limit safety
        if len(reply_text) <= 1900:
            await message.reply(reply_text)
        else:
            # Split long messages along new lines
            lines = reply_text.split('\n')
            chunks = []
            current_chunk = ""
            
            for line in lines:
                # If adding this line would exceed the limit, start a new chunk
                if len(current_chunk) + len(line) + 1 > 1900:  # +1 for newline
                    if current_chunk:
                        chunks.append(current_chunk.rstrip())
                        current_chunk = line
                    else:
                        # If a single line is too long, split it by character count
                        chunks.append(line[:1900])
                        current_chunk = line[1900:]
                else:
                    current_chunk += line + '\n'
            
            # Add the last chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.rstrip())
            
            for i, chunk in enumerate(chunks):
                await message.reply(chunk, mention_author=False)

    async def on_message(self, message: discord.Message):
        print(f"Message received: {message.content}")
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
            print(f"Message mentions: {message.mentions}")
            print(f"User name: {self.user.name}")
            print(f"Role mentions: {message.role_mentions}")
            # Check if the user is mentioned by role
            should_reply = self.user.name.lower() in {
                role.name.lower() for role in message.role_mentions
            }
        else:
            should_reply = True
        print(f"Should reply: {should_reply}")
        if not should_reply:
            return

        async with message.channel.typing():
            await self._create_reply(message)


class DiscordBridge:
    def run(
        self,
        *,
        token: Optional[str] = None,
        model: Optional[str] = None,
        mention_only: bool = True,
        allowed_channels: Optional[str] = None,
        context_window: int = 16384,
        max_voice_lines: int = 3,
        config_path: str = "personas/democracy_officer/config.yaml",
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
            model=model,
            mention_only=mention_only,
            allowed_channels=allowed_set,
            context_window=context_window,
            max_voice_lines=max_voice_lines,
            config_path=config_path,
        )
        client.run(token_value)


if __name__ == "__main__":
    fire.Fire(DiscordBridge)
