import asyncio
import json
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from time import sleep
import emoji
import grapheme
from groq import Groq
from openai import OpenAI
import pytz
import os

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import ollama
from transformers import GPT2Tokenizer

from pyrogram.raw import functions, types
from pyrogram import Client
from pyrogram.enums.parse_mode import ParseMode
from pyrogram.types.messages_and_media.message_reactions import MessageReactions
from pyrogram.raw.types import MessageEntityTextUrl, MessageEntityUrl
from pyrogram.types.messages_and_media.reaction import Reaction
from pyrogram.types.messages_and_media.message import Message
from pyrogram.types.messages_and_media.message_entity import MessageEntity
from pyrogram.enums.message_entity_type import MessageEntityType
from pyrogram.enums.chat_type import ChatType
from pyrogram.types import Chat, ForumTopicCreated

import yaml
import re
from dataclasses import dataclass, field
from typing import List, Tuple

try:
    import uvloop

    uvloop.install()
except ImportError:
    uvloop = None

url_pattern = re.compile(r"(https?://\S+)")
ALL_TOPIC = "Main topic"


@dataclass
class TopicConfig:
    name: str
    context: str = None
    ignore: bool = False


@dataclass
class ChannelConfig:
    id: int
    name: str
    filters: list[str] = field(default_factory=list)
    language: str = None
    context: str = None
    topics: dict[str, TopicConfig] = field(default_factory=dict)


# Load configuration from a YAML file
def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


# Initialize the tokenizer globally
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2", clean_up_tokenization_spaces=False
)


format_style = "Keep the response concise and structured."


def is_private_group(chat: Chat) -> bool:
    try:
        return chat.type in ["group", "supergroup"] and not chat.username
    except Exception as e:
        print(f"Error checking chat: {e}")
        return False


def make_hashtag(text: str) -> str:
    return "#" + text.lower().replace(" ", "").replace(".", "").replace(
        "-", ""
    ).replace("/", "").replace("(", "").replace(")", "").replace(",", "").replace(
        "|", ""
    )


def pick_unicore_emoji(name: str) -> str:
    em = emoji.EMOJI_ALIAS_UNICODE[":headphone:"]
    return em


def count_offsets(text: str) -> int:
    # be aware of the new line character and emojis and non-ascii characters
    return len(text.encode("utf-8")) + text.count("\n") + text.count(emoji.emojize(":"))


def format_user_friendly_date(date: datetime) -> str:
    return date.strftime("%A, %B %d, %Y")


def create_collapsible_quote(*lines, hidden=None):
    """
    Create a collapsible quote for Telegram messages using Markdown syntax.

    Args:
    *lines: Variable number of strings, each representing a visible line in the quote.
    hidden: Optional string or list of strings to be hidden (expandable part).

    Returns:
    str: Markdown formatted string for a collapsible quote.
    """
    quote = ">" + "\n>".join(lines)

    if hidden:
        if isinstance(hidden, list):
            hidden_text = "\n>".join(hidden)
        else:
            hidden_text = hidden
        quote += f"\n>{hidden_text}||"

    return quote


# Function to extract URLs from message entities
def extract_urls(entities, message_text):
    urls = []
    if not entities:
        return urls
    for entity in entities:
        if isinstance(entity, MessageEntityTextUrl):
            # Extract the URL from MessageEntityTextUrl
            url = entity.url
            urls.append(url)
        elif isinstance(entity, MessageEntityUrl) or entity.type.name == "URL":
            # Extract the URL from MessageEntityUrl
            offset = entity.offset
            length = entity.length
            url = message_text[offset : offset + length]
            urls.append(url)

    return urls


class SummarizationModel:
    def __init__(self):
        self.context = ""

    def chat(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses should implement this method")

    def set_context(self, context: str):
        self.context = context


class OllamaModel(SummarizationModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def chat(self, prompt: str) -> str:
        if self.context:
            prompt = f"{prompt}\n\n{self.context}"

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response["message"]["content"].strip()


class GroqModel(SummarizationModel):
    def __init__(self, model_name: str):
        super().__init__()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def chat(self, prompt: str) -> str:
        if self.context:
            prompt = f"{prompt}\n\n{self.context}"
        sleep(1)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            # temperature=1,
            # max_tokens=1024,
            # top_p=1,
            stream=True,
            # stop=None,
        )

        response = ""
        for chunk in completion:
            if isinstance(chunk, tuple):
                # rate limit exceeded
                sleep(1)
                continue
            c = chunk.choices
            chunk_choice = c[0]
            response += chunk_choice.delta.content or ""
        return response.strip()


class OpenAIModel(SummarizationModel):
    def __init__(self, model_name: str):
        super().__init__()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def chat(self, prompt: str) -> str:
        if self.context:
            prompt = f"{prompt}\n\n{self.context}"

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = completion.choices[0].message.content
        return content.strip() if content else ""


@dataclass
class MessageInfo:
    id: int
    text: str
    author_id: int | None
    author_name: str | None
    reactions: int
    url: str | None = None
    replies: List["MessageInfo"] = field(default_factory=list)
    reply_to_message_id: int | None = None

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    # lt
    def __lt__(self, other):
        return self.id < other.id

    @staticmethod
    def print_conversation(message: "MessageInfo", indent: int = 0):
        print(f"{'  ' * indent}{message.author_name}: {message.text}")
        for reply in message.replies:
            MessageInfo.print_conversation(reply, indent + 1)


@dataclass
class ThreadInfo:
    root_message: MessageInfo
    depth: int
    total_reactions: int
    unique_participants: set[int]

    def __str__(self):
        url = self.root_message.url or "NA"
        author = self.root_message.author_name or "NA"
        replies = len(self.root_message.replies)
        return f"🧵 Thread: {url} by {author} with {replies} replies and {self.total_reactions} reactions 👍"


@dataclass
class MsgAnalysis:
    reaction_total_count: int
    unique_reactions: int
    top_reactions: List[Tuple[str, int]]
    message: Message
    replies_count: int = 0
    msg_link: str = None
    hour_of_day: int = None
    user_course: str = None
    is_admin: bool = False


def get_chat_name_from_msg(message: Message) -> str:
    chat = message.chat

    return get_chat_name(chat)


def get_chat_name(chat: Chat) -> str:
    return chat.title or chat.username or chat.first_name


class TelegramSummarizer:
    @staticmethod
    def _resolve_config_value(config: dict, key: str, env_key: str | None = None):
        value = config.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value in (None, "", "TODO"):
            if env_key:
                env_value = os.environ.get(env_key, "").strip()
                if env_value:
                    return env_value
            return None
        return value

    @staticmethod
    def _parse_required_int(value, field_name: str, hint: str = "") -> int:
        if value is None:
            raise ValueError(
                f"Missing required '{field_name}'. {hint}".strip()
            )
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid '{field_name}': expected integer, got '{value}'. {hint}".strip()
            )

    def reload_config(self):
        self.config = load_config()
        config = self.config
        self.channels = config["channels"]

        self.session_name = self._resolve_config_value(config, "session_name") or "chatSummary"
        self.api_id = self._parse_required_int(
            self._resolve_config_value(config, "api_id", "TELEGRAM_API_ID"),
            "api_id",
            "Set api_id in config.yaml or TELEGRAM_API_ID env var.",
        )
        self.api_hash = self._resolve_config_value(
            config, "api_hash", "TELEGRAM_API_HASH"
        )
        if not self.api_hash:
            raise ValueError(
                "Missing required 'api_hash'. Set api_hash in config.yaml or TELEGRAM_API_HASH env var."
            )

        self.summary_channel_id = self._parse_required_int(
            self._resolve_config_value(config, "summary_channel_id"),
            "summary_channel_id",
            "Set summary_channel_id in config.yaml (Telegram chat/channel id).",
        )
        if any(c.get("id") == self.summary_channel_id for c in self.channels):
            print(
                "Warning: one of source channels equals summary_channel_id. "
                "The bot may summarize its own summary posts."
            )

        default_model_provider = config.get("model_provider", "ollama")
        default_model_name = config.get("model_name", "llama3.2")
        if default_model_provider == "ollama":
            self.model = OllamaModel(default_model_name)
        elif default_model_provider == "groq":
            self.model = GroqModel(default_model_name)
        elif default_model_provider == "openai":
            self.model = OpenAIModel(default_model_name)
        else:
            raise ValueError("Invalid model provider")
        self.max_length = config.get("max_length", 4000)
        self.summarization_frequency_hours = config.get("summarization_frequency", 24)
        self.output_dir = config.get("output_dir", "summaries")
        self.attachments_dir = os.path.join(self.output_dir, "attachments")
        self.urls_dir = os.path.join(self.output_dir, "urls")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.attachments_dir, exist_ok=True)
        os.makedirs(self.urls_dir, exist_ok=True)

    def __init__(self):
        self.reload_config()
        self.app = None
        self.thread_cache: dict[str, ThreadInfo] = {}

    def ensure_app(self):
        """
        Create the Pyrogram client after the asyncio loop is running.
        This avoids cross-loop issues when used with asyncio.run(...).
        """
        if self.app is None:
            self.app = Client(
                self.session_name,
                api_id=self.api_id,
                api_hash=self.api_hash,
            )

    async def get_chat_topics(self, chat_id) -> dict[str, int]:
        rv = {}

        try:
            input_channel = await self.app.resolve_peer(chat_id)

            # Get forum topics
            result = await self.app.invoke(
                functions.channels.GetForumTopics(
                    channel=input_channel,
                    offset_date=0,  # Use 0 or any date in the future to get results from the last topic
                    offset_id=0,  # ID of the last message of the last found topic (or initially 0)
                    offset_topic=0,  # ID of the last found topic (or initially 0)
                    limit=100,  # Maximum number of results to return
                    q="",  # Optional search query
                )
            )

            # Process the result
            if isinstance(result, types.messages.ForumTopics):
                topics = result.topics
                for topic in topics:
                    print(topic.title)
                    rv[topic.title] = topic.id
            else:
                print("Failed to retrieve topics.")
        except Exception as e:
            print(f"Error retrieving topics: {e}")
        return rv

    async def get_or_create_topic(self, channel: Chat, topic_name: str) -> int:
        topics: dict[str, int] = await self.get_chat_topics(channel.id)

        if topic_name in topics:
            return topics[topic_name]

        # If the topic doesn't exist, create it
        try:
            new_topic = await self.app.create_forum_topic(
                chat_id=self.summary_channel_id, title=topic_name
            )
            print(f"Created new topic: {topic_name}")
            return new_topic.id
        except Exception as e:
            print(f"Error creating topic {topic_name}: {str(e)}")
            return None

    async def fetch_dialogs(self):
        # Fetch and print available dialogs (channels and chats)
        self.ensure_app()
        async with self.app:
            tex = """
            
"""
            await self.app.send_message(chat_id="me", text=tex)
            dialogs = self.app.get_dialogs()
            chat_map = {}
            async for dialog in dialogs:
                title = dialog.chat.first_name or dialog.chat.title
                chat_id = dialog.chat.id
                chat_map[title] = chat_id
            return chat_map

    async def process_channels(self):
        # Calculate the time offset based on summarization frequency
        self.ensure_app()
        async with self.app:
            while True:
                # await self.test_formating()
                time_offset = datetime.now(pytz.utc) - timedelta(
                    hours=self.summarization_frequency_hours
                )
                for channel_info in self.channels:
                    channel_config = ChannelConfig(
                        id=channel_info["id"],
                        name=channel_info["name"],
                        filters=channel_info.get("filters", []),
                        topics={
                            topic_name: TopicConfig(
                                name=topic_name,
                                context=topic.get("context"),
                                ignore=topic.get("ignore", False),
                            )
                            for topic_name, topic in channel_info.get(
                                "topics", {}
                            ).items()
                        },
                    )
                    print(f"Processing {channel_config.name}...")
                    try:
                        await self.process_chat_history(time_offset, channel_config)
                    except Exception as e:
                        print(f"Error processing {channel_config.name}: {e}")
                        traceback.print_exc()
                print("Waiting for the next cycle...")
                await asyncio.sleep(
                    self.summarization_frequency_hours * 3600
                )  # Convert hours to seconds
                self.reload_config()

    async def test_formating(self):
        tex = f"""
📊 Daily Summary: 
"""
        # entities: list[MessageEntity] = [
        #     self.create_blockquote_entity(),
        # ]
        await self.app.send_message(
            chat_id="me",
            text=tex,  # entities=entities, parse_mode=ParseMode.MARKDOWN
        )
        # https://docs.pyrogram.org/topics/text-formatting
        # await self.app.send_message(
        #     "me",
        #     "text user mention",
        #     entities=[
        #         MessageEntity(type="mention", offset=0, length=15, user=123456789)
        #     ],
        # )
        sys.exit(0)

    def create_blockquote_entity(self, offset: int, length: int) -> MessageEntity:
        return MessageEntity(
            type=MessageEntityType.BLOCKQUOTE,
            offset=offset,
            length=length,
            expandable=True,
        )

    @staticmethod
    def generate_hourly_message_summary(msg_per_hour_of_day: dict):
        blocks = "▁▂▃▄▅▆▇█"

        def normalize(value, min_value, max_value, new_min, new_max):
            if value == 0:
                return 0
            if max_value == min_value:
                return new_min
            return int(
                ((value - min_value) / (max_value - min_value)) * (new_max - new_min)
                + new_min
            )

        if msg_per_hour_of_day:
            max_count = max(msg_per_hour_of_day.values())
            min_count = min(msg_per_hour_of_day.values())
        else:
            max_count = min_count = 0

        periods = {
            "🌅 Morning (6AM-11AM)  ": range(6, 12),
            "☀️ Afternoon (12PM-5PM)": range(12, 18),
            "🌆 Evening (6PM-11PM)  ": range(18, 24),
            "🌙 Night (12AM-5AM)    ": range(0, 6),
        }

        visualization = ""

        for period, hours in periods.items():
            visualization += f"\n{period} "
            for hour in hours:
                count = msg_per_hour_of_day.get(hour, 0)
                normalized = normalize(count, min_count, max_count, 0, len(blocks) - 1)
                visualization += blocks[normalized]

        return visualization

    def create_message_info(self, message: Message) -> MessageInfo:
        if message.empty:
            return MessageInfo(
                id=message.id,
                text="Empty message",
                author_id=None,
                author_name="Unknown",
                reactions=0,
            )
        desc = message.text
        if not desc:
            # try get media caption
            if message.media:
                desc = message.caption
                if not desc:
                    # get type of media
                    desc = f"Media: {message.media.__class__.__name__}"
            else:
                desc = "No text content"

        return MessageInfo(
            id=message.id,
            text=desc,
            author_id=message.from_user.id if message.from_user else None,
            author_name=self.get_user_str(message),
            reactions=sum(
                r.count
                for r in (message.reactions.reactions if message.reactions else [])
            ),
            url=message.link if message.link else None,
            reply_to_message_id=message.reply_to_message_id,
        )

    async def analyze_thread(
        self, message: Message, chat_history_cutoff: datetime
    ) -> ThreadInfo:
        if message.id in self.thread_cache:
            return self.thread_cache[message.id]

        depth = 0
        total_reactions = 0
        unique_participants = set()
        message_stack = []

        current_message = message
        while True:
            message_info = self.create_message_info(current_message)
            message_stack.append(message_info)

            total_reactions += message_info.reactions
            if message_info.author_id:
                unique_participants.add(message_info.author_id)

            # Check if the current message is older than 1 month from chat_history_cutoff
            if current_message.empty or current_message.date.astimezone(pytz.utc) < (
                chat_history_cutoff - timedelta(days=30)
            ).astimezone(pytz.utc):
                break

            if not current_message.reply_to_message_id:
                break

            depth += 1
            current_message = await self.load_previous_message(
                message.chat.id,
                current_message.reply_to_message_id,
            )

        # Reconstruct the thread structure
        root_message = message_stack.pop()
        while message_stack:
            reply = message_stack.pop()
            root_message.replies.append(reply)

        thread_info = ThreadInfo(
            root_message, depth, total_reactions, unique_participants
        )
        self.thread_cache[message.id] = thread_info
        return thread_info

    async def load_previous_message(self, chat_id, message_id):
        try:
            return await self.app.get_messages(
                chat_id=chat_id,
                message_ids=message_id,
            )
        except Exception as e:
            print(f"Error fetching message {message_id}: {e}")
            return None

    async def analyze_message(
        self,
        message,
        conversation_replies: dict,
        thread_roots: dict[int, ThreadInfo],
        chat_history_cutoff: datetime,
    ) -> MsgAnalysis | None:
        if not message:
            return None

        # Extract reactions
        reaction_total_count = 0
        unique_reactions = 0
        top_reactions = []
        if message.reactions:
            reaction_list: list[Reaction] = message.reactions.reactions
            reaction_total_count = sum(reaction.count for reaction in reaction_list)
            reaction_types = Counter(reaction.emoji for reaction in reaction_list)
            unique_reactions = len(reaction_types)
            top_reactions = reaction_types.most_common(3)

        reply_to_top_message_id = message.reply_to_top_message_id
        reply_to_message_id = message.reply_to_message_id
        if reply_to_message_id:
            print(f"Reply to message: {message.text}")
            # conversation_replies[reply_to_message_id].append(message.id)
            thread_info = await self.analyze_thread(message, chat_history_cutoff)
            root_id = thread_info.root_message.id

            if (
                root_id not in thread_roots
                or thread_roots[root_id].depth < thread_info.depth
            ):
                thread_roots[root_id] = thread_info

        if reply_to_top_message_id:
            print(f"Reply to top message: {message.text}")
            conversation_replies[reply_to_top_message_id].append(message.id)

        if_foward_from = message.forward_from
        if if_foward_from:
            print(f"Forwarded from: {message.text}")

        # Extract replies count
        # replies_count = message.replies.replies if message.replies else 0
        # if replies_count:
        #     print(f"Replies count: {replies_count}")

        # Extract message landing page
        msg_link = message.link if message.link else ""

        # Extract hour of the day
        hour_of_day = message.date.hour

        # Extract user course (assuming user course is a custom field in the message)
        user_course = (
            message.from_user.course
            if message.from_user and hasattr(message.from_user, "course")
            else ""
        )

        is_amdmin = message.from_user and hasattr(message.from_user, "is_admin")
        if is_amdmin:
            print(f"Admin message: {message.text}")

        has_mentions = message.entities and any(
            entity.type == "mention" for entity in message.entities
        )
        if has_mentions:
            print(f"Mentioned user: {message.text}")

        has_hashtags = message.entities and any(
            entity.type == "hashtag" for entity in message.entities
        )
        if has_hashtags:
            print(f"Hashtags: {message.text}")

        has_media = message.media
        if has_media:
            print(f"Media: {message.text}")

        msg_len = len(message.text) if message.text else 0

        if hasattr(message, "pinned"):
            is_pinned = message.pinned
            if is_pinned:
                print(f"Pinned message: {message.text}")

        # Create a message analysis object
        message_analysis = MsgAnalysis(
            reaction_total_count=reaction_total_count,
            unique_reactions=unique_reactions,
            top_reactions=top_reactions,
            # replies_count=replies_count,
            msg_link=msg_link,
            hour_of_day=hour_of_day,
            user_course=user_course,
            message=message,
            is_admin=is_amdmin,
        )

        return message_analysis

    @staticmethod
    def remove_intermediate_roots(thread_roots: dict[int, ThreadInfo]):
        # Remove intermediate roots that are part of a longer thread
        roots_to_remove = set()

        for root_id, thread_info in thread_roots.items():
            current_message: MessageInfo = thread_info.root_message

            for reply in current_message.replies:
                if reply.id in thread_roots:
                    roots_to_remove.add(reply.id)

        # Remove intermediate roots
        for root_id in roots_to_remove:
            del thread_roots[root_id]

    @staticmethod
    def prepare_thread_for_llm(
        thread: ThreadInfo, channel_name: str, context_info: str
    ) -> str:
        def format_message(message: MessageInfo, indent: int = 0) -> str:
            author = message.author_name or "Unknown User"
            reactions = f" [{message.reactions} reactions]" if message.reactions else ""
            formatted = f"{'  ' * indent}{author}: {message.text}{reactions}\n"
            for reply in message.replies:
                formatted += format_message(reply, indent + 1)
            return formatted

        formatted_conversation = format_message(thread.root_message)
        word_count = len(formatted_conversation.split())

        # Calculate desired summary length
        base_length = 50  # Minimum summary length
        max_length = 300  # Maximum summary length
        length_factor = min(word_count / 100, 1)  # Scale factor based on input size

        # Determine the relative size of the thread
        if word_count < 50:
            thread_size = "very short"
            summary_length = "very brief"
        elif word_count < 200:
            thread_size = "short"
            summary_length = "very brief"
        elif word_count < 500:
            thread_size = "medium-length"
            summary_length = "very brief"
        elif word_count < 1000:
            thread_size = "long"
            summary_length = "very brief"
        else:
            thread_size = "very long"
            summary_length = "very brief"

        # Adjust instructions based on thread size
        if thread_size == "very short":
            summary_instruction = f"Provide a {summary_length} summary of this {thread_size} thread. Focus on the main point or question."
        elif thread_size == "short":
            summary_instruction = f"Summarize this {thread_size} thread {summary_length}ly, highlighting the key points and any conclusions reached."
        else:
            summary_instruction = f"Provide a {summary_length} summary of this {thread_size} thread. Include the main topics discussed, key questions and answers, and overall sentiment or conclusions."

        prompt = f"""{summary_instruction}. Розмова українською мовою. Відповідай українською. Розмова з каналу "{channel_name}". {format_style}
    Conversation:
    {formatted_conversation}
    __
    """
        return prompt

    # Update the summarize_thread method in your class
    def summarize_thread(
        self, thread_root: ThreadInfo, channel_name: str, context_info: str
    ) -> str:
        prompt = TelegramSummarizer.prepare_thread_for_llm(
            thread_root, channel_name, context_info
        )
        try:
            return self.model.chat(prompt)
        except Exception as e:
            print(f"Error summarizing thread: {e}")
            return None

    def scrape_links(self, links: list[str]) -> dict[str, str]:
        # Scrape the content of the links and summarize them

        scraped_links = {}
        for link in links:
            scraped_links[link] = "Summary of the content"
        return scraped_links

    async def process_chat_history(
        self, chat_history_cutoff: datetime, channel_config: ChannelConfig
    ):
        msgs = defaultdict(list)
        collected_links = defaultdict(list)
        extracted_attachments = defaultdict(list)
        first_url = None
        last_url = None
        entities: list[MessageEntity] = []
        chat: Chat = None

        active_participants = defaultdict(lambda: defaultdict(list))
        total_number_of_messages = 0
        msg_per_hour_of_day = defaultdict(int)
        analysed_msgs: dict[int, List[MsgAnalysis]] = defaultdict(list)
        conversation_replies: dict[int, dict] = defaultdict(lambda: defaultdict(list))
        thread_roots: dict[int, dict[int, ThreadInfo]] = defaultdict(
            lambda: defaultdict(ThreadInfo)
        )

        chat_context = channel_config.context
        if channel_config.language:
            lang = channel_config.language
            chat_context += f"Use {lang} language for both input and output."
        if chat_context:
            self.model.set_context(chat_context)

        async for message in self.app.get_chat_history(channel_config.id, limit=3000):
            date = message.date.astimezone(pytz.utc)
            channel_name = get_chat_name_from_msg(message)

            if date < chat_history_cutoff:
                break

            total_number_of_messages += 1
            topic_id = message.topic.title if message.topic else ALL_TOPIC
            topic_config = channel_config.topics.get(topic_id)
            topic_context = None
            if topic_config:
                if topic_config.ignore:
                    continue
                # TODO: use the topic context
                topic_context = topic_config.context

            if analysis := await self.analyze_message(
                message,
                conversation_replies[topic_id],
                thread_roots[topic_id],
                chat_history_cutoff,
            ):
                analysed_msgs[topic_id].append(analysis)

            if message.from_user:
                active_participants[topic_id][message.from_user.id].append(message)

            chat = message.chat
            if not is_private_group(chat):
                if not last_url:
                    last_url = message.link
                first_url = message.link

            msg_per_hour_of_day[date.hour] += 1

            if not self.apply_filters(message, channel_config.filters):
                continue

            if links := extract_urls(message.entities, message.text):
                collected_links[topic_id].extend(links)

            if attachments := await self.extract_attachments(
                message, channel_name, date
            ):
                extracted_attachments[topic_id].extend(attachments)

            user = self.get_user_str(message)
            text_ = f"{user}: {message.text}"
            msgs[topic_id].append(text_)

        total_number_of_messages = sum(len(m) for m in msgs.values())
        if total_number_of_messages == 0:
            print(f"No messages found in {channel_name}.")
            return

        # Generate summaries and insights
        highlights = self.generate_highlights(
            channel_name,
            chat_history_cutoff,
            msgs,
            analysed_msgs,
            active_participants,
            thread_roots,
            collected_links,
            extracted_attachments,
            msg_per_hour_of_day,
            first_url,
            last_url,
        )

        # Save URLs and attachments
        # self.save_urls(collected_links, channel_name, offset)
        # self.save_attachments_info(extracted_attachments, channel_name, offset)

        # Send summary to channel
        await self.send_summary_to_channel(chat, highlights)
        print(f"Summary for {channel_name} sent to channel.")

    def generate_highlights(
        self,
        channel_name,
        offset,
        topic_msgs: dict[str, list[str]],
        analysed_msgs,
        active_participants,
        thread_roots,
        collected_links,
        extracted_attachments,
        msg_per_hour_of_day,
        first_url,
        last_url,
    ):
        topics, events = self.generate_daily_sections(channel_name, topic_msgs)
        total_messages, active_users, top_users = self.generate_daily_statistics(
            topic_msgs, active_participants
        )
        return self.render_daily_summary(
            topics, events, total_messages, active_users, top_users
        )

    @staticmethod
    def _try_parse_json(text: str) -> dict:
        if not text:
            return {}

        normalized = text.strip()
        if normalized.startswith("```"):
            normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
            normalized = re.sub(r"\s*```$", "", normalized)

        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", normalized, re.DOTALL)
        if not match:
            return {}

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _fallback_topics(text: str, limit: int = 3) -> list[str]:
        topics = []
        for raw_line in text.splitlines():
            line = raw_line.strip().lstrip("-• ").strip()
            if not line:
                continue
            if len(line) < 8:
                continue
            line = line[:120].rstrip(".")
            topics.append(line)
            if len(topics) == limit:
                break
        if not topics:
            topics = ["Обговорення поточних питань у чаті"]
        return topics

    def generate_daily_sections(
        self, channel_name: str, topic_msgs: dict[str, list[str]]
    ) -> tuple[list[str], str]:
        merged_messages = []
        many_topic = len(topic_msgs) > 1

        for topic_name, messages in topic_msgs.items():
            if many_topic:
                merged_messages.append(f"[Тема: {topic_name}]")
            merged_messages.extend(reversed(messages))

        if not merged_messages:
            return (
                ["Суттєвих тем за період не зафіксовано"],
                "Суттєвих подій за період не зафіксовано.",
            )

        base_summary = self.generate_summaries(merged_messages, channel_name, None).strip()
        if not base_summary:
            return (
                ["Суттєвих тем за період не зафіксовано"],
                "Суттєвих подій за період не зафіксовано.",
            )

        prompt = f"""
Ти аналізуєш підсумок повідомлень із Telegram-чату "{channel_name}".
Поверни тільки JSON без markdown у форматі:
{{
  "topics": ["тема 1", "тема 2", "тема 3"],
  "events": "Зв'язний опис подій у 1-2 абзацах."
}}
Вимоги:
- Мова: тільки українська.
- "topics": 2-5 коротких пунктів без нумерації та без крапки в кінці.
- "events": стислий зв'язний опис того, що відбувалось.

Текст для аналізу:
{base_summary}
"""
        try:
            parsed = self._try_parse_json(self.model.chat(prompt))
        except Exception as e:
            print(f"Error generating daily sections: {e}")
            return self._fallback_topics(base_summary), base_summary

        topics = parsed.get("topics", [])
        events = parsed.get("events", "")

        if not isinstance(topics, list):
            topics = []
        topics = [str(t).strip().rstrip(".") for t in topics if str(t).strip()]
        topics = topics[:5]
        if not topics:
            topics = self._fallback_topics(base_summary)

        if not isinstance(events, str) or not events.strip():
            events = base_summary

        return topics, events.strip()

    def generate_daily_statistics(
        self, topic_msgs: dict[str, list[str]], active_participants: dict[int, list[Message]]
    ) -> tuple[int, int, list[tuple[str, int]]]:
        total_messages = sum(len(messages) for messages in topic_msgs.values())

        user_message_counter = Counter()
        user_names = {}
        for topic_participants in active_participants.values():
            for user_id, messages in topic_participants.items():
                if not messages:
                    continue
                user_message_counter[user_id] += len(messages)
                user_names[user_id] = self.get_user_str(messages[0])

        active_users = len(user_message_counter)
        top_users = [
            (user_names.get(user_id, "Unknown User"), count)
            for user_id, count in user_message_counter.most_common(3)
        ]
        return total_messages, active_users, top_users

    def render_daily_summary(
        self,
        topics: list[str],
        events: str,
        total_messages: int,
        active_users: int,
        top_users: list[tuple[str, int]],
    ) -> str:
        top_users_str = (
            ", ".join(f"{name} ({count} пов.)" for name, count in top_users)
            if top_users
            else "немає даних"
        )

        lines = [
            "#ПідсумокДня",
            "",
            "Привіт, сусіди! Короткий огляд того, що сьогодні відбувалось у нашому чаті.",
            "",
            "🏠 Основні теми:",
        ]
        lines.extend(f"- {topic}" for topic in topics)
        lines.extend(
            [
                "",
                "📝 Що відбувалось:",
                events,
                "",
                "📊 Статистика:",
                f"- Кількість повідомлень: {total_messages}",
                f"- Активних учасників: {active_users}",
                f"- Топ-3 найактивніших дописувачів: {top_users_str}.",
            ]
        )
        return "\n".join(lines)

    def generate_group_level_insights(
        self,
        channel_name: str,
        msgs: dict[str, list[str]],
        active_participants: dict[int, list[Message]],
        msg_per_hour_of_day: dict[int, int],
        first_url: str,
        last_url: str,
    ):
        insights = f"📊 Group-Level Insights for {channel_name}:\n\n"

        total_messages = sum(len(m) for m in msgs.values())
        total_participants = sum(len(p) for p in active_participants.values())
        insights += (
            f"Total Messages: {total_messages} from {total_participants} participants\n"
        )

        insights += "\n⏰ Message Frequency:\n"
        insights += self.generate_hourly_message_summary(msg_per_hour_of_day) + "\n"

        if first_url or last_url:
            insights += "\n🔗 Group Links:\n"
            if first_url:
                insights += f"   • First post: {first_url}\n"
            if last_url:
                insights += f"   • Last post: {last_url}\n"

        return insights

    def generate_topic_summary(
        self,
        topic_name: str,
        topic_msgs: list[str],
        topic_analysed_msgs: list[MsgAnalysis],
        topic_active_participants: dict[int, list[Message]],
        topic_thread_roots: dict[int, ThreadInfo],
        topic_collected_links: list[str],
    ):

        topic_msgs.reverse()
        topic_collected_links.reverse()

        summary = f"Summary for topic '{topic_name}':\n"

        current_context = self.model.context
        self.model.set_context(
            f"{current_context}\n Current sub topic of the channel: {topic_name}"
        )

        # Generate summary text
        summary_text = self.generate_summaries(topic_msgs, topic_name, None)
        # restore the context
        self.model.set_context(current_context)
        summary += f"{summary_text}\n\n"

        # Top important messages
        # top_msg_limit = 3
        # if top_messages := self.get_scored_messages(topic_analysed_msgs[:top_msg_limit]):
        #     summary += "Top Important Messages:\n"
        #     for msg in top_messages:
        #         summary += f"• {msg['text']} (Score: {msg['importance_score']})\n"

        # Active participants
        top_participants_limit = min(5, len(topic_active_participants))
        summary += "\nActive Participants (Top 5):\n"
        sorted_participants = sorted(
            topic_active_participants.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )[:top_participants_limit]
        for user_id, messages in sorted_participants:
            user = self.get_user_str(messages[0])
            summary += f"• {user}: {len(messages)} messages\n"

        # Important threads
        if topic_thread_roots:
            top_threads_limit = 3
            summary += f"\nImportant Threads (Top {top_threads_limit}):\n"
            thread_info = self.add_threads_info(
                topic_name, topic_thread_roots, limit=top_threads_limit
            )
            summary += thread_info

        # Collected links
        if topic_collected_links:
            summary += "\n🔗 Relevant Links:\n"
            for link in topic_collected_links[:5]:  # Limit to top 5 links
                summary += f"• {link}\n"

        return summary

    def get_scored_messages(self, analysed_msgs):
        scored_messages = []
        for im in analysed_msgs:
            importance_score = im.reaction_total_count * 2 + im.unique_reactions * 3
            scored_messages.append(
                {
                    "id": im.message.id,
                    "text": im.message.text,
                    "link": im.msg_link,
                    "date": im.message.date,
                    "importance_score": importance_score,
                }
            )
        return sorted(
            scored_messages, key=lambda x: x["importance_score"], reverse=True
        )

    def add_threads_info(
        self, channel_name, thread_roots: dict[int, ThreadInfo], limit: int
    ) -> str:
        self.remove_intermediate_roots(thread_roots)  # most likely not needed

        for thread_id, thread_root in thread_roots.items():
            print(f"Thread {thread_id}:")
            MessageInfo.print_conversation(thread_root.root_message)

        # order thread_roots by depth and number of reactions and get to 10
        thread_roots = dict(
            sorted(
                thread_roots.items(),
                key=lambda item: (item[1].depth, item[1].total_reactions),
                reverse=True,
            )[:limit]
        )

        # add each thread info to highlights
        highlights = ""
        for thread_id, thread_root in thread_roots.items():
            highlights += f"\n {thread_root}\n"
            llm_summary = self.summarize_thread(
                thread_root, channel_name, context_info="thread"
            )
            highlights += f"   Summary:\n {llm_summary}\n"
        return highlights

    def apply_filters(self, message, filters):
        return True
        # Implement filtering logic based on keywords or other criteria
        if "keywords" in filters:
            message_text = message.text.lower() if message.text else ""
            for keyword in filters["keywords"]:
                if keyword.lower() in message_text:
                    return True
            return False  # Skip messages that don't contain the keywords
        return True  # No filters applied

    def get_user_str(self, message: Message) -> str:
        if message.from_user:
            name = message.from_user.first_name or ""
            if name and message.from_user.last_name:
                name += " " + message.from_user.last_name
            username = message.from_user.username or ""
            if name and username:
                return f"{name} (@{username})".strip()
            if name:
                return name.strip()
            if username:
                return f"@{username}"
            return "Unknown User"
        else:
            return "Unknown User"

    # def get_user_url(self, message):

    async def extract_attachments(self, message, channel_name, date):
        attachments = []
        return attachments
        # Define the directory path for the channel and date
        channel_dir = os.path.join(self.attachments_dir, channel_name)
        date_dir = os.path.join(channel_dir, date.strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)

        # Check for different types of attachments
        if message.photo:
            file_path = await self.download_attachment(message, date_dir, "photo")
            if file_path:
                attachments.append({"type": "photo", "path": file_path})
        if message.document:
            file_path = await self.download_attachment(
                message, date_dir, message.document.file_name
            )
            if file_path:
                attachments.append({"type": "document", "path": file_path})
        if message.video:
            file_path = await self.download_attachment(message, date_dir, "video")
            if file_path:
                attachments.append({"type": "video", "path": file_path})
        # Add more attachment types as needed

        return attachments

    async def download_attachment(self, msg, directory, file_name):
        # Download the attachment and save it to the specified directory
        try:
            name = msg.link.split("/")[-1]
            file_name = os.path.join(directory, name)
            file_path = await self.app.download_media(
                msg,
                file_name=file_name,
                # file_ref=None,
                # progress=None,
                # progress_args=None,
            )
            if file_path:
                # Move the file to the desired directory
                final_path = os.path.join(directory, os.path.basename(file_path))
                os.rename(file_path, final_path)
                print(f"Downloaded attachment to {final_path}")
                return final_path
        except Exception as e:
            print(f"Error downloading attachment: {e}")
        return None

    def chunk_text(self, lines: list[str]) -> list[list[str]]:
        chunks = []
        current_chunk = []
        current_length = 0

        for idx, line in enumerate(lines):
            tokens = tokenizer.encode(line, add_special_tokens=False)
            token_length = len(tokens)

            if token_length > self.max_length:
                # Skip lines that are too long
                print(f"Skipping line {idx + 1} as it exceeds the max_length.")
                continue

            if current_length + token_length > self.max_length:
                # Start a new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [line]
                current_length = token_length
            else:
                current_chunk.append(line)
                current_length += token_length

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def summarize_chunk(self, chunk, channel_name):
        prompt = f"""
Ти підсумовуєш повідомлення з Telegram-чату "{channel_name}".
Зроби стислий зміст українською мовою: ключові події, запити, рішення, домовленості.
Ігноруй дрібний офтоп. Відповідь без вступних фраз.
Ось повідомлення:
{chunk}
"""
        try:
            return self.model.chat(prompt)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            return None

    def generate_summaries(
        self, msgs: list[str], channel_name: str, offset: datetime
    ) -> str:
        chunks = self.chunk_text(msgs)  # TODO: fix msgs with value None; remove them
        if not chunks:
            print(f"No chunks created for {channel_name}; input lines: {len(msgs)}")
            return ""

        summaries = []
        for idx, chunk in enumerate(chunks):
            chunk_input = "\n".join(chunk)
            print(f"Summarizing chunk {idx + 1} of {len(chunks)} for {channel_name}...")
            summary = self.summarize_chunk(chunk_input, channel_name)
            if summary:
                summaries.append(summary)
            else:
                summaries.append(f"Summary of chunk {idx + 1} could not be generated.")

        # self.save_summaries(summaries, channel_name, offset)

        return self.summarize_summaries(summaries, offset)

    def get_dir_path(self, channel_name, offset):
        date_str = offset.strftime("%Y-%m-%d")
        channel_date_dir = os.path.join(self.output_dir, channel_name, date_str)
        os.makedirs(channel_date_dir, exist_ok=True)
        return channel_date_dir

    def save_summaries(self, summaries, channel_name, offset):
        output_file = os.path.join(
            self.get_dir_path(channel_name, offset), f"summary.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for summary in summaries:
                f.write(summary + "\n")
        print(f"Summaries saved to {output_file}")
        # print summaries to console
        for summary in summaries:
            print(summary)

    def save_urls(self, urls, channel_name, offset):
        if not urls:
            return
        output_file = os.path.join(self.get_dir_path(channel_name, offset), f"urls.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for url in urls:
                f.write(url + "\n")
        print(f"URLs saved to {output_file}")

    def save_attachments_info(self, attachments, channel_name, offset):
        if not attachments:
            return
        output_file = os.path.join(
            self.get_dir_path(channel_name, offset), f"attachments.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for attachment in attachments:
                f.write(f"Type: {attachment['type']}, Path: {attachment['path']}\n")
        print(f"Attachments info saved to {output_file}")

    @staticmethod
    def split_message(message, max_length=4096):
        return [message[i : i + max_length] for i in range(0, len(message), max_length)]

    # Modified send_summary_to_channel function
    async def send_summary_to_channel(self, source_chat: Chat, summary: str):

        # Get the summary channel
        summary_channel = await self.app.get_chat(self.summary_channel_id)

        source_channel_name = get_chat_name(source_chat)

        # Check if the topic exists for the source channel
        topic_id = await self.get_or_create_topic(summary_channel, source_channel_name)

        # Send the summary message to the topic
        parts = self.split_message(summary)
        for i, part in enumerate(parts):
            part = f"{part}\n\nPart {i + 1}/{len(parts)}"
            await self.app.send_message(
                chat_id=self.summary_channel_id, text=part, reply_to_message_id=topic_id
            )
            print(f"Summary for {source_channel_name} sent successfully.")

    def summarize_summaries(self, summaries: list[str], offset: datetime) -> str:
        # if summaries are too long, summarize them
        if not summaries:
            return ""
        if len(summaries) <= 1:
            return summaries[0]

        # insert offset time into first summary

        # split the string into tokens and again ask model to summarize

        prompt = (
            "Ти асистент, який стисло об'єднує підсумки повідомлень. "
            "Поверни фінальний зведений підсумок українською, без води.\n"
            f"{chr(10).join(summaries)}"
        )

        try:
            return self.model.chat(prompt)
        except Exception as e:
            print(f"Error merging chunk summaries: {e}")
            return "\n".join(summaries)

if __name__ == "__main__":
    summarizer = TelegramSummarizer()
    asyncio.run(summarizer.process_channels())
