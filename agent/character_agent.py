import traceback
from langgraph.graph import START, StateGraph
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.state import State
from tools.tools import save_tool, search_tool, wiki_tool
from tools.memory import memory_add, memory_search
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt import tools_condition

# from typing import Optional
from langchain_core.messages import SystemMessage
import random
from tools.voice_rag import build_voice_retriever, select_voice_lines
from tools.lore_db import build_lore_retriever


class SimpleCharacterAgent:
    model: str

    def __init__(
        self,
        *,
        model: str = "gpt-oss:20b",
        context_window: int = 8192,
        debug: bool = False,
        system_prompt_path: str,
        voice_lines_path: str,
        max_voice_lines: int = 6,
        lore_search_k: int = 13,
        fandom_wiki: str,
        character_name: str,
    ):
        print(f"Initializing agent with model: {model}")
        self.model = model
        self.context_window = context_window
        self.debug = debug
        self.system_prompt_text: str
        self.voice_lines: list[str] = []
        self.max_voice_lines = max_voice_lines
        self.lore_search_k = lore_search_k
        self.voice_retriever = None
        self.fandom_wiki = fandom_wiki
        self.character_name = character_name
        # Pre-build lore retriever filtered to this character and wiki
        self.lore_retriever = build_lore_retriever(
            collection_name=f"{self.character_name}_lore",
            where={"character": self.character_name},
            search_k=self.lore_search_k,
        )

        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt_text = f.read().strip()
        if voice_lines_path:
            with open(voice_lines_path, "r", encoding="utf-8") as f:
                self.voice_lines = [ln.strip() for ln in f if ln.strip()]

            self.voice_retriever = build_voice_retriever(
                voice_lines_path, self.character_name
            )

    def build_graph(self) -> CompiledStateGraph:
        # Core tools plus Discord toolkit (read/send messages)
        tools = [save_tool, wiki_tool, search_tool, memory_add, memory_search]
        llm_raw = ChatOllama(
            model=self.model,
            temperature=0.6,
            num_gpu=-1,
            num_ctx=self.context_window,
            keep_alive=True,
        )
        llm = llm_raw.bind_tools(tools)

        def _style_message(user_text: str | None) -> SystemMessage | None:
            picks: list[str] = []
            # Prefer semantic selection if available
            if self.voice_retriever:
                try:
                    picks = [d.page_content for d in self.voice_retriever.invoke(user_text)]
                except Exception as e:
                    picks = []
                    print(f"Warning: Failed to select voice lines from {e}")
            # Fallback to deterministic sampling for stability and variety
            if not picks and self.voice_lines:
                seed = hash(user_text or "") & 0xFFFFFFFF
                rng = random.Random(seed)
                k = min(self.max_voice_lines, len(self.voice_lines))
                picks = rng.sample(self.voice_lines, k=k)
            if not picks:
                return None
            print(f"Selected {len(picks)} voice lines")
            print(f"Voice lines: {'\n'.join(picks)}")
            content = "Style exemplars (tone and cadence):\n- " + "\n- ".join(picks)
            return SystemMessage(content=content)
        
        def _lore_message(user_text: str) -> SystemMessage | None:
            try:
                docs = self.lore_retriever.invoke(user_text) or []
                if docs:
                    lines: list[str] = []
                    for d in docs:
                        title = (getattr(d, "metadata", {}) or {}).get("title", "Lore")
                        url = (getattr(d, "metadata", {}) or {}).get("url", "")
                        snippet = (getattr(d, "page_content", "") or "")
                        if url:
                            lines.append(f"- {title}: {snippet} (source: {url})")
                        else:
                            lines.append(f"- {title}: {snippet}")
                    if lines:
                        lore_msg = SystemMessage(
                            content="Relevant lore (use for context, do not quote verbatim unless asked):\n" + "\n".join(lines)
                        )
                        print(f"Found {len(lines)} lore notes, with total length of {sum([len(line) for line in lines])}, titles: {', '.join([line.split(':')[0] for line in lines])}")
                        return lore_msg
            except Exception as e:
                print(f"Warning: Lore retrieval failed: {e} \n{traceback.format_exc()}")

        def chatbot(state: State) -> State:
            messages = state["messages"]
            system_messages = []
            # Use last user message as query for exemplar selection
            user_text = None
            for m in reversed(messages):
                try:
                    role = getattr(m, "type", None) or getattr(m, "role", None)
                    if role in ("human", "user"):
                        user_text = m.content
                        assert isinstance(user_text, str)
                        break
                except Exception as e:
                    print(f"Warning: Failed to get user text from {e}")
            style_msg = _style_message(user_text)
            lore_msg = _lore_message(user_text)
            if lore_msg:
                system_messages.append(lore_msg)
            if style_msg:
                system_messages.append(style_msg)
            if self.system_prompt_text:
                system_messages.append(SystemMessage(content=self.system_prompt_text))  
            message = llm.invoke(system_messages + messages)
            print(f"Message Usage: {message.usage_metadata}")
            return {"messages": [message]}

        tool_node = ToolNode(tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")
        memory = InMemorySaver()
        graph = graph_builder.compile(checkpointer=memory)
        if self.debug:
            print(graph.get_graph().draw_mermaid())
        return graph

    @staticmethod
    def stream_graph_updates(graph: CompiledStateGraph, user_input: str):
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "1"}},
            stream_mode="values",
        )
        for event in events:
            if message := event.get("messages"):
                message[-1].pretty_print()

    def run_agent(self):
        graph = self.build_graph()
        while True:
            try:
                print("--------------------------------")
                user_input = input("User: ")
                if user_input.lower() == "\\q":
                    self.stream_graph_updates(graph, "Goodbye!")
                    break
                self.stream_graph_updates(graph, user_input)
            except KeyboardInterrupt:
                print("Goodbye!")
                break
            except Exception as e:
                print(e)
                # fallback to print the error
                user_input = f"What do you think of the error?\n{e}"
                self.stream_graph_updates(graph, user_input)
                break
