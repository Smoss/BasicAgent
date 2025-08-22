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


class Agent:
    model: str

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        context_window: int = 8192,
        debug: bool = False,
        system_prompt_path: str | None = None,
        voice_lines_path: str | None = None,
        max_voice_lines: int = 3,
    ):
        print(f"Initializing agent with model: {model}")
        self.model = model
        self.context_window = context_window
        self.debug = debug
        self.system_prompt_text: str | None = None
        self.voice_lines: list[str] = []
        self.max_voice_lines = max_voice_lines
        self.voice_store = None
        if system_prompt_path:
            try:
                with open(system_prompt_path, "r", encoding="utf-8") as f:
                    self.system_prompt_text = f.read().strip()
            except Exception as e:
                print(
                    f"Warning: Failed to load system prompt from {system_prompt_path}: {e}"
                )
        if voice_lines_path:
            with open(voice_lines_path, "r", encoding="utf-8") as f:
                self.voice_lines = [ln.strip()[:200] for ln in f if ln.strip()]

            self.voice_store = build_voice_retriever(voice_lines_path)

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
            if self.voice_store:
                try:
                    picks = select_voice_lines(
                        self.voice_store, user_text, self.max_voice_lines
                    )
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
            content = "Style exemplars (tone and cadence):\n- " + "\n- ".join(picks)
            return SystemMessage(content=content)

        def chatbot(state: State) -> State:
            messages = state["messages"]
            if self.system_prompt_text:
                messages = [SystemMessage(content=self.system_prompt_text)] + messages
            # Use last user message as query for exemplar selection
            user_text = None
            for m in reversed(messages):
                try:
                    role = getattr(m, "type", None) or getattr(m, "role", None)
                    if role in ("human", "user"):
                        user_text = m.content
                        break
                except Exception as e:
                    print(f"Warning: Failed to get user text from {e}")
            style_msg = _style_message(user_text)
            if style_msg:
                if self.system_prompt_text:
                    messages = [messages[0], style_msg] + messages[1:]
                else:
                    messages = [style_msg] + messages
            message = llm.invoke(messages)
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
