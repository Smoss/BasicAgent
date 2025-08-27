import datetime
from langchain_core.language_models import LanguageModelInput
from langgraph.graph import START, StateGraph
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph, Runnable

from agent.context_prompt import create_context_prompt
from agent.helldivers.nodes import (
    LoreRetrievalNode,
    retrieve_campaigns,
    retrieve_current_status,
    retrieve_major_orders,
    retrieve_news,
    route_lore_retrieval,
)
from agent.state import State
from agent.types import DocumentsQuery
from tools.helldivers.training_manual_types import (
    convert_current_event_list,
    convert_major_order_list,
    convert_news_list,
    convert_planet_list,
)
from langgraph.checkpoint.memory import InMemorySaver


# from typing import Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
import random
from tools.voice_rag import build_voice_retriever
from tools.lore_db import build_lore_retriever, build_lore_retriever_tool, get_all_documents, get_lore_store
from langchain_core.documents import Document

from utils.persona_config import PersonaConfig


class SimpleCharacterAgent:
    context_model: str
    chat_model: str

    def __init__(
        self,
        *,
        context_model: str = "gpt-oss:20b",
        chat_model: str = "gpt-oss:20b",
        context_window: int = 8192,
        debug: bool = False,
        max_voice_lines: int = 5,
        lore_search_k: int = 5,
        additional_context_k: int = 10,
        persona_config: PersonaConfig,
    ):
        print(f"Initializing agent with context model: {context_model}")
        print(f"Initializing agent with chat model: {chat_model}")
        self.context_model = context_model
        self.chat_model = chat_model
        self.context_window = context_window
        self.debug = debug
        self.system_prompt_text: str
        self.voice_lines: list[str] = []
        self.max_voice_lines = max_voice_lines
        self.lore_search_k = lore_search_k
        self.voice_retriever = None
        self.fandom_wiki = persona_config.fandom_wiki
        self.character_name = persona_config.character_name
        # Pre-build lore retriever filtered to this character and wiki
        self.lore_retriever = build_lore_retriever(
            collection_name=f"{self.character_name}_lore",  # type: ignore
            where={"character": self.character_name},
            search_k=self.lore_search_k,
        )

        self.planet_store = get_lore_store(f"{self.character_name}_planets")

        self.lore_store = get_lore_store(f"{self.character_name}_lore")

        # Get all available document titles for context retrieval
        self.all_available_documents = get_all_documents(
            store=self.lore_store,
        )
        self.all_available_planets = get_all_documents(
            store=self.planet_store,
        )
        self.additional_context_k = additional_context_k

        with open(persona_config.system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt_text = f.read().strip()
        if persona_config.voice_lines_path:
            with open(persona_config.voice_lines_path, "r", encoding="utf-8") as f:
                self.voice_lines = [ln.strip() for ln in f if ln.strip()]

            self.voice_retriever = build_voice_retriever(self.character_name)

    def build_graph(self) -> CompiledStateGraph:
        # Core tools plus Discord toolkit (read/send messages)
        llm: Runnable[LanguageModelInput, BaseMessage] = ChatOllama(
            model=self.chat_model,
            temperature=0.6,
            num_gpu=-1,
            num_ctx=self.context_window,
            keep_alive=True,
        )

        context_retriever_tool = build_lore_retriever_tool(
            store=self.lore_store,
            collection_name=f"context",
        )
        planet_retriever_tool = build_lore_retriever_tool(
            store=self.planet_store,
            collection_name=f"planet",
        )

        lore_tools = [context_retriever_tool, planet_retriever_tool]

        lore_retrieval_node = LoreRetrievalNode(lore_tools)
        
        # Create a context LLM to decide what additional documents to retrieve
        context_llm = ChatOllama(
            model=self.context_model,
            temperature=0.3,
            num_gpu=-1,
            num_ctx=self.context_window,
        ).bind_tools(lore_tools)

        def retrieve_lore(state: State) -> State:
            """Step 1: Retrieve lore using vector search and store in state"""
            messages = state["messages"]
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

            if not user_text:
                return {"retrieved_lore_docs": []}

            try:
                docs = self.lore_retriever.invoke(user_text) or []
                return {"retrieved_lore_docs": docs}
            except Exception as e:
                print(f"Warning: Lore retrieval failed: {e}")
                return {"retrieved_lore_docs": []}

        def retrieve_style(state: State) -> State:
            """Step 2: Retrieve style messages"""
            messages = state["messages"]
            user_text = ""
            for m in reversed(messages):
                try:
                    role = getattr(m, "type", None) or getattr(m, "role", None)
                    if role in ("human", "user"):
                        user_text = m.content if isinstance(m.content, str) else ""
                        break
                except Exception as e:
                    print(f"Warning: Failed to get user text from {e}")

            assert isinstance(user_text, str)
            style_docs = []
            # Prefer semantic selection if available
            if self.voice_retriever:
                try:
                    style_docs = self.voice_retriever.invoke(user_text or "")
                except Exception as e:
                    print(f"Warning: Failed to select voice lines from {e}")

            # Fallback to deterministic sampling for stability and variety
            if not style_docs and self.voice_lines:
                seed = hash(user_text or "") & 0xFFFFFFFF
                rng = random.Random(seed)
                k = min(self.max_voice_lines, len(self.voice_lines))
                picks = rng.sample(self.voice_lines, k=k)
                # Convert to Document format for consistency
                style_docs = [
                    Document(page_content=pick, metadata={"source": "fallback"})
                    for pick in picks
                ]

            return {"retrieved_style_docs": style_docs}

        def retrieve_context(state: State) -> State:
            """Step 5: Create context retriever that picks additional documents"""
            messages = state["messages"]
            retrieved_lore_docs = state.get("retrieved_lore_docs", [])

            user_text = None
            for m in reversed(messages):
                try:
                    role = getattr(m, "type", None) or getattr(m, "role", None)
                    if role in ("human", "user"):
                        user_text = m.content if isinstance(m.content, str) else ""
                        break
                except Exception as e:
                    print(f"Warning: Failed to get user text from {e}")

            assert isinstance(user_text, str)
            if not user_text:
                return {"retrieved_context_docs": []}

            # Build context about what's already retrieved
            already_retrieved_titles = [
                doc.metadata.get("title", "Unknown") for doc in retrieved_lore_docs
            ]
            already_retrieved_content = [
                doc.page_content for doc in retrieved_lore_docs
            ]

            current_planets = state.get("active_campaigns", [])
            current_major_orders = state.get("active_major_orders", [])

            current_planets_context = f"The following planets are currently being contested:\n{convert_planet_list(current_planets)}"
            current_major_orders_context = f"The following major orders are currently active:\n{convert_major_order_list(current_major_orders)}"

            current_status = state.get("current_status", None)
            current_events = []
            if current_status:
                current_events = current_status.globalEvents

            current_status_context = (
                f"The current status is:\n{convert_current_event_list(current_events)}"
            )

            current_news = state.get("past_week_news", [])
            current_news_context = (
                f"The past week's news is:\n{convert_news_list(current_news)}"
            )

            additional_context = f"{current_planets_context}\n{current_major_orders_context}\n{current_status_context}\n{current_news_context}"
            context_prompt = create_context_prompt(
                self.system_prompt_text,
                user_text,
                already_retrieved_titles,
                already_retrieved_content,
                self.all_available_documents,
                self.all_available_planets,
                additional_context,
            )

            try:
                # response = 
                # ids_to_retrieve = [doc.doc_id for doc in response.documents]
                # additional_docs = []
                # try:
                #     if len(ids_to_retrieve) > 0:
                #         additional_docs.extend(
                #             self.lore_store.get_by_ids(ids_to_retrieve)
                #         )
                # except Exception as e:
                #     print(
                #         f"Warning: Context retrieval failed for IDs '{ids_to_retrieve}': {e}"
                #     )

                # # Remove duplicates based on document ID
                # seen_ids = set()
                # unique_docs = []
                # for doc in additional_docs:
                #     doc_id = getattr(doc, "id", None) or doc.page_content[:50]
                #     if doc_id not in seen_ids:
                #         seen_ids.add(doc_id)
                #         unique_docs.append(doc)

                # # Limit to 5 additional documents
                # unique_docs = unique_docs[: self.additional_context_k]
                return {"tool_messages": [context_llm.invoke(
                    [HumanMessage(content=context_prompt)]
                )]}

            except Exception as e:
                print(f"Warning: Context retrieval failed: {e}")
                return {"tool_messages": []}

        # def check_planets

        def chatbot(state: State) -> State:
            """Main chatbot node that uses all retrieved documents"""
            messages = state["messages"]
            retrieved_lore_docs = state.get("retrieved_lore_docs", [])
            retrieved_style_docs = state.get("retrieved_style_docs", [])
            retrieved_context_docs = state.get("retrieved_context_docs", [])
            system_messages = []
            current_in_universe_year = datetime.datetime.now().year + 160
            current_date_prompt = SystemMessage(
                content=f"The current date is {datetime.datetime.now().strftime('%B %d')}, {current_in_universe_year}"
            )
            system_messages.append(current_date_prompt)
            # Add lore context
            if retrieved_lore_docs or retrieved_context_docs:
                all_lore_docs = retrieved_lore_docs + retrieved_context_docs
                lines: list[str] = []
                for d in all_lore_docs:
                    title = (getattr(d, "metadata", {}) or {}).get("title", "Lore")
                    url = (getattr(d, "metadata", {}) or {}).get("url", "")
                    snippet = getattr(d, "page_content", "") or ""
                    if url:
                        lines.append(f"- {title}: {snippet} (source: {url})")
                    else:
                        lines.append(f"- {title}: {snippet}")
                if lines:
                    lore_msg = SystemMessage(
                        content="Relevant lore (use for context, do not quote verbatim unless asked):\n"
                        + "\n".join(lines)
                    )
                    system_messages.append(lore_msg)

            # Add style context
            if retrieved_style_docs:
                style_lines = [d.page_content for d in retrieved_style_docs]
                style_msg = SystemMessage(
                    content="Style exemplars (tone and cadence):\n- "
                    + "\n- ".join(style_lines)
                )
                system_messages.append(style_msg)
            
            retrieved_planet_lore = state.get("retrieved_planet_lore", [])
            if retrieved_planet_lore:
                planet_lines = [d.page_content for d in retrieved_planet_lore]
                planet_msg = SystemMessage(
                    content="Planet lore (use for context, do not quote verbatim unless asked):\n- "
                    + "\n- ".join(planet_lines)
                )
                system_messages.append(planet_msg)

            if active_campaigns := state.get("active_campaigns"):
                current_planets = (
                    "When describing planets avoid repeating the same information.\n"
                )
                "Don't quote exact percentages, use evocative language.\n"
                "Provide a description of the planet's biome\n"
                "Don't refer to planets not mentioned here unless they are explicitly mentioned in the conversation, Super Earth is the exception to this rule.\n"
                "Please keep track of which faction is fighting on which planet.\n"
                "If the player asks for suggetsions on where to deploy or for current events, only suggest planets that are currently being contested.\n"
                "The following planets are currently being contested:\n"
                current_planets += convert_planet_list(active_campaigns)
                campaign_msg = SystemMessage(content=current_planets)
                system_messages.append(campaign_msg)

            if active_major_orders := state.get("active_major_orders"):
                current_major_orders = "When describing major orders avoid repeating the same information.\n"
                "Don't quote exact percentages, use evocative language.\nProvide a description of the major order.\n"
                "When talking about the time remaining, use vague language. E.g. 'Only a week remains', 'We'll fail in a few days', 'Only a few hours remain'.\n"
                "Only use the first 3 significant digits of the current value and target value.\n"
                "The following major orders are currently active:\n"
                current_major_orders += convert_major_order_list(active_major_orders)
                major_order_msg = SystemMessage(content=current_major_orders)
                system_messages.append(major_order_msg)

            # if (past_week_news := state.get("past_week_news")):
            #     current_news = "The past week's news is:\n"
            #     current_news += convert_news_list(past_week_news)
            #     print(f"Current news: {current_news}")
            #     news_msg = SystemMessage(
            #         content=current_news
            #     )
            #     system_messages.append(news_msg)

            # if (current_status := state.get("current_status")):
            #     current_status_msg = SystemMessage(
            #         content=f"The current events are:\n{convert_current_event_list(current_status.globalEvents)}"
            #     )
            #     system_messages.append(current_status_msg)

            # Add system prompt
            if self.system_prompt_text:
                system_messages.append(SystemMessage(content=self.system_prompt_text))

            message = llm.invoke(system_messages + messages)  # type: ignore
            return {"messages": [message]}

        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("retrieve_lore", retrieve_lore)  # type: ignore
        graph_builder.add_node("retrieve_style", retrieve_style)
        graph_builder.add_node("retrieve_context", retrieve_context)
        graph_builder.add_node("retrieve_campaigns", retrieve_campaigns)  # type: ignore
        graph_builder.add_node("retrieve_major_orders", retrieve_major_orders)  # type: ignore
        graph_builder.add_node("retrieve_current_status", retrieve_current_status)  # type: ignore
        graph_builder.add_node("retrieve_news", retrieve_news)  # type: ignore
        graph_builder.add_node("lore_retrieval", lore_retrieval_node)
        graph_builder.add_node("chatbot", chatbot)

        # Add edges
        graph_builder.add_edge(START, "retrieve_lore")
        graph_builder.add_edge("retrieve_lore", "retrieve_style")
        graph_builder.add_edge("retrieve_style", "retrieve_campaigns")
        graph_builder.add_edge("retrieve_campaigns", "retrieve_major_orders")
        graph_builder.add_edge("retrieve_major_orders", "retrieve_current_status")
        graph_builder.add_edge("retrieve_current_status", "retrieve_news")
        graph_builder.add_edge("retrieve_news", "retrieve_context")
        # graph_builder.add_edge("retrieve_context", "chatbot")
        graph_builder.add_conditional_edges(
            "retrieve_context",
            route_lore_retrieval,
            {
                "lore_retrieval": "lore_retrieval",
                "chatbot": "chatbot",
            },
        )
        graph_builder.add_edge("lore_retrieval", "chatbot")

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
                user_input = f"What do you think of the error?\n{e}"
                self.stream_graph_updates(graph, user_input)
                break
