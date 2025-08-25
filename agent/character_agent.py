import datetime
import traceback
from langchain_core.language_models import LanguageModelInput
from langgraph.graph import START, StateGraph
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph, Runnable
from langgraph.prebuilt import ToolNode

from agent.context_prompt import create_context_prompt
from agent.state import State
from tools.tools import save_tool, search_tool, wiki_tool
from tools.memory import memory_add, memory_search
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt import tools_condition

# from typing import Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
import random
from tools.voice_rag import build_voice_retriever, select_voice_lines
from tools.lore_db import build_lore_retriever, get_all_document_titles
from langchain_core.documents import Document


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
        lore_search_k: int = 8,
        fandom_wiki: str,
        character_name: str,
        additional_context_k: int = 5,
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

        # Get all available document titles for context retrieval
        self.all_available_titles = get_all_document_titles(
            collection_name=f"{self.character_name}_lore",
            where={"character": self.character_name},
        )
        self.additional_context_k = additional_context_k

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
        llm: Runnable[LanguageModelInput, BaseMessage] = ChatOllama(
            model=self.model,
            temperature=0.6,
            num_gpu=-1,
            num_ctx=self.context_window,
            keep_alive=True,
        )

        # Create a context LLM to decide what additional documents to retrieve
        context_llm = ChatOllama(
            model=self.model,
            temperature=0.3,
            num_gpu=-1,
            num_ctx=self.context_window,
        )

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
                print(f"Retrieved {len(docs)} lore documents")
                return {"retrieved_lore_docs": docs}
            except Exception as e:
                print(f"Warning: Lore retrieval failed: {e}")
                return {"retrieved_lore_docs": []}

        def retrieve_style(state: State) -> State:
            """Step 2: Retrieve style messages"""
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
            
            style_docs = []
            # Prefer semantic selection if available
            if self.voice_retriever:
                try:
                    style_docs = self.voice_retriever.invoke(user_text or "")
                    print(f"Retrieved {len(style_docs)} style documents via semantic search")
                except Exception as e:
                    print(f"Warning: Failed to select voice lines from {e}")
            
            # Fallback to deterministic sampling for stability and variety
            if not style_docs and self.voice_lines:
                seed = hash(user_text or "") & 0xFFFFFFFF
                rng = random.Random(seed)
                k = min(self.max_voice_lines, len(self.voice_lines))
                picks = rng.sample(self.voice_lines, k=k)
                # Convert to Document format for consistency
                style_docs = [Document(page_content=pick, metadata={"source": "fallback"}) for pick in picks]
                print(f"Retrieved {len(style_docs)} style documents via fallback")
            
            return {"retrieved_style_docs": style_docs}

        def retrieve_context(state: State) -> State:
            """Step 3: Create context retriever that picks additional documents"""
            messages = state["messages"]
            retrieved_lore_docs = state.get("retrieved_lore_docs", [])
            
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
                return {"retrieved_context_docs": []}
            
            # Build context about what's already retrieved
            already_retrieved_titles = [doc.metadata.get("title", "Unknown") for doc in retrieved_lore_docs]
            already_retrieved_content = [doc.page_content for doc in retrieved_lore_docs]
            print('--------------------------------')
            print(f"Already retrieved {len(already_retrieved_titles)} documents")
            print(f"With titles: {already_retrieved_titles}")
            
            context_prompt = create_context_prompt(self.system_prompt_text, user_text, already_retrieved_titles, already_retrieved_content, self.all_available_titles)

            try:
                response = context_llm.invoke([HumanMessage(content=context_prompt)])
                response_free_of_thought = response.content
                if '</think>' in response_free_of_thought:
                    think_end = response_free_of_thought.find('</think>')
                    response_free_of_thought = response_free_of_thought[think_end+len('</think>'):].strip()
                print(f"Free of thought: {response_free_of_thought}")
                # print(response)
                selected_titles = [title.strip() for title in response_free_of_thought.split('\n') if title.strip()]
                print(f"Selected {len(selected_titles)} additional document titles: {selected_titles}")
                
                # Retrieve the selected documents by title
                additional_docs = []
                for title in selected_titles:
                    try:
                        # Search for documents with this specific title
                        docs = self.lore_retriever.invoke(f"title: {title}")
                        # Filter to exact title match
                        matching_docs = [doc for doc in docs if doc.metadata.get("title") == title]
                        print(f"Found {len(docs)} documents for title '{title}', with matching docs: {len(matching_docs)}")
                        additional_docs.extend(matching_docs)
                    except Exception as e:
                        print(f"Warning: Context retrieval failed for title '{title}': {e}")
                
                # Remove duplicates based on document ID
                seen_ids = set()
                unique_docs = []
                for doc in additional_docs:
                    doc_id = getattr(doc, 'id', None) or doc.page_content[:50]
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_docs.append(doc)
                
                # Limit to 5 additional documents
                unique_docs = unique_docs[:self.additional_context_k]
                print(f"Retrieved {len(unique_docs)} additional context documents")
                print('--------------------------------')
                return {"retrieved_context_docs": unique_docs}
                
            except Exception as e:
                print(f"Warning: Context retrieval failed: {e}")
                return {"retrieved_context_docs": []}

        def chatbot(state: State) -> State:
            """Main chatbot node that uses all retrieved documents"""
            messages = state["messages"]
            retrieved_lore_docs = state.get("retrieved_lore_docs", [])
            retrieved_style_docs = state.get("retrieved_style_docs", [])
            retrieved_context_docs = state.get("retrieved_context_docs", [])
            system_messages = []
            current_in_universe_year = datetime.datetime.now().year + 160
            current_date_prompt = SystemMessage(content=f"The current date is {datetime.datetime.now().strftime('%B %d')}, {current_in_universe_year}")
            print(f"Current date: {current_date_prompt}")
            system_messages.append(current_date_prompt)
            # Add lore context
            if retrieved_lore_docs or retrieved_context_docs:
                all_lore_docs = retrieved_lore_docs + retrieved_context_docs
                lines: list[str] = []
                for d in all_lore_docs:
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
                    system_messages.append(lore_msg)
            
            # Add style context
            if retrieved_style_docs:
                style_lines = [d.page_content for d in retrieved_style_docs]
                style_msg = SystemMessage(
                    content="Style exemplars (tone and cadence):\n- " + "\n- ".join(style_lines)
                )
                print(f"Selected {len(style_lines)} voice lines")
                print(f"Voice lines: {'\n'.join(style_lines)}")
                system_messages.append(style_msg)
            
            # Add system prompt
            if self.system_prompt_text:
                system_messages.append(SystemMessage(content=self.system_prompt_text))
            
            message = llm.invoke(system_messages + messages)
            print(f"Message Usage: {message.usage_metadata}")
            return {"messages": [message]}

        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("retrieve_lore", retrieve_lore)
        graph_builder.add_node("retrieve_style", retrieve_style)
        graph_builder.add_node("retrieve_context", retrieve_context)
        graph_builder.add_node("chatbot", chatbot)  
        
        # Add edges
        graph_builder.add_edge(START, "retrieve_lore")
        graph_builder.add_edge("retrieve_lore", "retrieve_style")
        graph_builder.add_edge("retrieve_style", "retrieve_context")
        graph_builder.add_edge("retrieve_context", "chatbot")
        
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
