from langgraph.graph import START, StateGraph
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.state import State
from tools.tools import save_tool, search_tool, wiki_tool
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt import tools_condition


class Agent:
    model: str

    def __init__(self, model: str = "gpt-oss:20b", context_window: int = 8192, debug: bool = False):
        print(f"Initializing agent with model: {model}")
        self.model = model
        self.context_window = context_window

    def build_graph(self) -> CompiledStateGraph:
        # Core tools plus Discord toolkit (read/send messages)
        tools = [save_tool, wiki_tool, search_tool]
        llm = ChatOllama(
            model=self.model,
            temperature=0.6,
            num_gpu=-1,
            num_ctx=self.context_window,
            keep_alive=True,
        )
        llm = llm.bind_tools(tools)

        def chatbot(state: State) -> State:
            message = llm.invoke(state["messages"])
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
