from langgraph.graph import END, START, StateGraph
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.state import State
from tools.human_in_the_loop import human_assist
from tools.tools import save_tool, search_tool, wiki_tool
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.prebuilt import tools_condition



class Agent:
    model: str

    def __init__(self, model: str = "gpt-oss:20b"):
        print(f"Initializing agent with model: {model}")
        self.model = model

    @staticmethod
    def route_tools(
        state: State,
    ):
        # print(state)
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError("No messages in state")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    def build_graph(self) -> CompiledStateGraph:
        tools = [save_tool, wiki_tool, human_assist, search_tool]
        llm = ChatOllama(model=self.model, temperature=0.6, num_gpu=-1, num_ctx=8192, keep_alive=True)
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
