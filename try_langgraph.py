import getpass
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

STORY_PROGRESS = {
   "objectives": [
    {
      "name": "Navigate the Mistwood Veil",
      "dependencies": [],
      "reward": "Moonlit Sigil",
      "tasks": [
        {
          "task": "Discover the Path of Echoes",
          "details": "Locate the hidden trail through the forest by deciphering ancient runes.",
          "coinsCollected": 50,
          "completed": False,
          "completionObjective": {
            "type": "direction",
            "question": "You are in a forest. The sun sets in the West. Which way is North?",
            "answer": "",
            "educationalHint": "Use the position of the sun to figure out directions."
          }
        },
        {
          "task": "Solve the Elder Willow's riddles",
          "details": "Answer three riddles to gain the forest's secrets.",
          "coinsCollected": 75,
          "completed": False,
          "completionObjective": {
            "type": "riddle",
            "question": "What is full of holes but still holds water?",
            "answer": "",
            "educationalHint": "Think of something you use to clean up messes!"
          }
        },
        {
          "task": "Defeat the Whispering Shades",
          "details": "Combat ghostly apparitions to claim the Moonlit Sigil.",
          "coinsCollected": 100,
          "completed": False,
          "completionObjective": {
            "type": "choice",
            "question": "The Whispering Shades are coming! Should you: A) Hide, B) Run, C) Shout loudly to scare them away?",
            "answer": "",
            "educationalHint": "Sometimes making noise can surprise and scare others!"
          }
        }
      ]
    },
  ],
}

TOTAL_COINS = 0 
ITEMS = []

@tool
def show_map(input_text: str) -> str:
  """ Returns the total story progress so far """
  return STORY_PROGRESS

@tool
def set_objective_completion_status(task_name: str, status: str) -> dict:
    """
    Loops over tasks in the story layout, sets the completion status to the specified value, 
    updates the coins, and returns the updated task dictionary for the task with the specified name.

    Args:
        task_name (str): The name of the task to search for.
        status (str): The completion status to set, expected values are "true" or "false".

    Returns:
        dict: The updated task dictionary with the new completion status, 
              or None if the task is not found.
    """
    # Convert string status to boolean
    status_value = status.lower() == "true"

    global TOTAL_COINS

    # Loop through both objectives and optionalChallenges
    for section in ["objectives", "optionalChallenges"]:
        for objective in STORY_PROGRESS.get(section, []):
            for task in objective.get("tasks", []):
                if task.get("task") == task_name:
                    # Set the 'completed' status to the specified value
                    task["completed"] = status_value
                    TOTAL_COINS += task.get("coinsCollected")
                    return task
    return None

@tool
def check_coins(input_text: str) -> str:
    """ Returns the amount of coins a user has """
    return str(TOTAL_COINS)


tools = [show_map, check_coins, set_objective_completion_status]

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition


memory = MemorySaver()

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

from langchain_openai import ChatOpenAI

api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values"):
        print("Assistant:", event["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What is 2 + 3"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break