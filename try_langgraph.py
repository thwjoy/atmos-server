import getpass
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

STORY_LAYOUT = {
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
          "completionObjective": {
            "type": "direction",
            "question": "You are in a forest. The sun sets in the West. Which way is North?",
            "answer": "To the right of the sunset",
            "educationalHint": "Use the position of the sun to figure out directions."
          }
        }, 
        {
          "task": "Solve the Elder Willow's riddles",
          "details": "Answer three riddles to gain the forest's secrets.",
          "coinsCollected": 75,
          "completionObjective": {
            "type": "riddle",
            "question": "What is full of holes but still holds water?",
            "answer": "A sponge",
            "educationalHint": "Think of something you use to clean up messes!"
          }
        },
        {
          "task": "Defeat the Whispering Shades",
          "details": "Combat ghostly apparitions to claim the Moonlit Sigil.",
          "coinsCollected": 100,
          "completionObjective": {
            "type": "choice",
            "question": "The Whispering Shades are coming! Should you: A) Hide, B) Run, C) Shout loudly to scare them away?",
            "answer": "C",
            "educationalHint": "Sometimes making noise can surprise and scare others!"
          }
        }
      ]
    },
    {
      "name": "Conquer the Wailing Swamp",
      "dependencies": ["Moonlit Sigil"],
      "reward": "Fangstone Amulet",
      "tasks": [
        {
          "task": "Secure safe passage through the Bog of Sinking Souls",
          "details": "Build makeshift bridges to cross safely.",
          "coinsCollected": 60,
          "completionObjective": {
            "type": "simple-action",
            "description": "Think about how to build a safe crossing using the resources around you!",
            "answer": "Use the axe to chop down the tree and cross the bog.",
            "educationalHint": "There is a stone axe hidden in a bush."
          }
        },
        {
          "task": "Defeat the Fenriven Leviathan",
          "details": "Overcome the swamp serpent to retrieve the Fangstone Amulet.",
          "coinsCollected": 150,
          "completionObjective": {
            "type": "choice",
            "question": "The Leviathan approaches! Do you: A) Distract it with a loud noise, B) Sneak past it, or C) Use a glowing object to scare it?",
            "answer": "C",
            "educationalHint": "Creatures often fear unfamiliar glowing objects!"
          }
        },
        {
          "task": "Collect Swampfire Essence",
          "details": "Gather glowing liquid from deep within the swamp's dangerous areas.",
          "coinsCollected": 80,
          "completionObjective": {
            "type": "color-match",
            "question": "You need three glowing essences. One is blue, one is green, and one is yellow. Can you find them?",
            "answer": ["Blue", "Green", "Yellow"],
            "educationalHint": "Think about the colors of nature around you!"
          }
        }
      ]
    },
    {
      "name": "Survive the Obsidian Wastes",
      "dependencies": ["Fangstone Amulet"],
      "reward": "Radiant Ore",
      "tasks": [
        {
          "task": "Navigate the Void Shards",
          "details": "Avoid unstable magical remnants while traversing the terrain.",
          "coinsCollected": 120,
          "completionObjective": {
            "type": "obstacle-navigation",
            "question": "The shards glow red when unstable. Which ones should you avoid?",
            "answer": "The glowing red shards",
            "educationalHint": "Red often means danger—stay away from the glowing ones!"
          }
        },
        { 
          "task": "Bypass or defeat Ashbound Sentinels",
          "details": "Strategically disable or destroy golem-like guardians.",
          "coinsCollected": 200,
          "completionObjective": {
            "type": "logic-puzzle",
            "question": "The Sentinels have a code to deactivate them: 2, 4, 6, ___. What comes next?",
            "answer": "8",
            "educationalHint": "Look for the pattern in the numbers!"
          }
        },
        {
          "task": "Mine Radiant Ore from volcanic vents",
          "details": "Use specialized tools to safely collect the ore.",
          "coinsCollected": 180,
          "completionObjective": {
            "type": "tool-selection",
            "question": "Which tool would you use to mine Radiant Ore: A) A hammer, B) A glowing pickaxe, C) A shovel?",
            "answer": "B",
            "educationalHint": "The glowing pickaxe is strong enough to break the ore safely!"
          }
        }
      ]
    },
    {
      "name": "Ascend the Celestial Spire",
      "dependencies": ["Radiant Ore"],
      "reward": "Shard of Eternity",
      "tasks": [
        {
          "task": "Locate the entrance through Crystalline Caverns",
          "details": "Solve reflective light puzzles to find the spire's entrance.",
          "coinsCollected": 100,
          "completionObjective": {
            "type": "light-puzzle",
            "question": "If light enters a prism at an angle, which direction does it refract?",
            "answer": "It bends towards the thicker part of the prism",
            "educationalHint": "Light changes direction when it moves through glass!"
          }
        },
        {
          "task": "Solve the Celestial Cipher",
          "details": "Crack a complex magical puzzle to access the Vault of Eternity.",
          "coinsCollected": 150,
          "completionObjective": {
            "type": "cipher",
            "question": "The cipher reads: 1-3-1-2. What comes next if it alternates between adding and subtracting?",
            "answer": "1",
            "educationalHint": "Look at how the sequence alternates!"
          }
        },
        {
          "task": "Confront the Eclipsed Warden",
          "details": "Defeat the cursed guardian protecting the shard.",
          "coinsCollected": 250,
          "completionObjective": {
            "type": "decision",
            "question": "The Warden asks: 'To pass, choose a path. One is fire, one is water. Which is safer?'",
            "answer": "Water",
            "educationalHint": "Water can extinguish fire—think wisely!"
          }
        }
      ]
    }
  ],
  # "optionalChallenges": [
  #   {
  #     "name": "Rescue the Lost Explorer",
  #     "dependencies": ["Path of Echoes"],
  #     "reward": "Valuable information about the shard's location",
  #     "tasks": [
  #       {
  #         "task": "Find the Lost Explorer",
  #         "details": "Track their location within the Mistwood Veil.",
  #         "coinsCollected": 30,
  #         "completionObjective": {
  #           "type": "direction",
  #           "question": "The explorer left tracks leading North, East, or South. Which way did they go?",
  #           "answer": "North",
  #           "educationalHint": "Look at the compass directions for guidance!"
  #         }
  #       },
  #       {
  #         "task": "Free them from entangling magic",
  #         "details": "Use knowledge from the Elder Willow to dispel the enchantment.",
  #         "coinsCollected": 50,
  #         "completionObjective": {
  #           "type": "spell",
  #           "question": "To break the spell, chant: 'Light as air, strong as ___. Fill the blank.'",
  #           "answer": "stone",
  #           "educationalHint": "Think of something strong and solid like stone!"
  #         }
  #       }
  #     ]
  #   }
  # ]
}
 

STORY_PROGRESS = {
    "objectives": [
        {
            "name": "Navigate the Mistwood Veil",
            "tasks": [
                {
                    "task": "Discover the Path of Echoes",
                    "details": "Locate the hidden trail through the forest by deciphering ancient runes.",
                    "completed": False,
                },
                {
                    "task": "Solve the Elder Willow's riddles",
                    "details": "Answer three riddles to gain the forest's secrets.",
                    "completed": False,
                },
                {
                    "task": "Defeat the Whispering Shades",
                    "details": "Combat ghostly apparitions to claim the Moonlit Sigil.",
                    "completed": False,
                }
            ]
        },
    ]
}

TOTAL_COINS = 0 
ITEMS = []

## retrieve tools ### 

@tool
def show_map(input_text: str) -> str:
  """ Returns the total story progress so far """
  return STORY_PROGRESS

@tool
def check_coins(input_text: str) -> str:
    """ Returns the amount of coins a user has """
    return str(TOTAL_COINS)

@tool
def get_task(task_name: str) -> str:
  """
  Loops over tasks in the story layout and returns the task answer.

  Args:
      task_name (str): The name of the task to search for.

  Returns:
      dict: The correct answer for this task.
  """
  # Loop through both objectives and optionalChallenges
  for section in ["objectives", "optionalChallenges"]:
      for objective in STORY_LAYOUT.get(section, []):
          for task in objective.get("tasks", []):
              if task.get("task") == task_name:
                  return task.get("completionObjective")
  return None


@tool
def set_task_completion_status(task_name: str, status: str) -> str:
    """
    Updates the completion status of a task in STORY_PROGRESS, retrieves coins from STORY_LAYOUT,
    and unlocks the next objective if all tasks in the current objective are completed.

    Args:
        task_name (str): The name of the task to search for.
        status (str): The completion status to set, expected values are "true" or "false".

    Returns:
        str: A new artifact (reward) added to the inventory, or None if no reward.
    """
    # Convert string status to boolean
    status_value = status.lower() == "true"

    global TOTAL_COINS, ITEMS

    # Retrieve coins from STORY_LAYOUT
    coins_to_add = 0
    for section in ["objectives", "optionalChallenges"]:
        for objective in STORY_LAYOUT.get(section, []):
            for task in objective.get("tasks", []):
                if task.get("task") == task_name:
                    coins_to_add = task.get("coinsCollected", 0)

    # Update the task in STORY_PROGRESS
    for section in ["objectives", "optionalChallenges"]:
        for objective in STORY_PROGRESS.get(section, []):
            for task in objective.get("tasks", []):
                if task.get("task") == task_name:
                    # Set the 'completed' status
                    task["completed"] = status_value
                    if status_value:  # If the task is marked as completed
                        TOTAL_COINS += coins_to_add

                    # Check if all tasks in this objective are completed
                    all_tasks_completed = all(t["completed"] for t in objective.get("tasks", []))
                    if all_tasks_completed:
                        # Fetch the reward from the corresponding objective in STORY_LAYOUT
                        reward = None
                        for layout_objective in STORY_LAYOUT.get(section, []):
                            if layout_objective["name"] == objective["name"]:
                                reward = layout_objective.get("reward")
                                break
                                                
                        if reward:
                            ITEMS.append(reward)

                        # Unlock the next objective
                        unlock_next_objective(objective["name"])               
                        return reward

    return None


def unlock_next_objective(current_objective_name: str):
    """
    Unlocks the next objective in STORY_LAYOUT and adds it to STORY_PROGRESS if dependencies are met.

    Args:
        current_objective_name (str): The name of the current objective.
    """
    for i, layout_objective in enumerate(STORY_LAYOUT["objectives"]):
        if layout_objective["name"] == current_objective_name:
            # Check if there is a next objective
            if i + 1 < len(STORY_LAYOUT["objectives"]):
                next_objective = STORY_LAYOUT["objectives"][i + 1]

                # Check if dependencies are met
                dependencies_met = all(
                    dep in ITEMS for dep in next_objective.get("dependencies", [])
                )

                if dependencies_met:
                    # Add the next objective to STORY_PROGRESS
                    STORY_PROGRESS["objectives"].append({
                        "name": next_objective["name"],
                        "dependencies": next_objective.get("dependencies", []),
                        "reward": next_objective["reward"],
                        "tasks": [
                            {
                                "task": task["task"],
                                "details": task["details"],
                                "completed": False,
                                "coinsCollected": task["coinsCollected"],
                                "completionObjective": task["completionObjective"],
                            }
                            for task in next_objective.get("tasks", [])
                        ],
                    })

tools = [show_map, check_coins, get_task, set_task_completion_status]

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def tools_condition(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return 'CONTINUE'
    else:
        return 'END'

memory = MemorySaver()


graph_builder = StateGraph(State)

from langchain_openai import ChatOpenAI

api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

view_tool_node = ToolNode(tools=tools)
# update_tool_node = ToolNode(tools=update_tools)
graph_builder.add_node("tools", view_tool_node)
# graph_builder.add_node("chatbot2", chatbot)
# graph_builder.add_node("update_tools", update_tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        # If `tools`, then we call the tool node.
        "CONTINUE": "tools",
        # Otherwise we finish.
        "END": END
    }
)
# graph_builder.add_conditional_edges(
#     "tools",
#     tools_condition,
#     {
#       # If `tools`, then we call the tool node.
#       "CONTINUE": "update_tools",
#       # Otherwise we finish.
#       "END": END
#     }
# )
# Any time a tool is called, we return to the chatbot to decide the next step
# graph_builder.add_edge("tools", "chatbot2")
# graph_builder.add_edge("chatbot2", "update_tools")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

system_prompt = """
You are the narrator of an audio game, it is your job to guide the user through the game
and answer any questions that they may have or perform any actions.
"""

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("system", system_prompt), ("user", user_input)]}, config, stream_mode="values"):
        print("Assistant:", event["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        break