from typing import Annotated, List
import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import initialize_agent, Tool
# from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import  MessagesPlaceholder
from langchain.schema import SystemMessage, AIMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage


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
      "reward": "Fangstone Amulet and Swampfire Essence",
      "tasks": [
        {
          "task": "Secure safe passage through the Bog of Sinking Souls",
          "details": "Build makeshift bridges or tame a swamp beast to cross safely.",
          "coinsCollected": 60,
          "completionObjective": {
            "type": "simple-action",
            "description": "Think about how to build a safe crossing using the resources around you!"
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
      "dependencies": ["Fangstone Amulet", "Swampfire Essence"],
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
  "optionalChallenges": [
    {
      "name": "Rescue the Lost Explorer",
      "dependencies": ["Path of Echoes"],
      "reward": "Valuable information about the shard's location",
      "tasks": [
        {
          "task": "Find the Lost Explorer",
          "details": "Track their location within the Mistwood Veil.",
          "coinsCollected": 30,
          "completionObjective": {
            "type": "direction",
            "question": "The explorer left tracks leading North, East, or South. Which way did they go?",
            "answer": "North",
            "educationalHint": "Look at the compass directions for guidance!"
          }
        },
        {
          "task": "Free them from entangling magic",
          "details": "Use knowledge from the Elder Willow to dispel the enchantment.",
          "coinsCollected": 50,
          "completionObjective": {
            "type": "spell",
            "question": "To break the spell, chant: 'Light as air, strong as ___. Fill the blank.'",
            "answer": "stone",
            "educationalHint": "Think of something strong and solid like stone!"
          }
        }
      ]
    }
  ]
}


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
BAG_OF_TRICKS = []

@tool
def show_map(input_text: str) -> str:
  """ Returns the total story progress so far """
  return STORY_PROGRESS

@tool
def get_completion_objective(task_name: str) -> dict:
    """
    Loops over tasks in the story layout and returns the completionObjective dictionary
    for the task with the specified name.
    
    Args:
        task_name (str): The name of the task to search for.
        story_layout (dict): The story layout containing objectives and optional challenges.
        
    Returns:
        dict: The completionObjective dictionary for the matching task, or None if not found.
    """
    # Loop through both objectives and optionalChallenges
    for section in ["objectives", "optionalChallenges"]:
        for objective in STORY_LAYOUT.get(section, []):
            for task in objective.get("tasks", []):
                if task.get("task") == task_name:
                    return task.get("completionObjective")
    return None

# @tool
# def evaluate_answer(task_name: str, user_answer: str) -> bool



def main():
    # Load API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set your OPENAI_API_KEY environment variable.")
        return

    print("LangChain Chat Agent Example with Tools")
    print("Type 'exit' to end the conversation.")

    chat_model = ChatOpenAI(model="gpt-4", openai_api_key=api_key)

    # Define the tool
    tools = [
        Tool(
          name="Show progess",
          func=show_map,
          description="Shows the user how far through the game they are, and what tasks and challenges are available for them to solve."
        ),
        Tool(
          name="Get completionObjective",
          func=get_completion_objective,
          description="Gets completionObjective, allowing it to be used to evaluate the answer"
        )
    ]

    message = SystemMessage(
        content=(
            """
            You are the narrator of childrend audio adventure game. Is is
            your role to encourage the child while answering and evaluating what they want
            to do. You should add lots of excitment and mystery to the game.
            """
        )
    )
    chat_history = []
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent="zero-shot-react-description",
        prompt=prompt,
        # memory=memory
    )

    # Chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        try:
            # Run the agent
            response = agent.invoke({"input": user_input, "chat_history": chat_history})
            base_message = AIMessage(content=response['output'])
            chat_history.append(base_message)
            print(f"ChatGPT: {response['output']}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()






