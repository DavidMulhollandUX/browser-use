import asyncio

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from browser_use import Agent

# Load environment variables (especially GOOGLE_API_KEY)
load_dotenv()

# Initialize the Gemini model
# Ensure you have GOOGLE_API_KEY set in your .env file or environment
# Example model, choose one appropriate for your needs and availability
# common models: "gemini-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"
# As per LangChain docs, "gemini-2.0-flash" was used as an example.
# Let's use a generally available one like "gemini-pro" for broad compatibility,
# or "gemini-1.5-flash-latest" if vision/multimodal features are implicitly tested by the task later.
# For a simple text-based task, "gemini-pro" is fine.
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  # Or "gemini-1.5-flash-latest" for potentially more features
    temperature=0.0,
    # safety_settings can be adjusted if needed, e.g.
    # from langchain_google_genai import HarmBlockThreshold, HarmCategory
    # safety_settings={
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    # }
)

# Define the task for the agent
# This task is similar to the one in the original quick_start.py
task = "Go to duckduckgo.com, search for 'LangChain Google Gemini integration', and find the main LangChain documentation page about it. Return the URL of that page."

# Initialize the Agent
agent = Agent(task=task, llm=gemini_llm)

async def main():
    print(f"Starting agent with task: {task}")
    result = await agent.run()
    print("\nAgent Result:")
    if result and result.history:
        # Print the last result which might contain the answer
        final_action_result = result.history[-1].result[-1]
        if final_action_result.extracted_content:
            print(f"  Extracted content: {final_action_result.extracted_content}")
        elif final_action_result.error:
            print(f"  Error: {final_action_result.error}")
        else:
            print("  No specific content or error extracted from the last step.")

        # Check if the task was marked as done
        if result.is_done():
            print("  Task marked as completed.")
        else:
            print("  Task not marked as completed.")
    else:
        print("  No history found or agent did not run.")

if __name__ == '__main__':
    asyncio.run(main())
