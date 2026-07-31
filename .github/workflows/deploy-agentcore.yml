"""
This module defines a conversational AI agent that can perform calculations
using the Strands framework.
"""

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator

# Initialize the Bedrock AgentCore application
app = BedrockAgentCoreApp()


# Configure the Claude model for the agent with guardrail
model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Load guardrail ID if available
guardrail_config = None
try:
    with open("guardrail_id.txt", "r", encoding="utf-8") as f:
        guardrail_id = f.read().strip()
        if guardrail_id:
            guardrail_config = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "1",
            }
            print(f"Loaded guardrail: {guardrail_id}")
except FileNotFoundError:
    print("No guardrail file found - running without guardrail")

model = BedrockModel(model_id=model_id, guardrail=guardrail_config)

# Create the agent with tools and system prompt
agent = Agent(
    model=model,
    tools=[calculator],
    system_prompt="You're a helpful assistant. You can do simple math calculation.",
)


@app.entrypoint
def strands_agent_bedrock(payload):
    """
    Main entrypoint for the Bedrock AgentCore Runtime.

    This function is called by AWS Bedrock AgentCore when the agent receives
    a request. It processes the user input and returns the agent's response.

    Args:
        payload (dict): Request payload containing user input
                       Expected format: {"prompt": "user question"}

    Returns:
        str: The agent's text response to the user's prompt
    """
    # Extract the user's prompt from the payload
    user_input = payload.get("prompt")

    # Process the input through the agent (handles tool selection and model inference)
    response = agent(user_input)

    # Extract and return the text content from the response
    return response.message["content"][0]["text"]


if __name__ == "__main__":
    # Run the application locally for testing
    # In production, this is handled by Bedrock AgentCore Runtime
    app.run()