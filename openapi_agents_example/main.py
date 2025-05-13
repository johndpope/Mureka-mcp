import asyncio
import os
import shutil
from typing import Any

from agents import Agent, Runner, trace, gen_trace_id, RunContextWrapper, AgentHooks, Tool, RunResult
from agents.mcp import MCPServer, MCPServerStdio


class CustomAgentHooks(AgentHooks):
    def __init__(self, display_name: str):
        self.event_counter = 0
        self.display_name = display_name

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        self.event_counter += 1
        print(f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started")

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended with output {output}"
        )

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, source: Agent) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {source.name} handed off to {agent.name}"
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} started tool {tool.name}"
        )

    async def on_tool_end(
            self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        self.event_counter += 1
        print(
            f"### ({self.display_name}) {self.event_counter}: Agent {agent.name} ended tool {tool.name} with result {result}"
        )


async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Music Assistant",
        instructions=f"generate lyrics, song and background music(instrumental)",
        mcp_servers=[mcp_server],
        hooks=CustomAgentHooks(display_name="Music Agent"),
    )

    message = "Please create a song for my daughter Jessica to wish her a happy birthday and play it"
    print("\n" + "-" * 40)
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)


async def main():
    # Ask the user for the directory path
    MUREKA_API_KEY = "<insert-your-api-key-here>"
    MUREKA_API_URL = "https://api.mureka.ai"
    TIME_OUT_SECONDS = "300"
    MUREKA_MCP_BASE_PATH = os.path.expanduser('~/Desktop')
    print("path: " + MUREKA_MCP_BASE_PATH)
    async with MCPServerStdio(
            cache_tools_list=True,  # Cache the tools list, for demonstration
            params={"command": "uvx", "args": ["mureka-mcp"],
                    "env": {
                        "MUREKA_API_KEY": MUREKA_API_KEY,
                        "MUREKA_API_URL": MUREKA_API_URL,
                        "TIME_OUT_SECONDS": TIME_OUT_SECONDS,
                        "MUREKA_MCP_BASE_PATH": MUREKA_MCP_BASE_PATH
                    }},
            client_session_timeout_seconds=300
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="mureka mcp Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server)


if __name__ == "__main__":
    if not shutil.which("uvx"):
        raise RuntimeError("uvx is not installed. Please install it with `pip install uvx`.")

    asyncio.run(main())
