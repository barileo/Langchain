from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for Seattle."""
    return "It's always sunny in Seattle"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")