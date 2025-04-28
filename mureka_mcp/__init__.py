"""Mureka MCP Server package."""
from .api import mcp


def main():
    """MCP Mureka api Server - HTTP call Mureka API for MCP"""
    mcp.run()


if __name__ == "__main__":
    main()
