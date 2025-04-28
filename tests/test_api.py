import pytest
import asyncio
from mureka_mcp.api import generate_lyrics, generate_song, generate_instrumental



@pytest.mark.asyncio
async def test_generate_lyrics():
    try:
        title_and_lyrics = await generate_lyrics("Embrace of Night")
        print(title_and_lyrics)
    except Exception as e:
        print(str(e))


@pytest.mark.asyncio
async def test_generate_song():
    try:
        lyrics = "[Verse]\nIn the stormy night, I wander alone\nLost in the rain, feeling like I have been thrown\nMemories of you, they flash before my eyes\nHoping for a moment, just to find some bliss"
        result = await generate_song(lyrics=lyrics, prompt="r&b, slow, passionate, male vocal")
        print(result)
    except Exception as e:
        print(str(e))


@pytest.mark.asyncio
async def test_generate_instrumental():
    try:
        result = await generate_instrumental(prompt="r&b, slow, passionate, male vocal")
        print(result)
    except Exception as e:
        print(str(e))
