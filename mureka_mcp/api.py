# api.py
import os
import sys
import time
import httpx
import requests
from pathlib import Path
from datetime import datetime
from typing import Iterator, Union
from urllib.parse import urlparse
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

mcp = FastMCP("Mureka")
# setup API key，for calling mureka API，please refer to the method for obtaining：https://platform.mureka.ai/apiKeys
# os.environ["MUREKA_API_KEY"] = "<MUREKA_API_KEY>"
api_key = os.getenv('MUREKA_API_KEY')
global_base_path = os.getenv("MUREKA_MCP_BASE_PATH")
api_url = os.getenv('MUREKA_API_URL')
if api_url is None:
    api_url = "https://api.mureka.ai"

default_time_out = 60.0  # seconds
time_out_env = os.getenv('TIME_OUT_SECONDS')
if time_out_env is not None:
    default_time_out = float(time_out_env)


def is_file_writeable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent_dir = path.parent
    return os.access(parent_dir, os.W_OK)


def make_output_path(
        output_directory: str | None, base_path: str | None = None
) -> Path:
    output_path = None
    if output_directory is None:
        output_path = Path.home() / "Desktop"
    elif not os.path.isabs(output_directory) and base_path:
        output_path = Path(os.path.expanduser(base_path)) / Path(output_directory)
    else:
        output_path = Path(os.path.expanduser(output_directory))
    if not is_file_writeable(output_path):
        raise Exception(f"Directory ({output_path}) is not writeable")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def extract_filename_from_url(url):
    # 解析URL
    parsed_url = urlparse(url)
    # 获取路径的最后一个部分，即文件名
    filename = parsed_url.path.split('/')[-1]
    return filename


def check_audio_file(path: Path) -> bool:
    audio_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".flac",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
    }
    return path.suffix.lower() in audio_extensions


def handle_input_file(file_path: str, audio_content_check: bool = True) -> Path:
    if not os.path.isabs(file_path) and not os.environ.get("ELEVENLABS_MCP_BASE_PATH"):
        raise Exception(
            "File path must be an absolute path if ELEVENLABS_MCP_BASE_PATH is not set"
        )
    path = Path(file_path)
    if not path.exists() and path.parent.exists():
        raise Exception(f"File ({path}) does not exist")
    elif not path.exists():
        raise Exception(f"File ({path}) does not exist")
    elif not path.is_file():
        raise Exception(f"File ({path}) is not a file")

    if audio_content_check and not check_audio_file(path):
        raise Exception(f"File ({path}) is not an audio or video file")
    return path


@mcp.tool(
    description="""Generate lyrics with a given prompt then return the title and lyrics text to the client directly.
    
 ⚠️ COST WARNING: This tool makes an API call to mureka.ai which may incur costs. Only use when explicitly requested by the user.
 
    Args:
        prompt (str): The prompt to generate lyrics for song
        
    Returns:
        The title and lyrics of song.
    """
)
async def generate_lyrics(prompt: str) -> dict:
    try:
        if not api_key:
            raise Exception("Can not found API key.")
        if prompt == "":
            raise Exception("Prompt text is required.")
        # call mureka API
        url = f"{api_url}/v1/lyrics/generate"

        # set request parameters
        # for more parameter information, please refer to:https://platform.mureka.cn/docs/api/operations/post-v1-lyrics-generate.html
        headers = {'Authorization': 'Bearer {}'.format(api_key),
                   'Content-Type': 'application/json'}
        params = {'prompt': prompt}

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(url, json=params, headers=headers)
            response.raise_for_status()
            result = response.json()

        result_lyrics = result.get("lyrics", "")
        result_title = result.get("title", "")
        return {"lyrics": result_lyrics, "title": result_title}
    except httpx.HTTPError as e:
        raise Exception(f"HTTP request failed: {str(e)}") from e
    except KeyError as e:
        raise Exception(f"Failed to parse response: {str(e)}") from e


@mcp.tool(
    description="""Generate song based on the lyrics text and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    
    ⚠️ COST WARNING: This tool makes an API call to mureka.ai which may incur costs. Only use when explicitly requested by the user.
    
    Args:
        lyrics (str): The lyrics to generate song
        prompt (str, optional): Control song generation by inputting a prompt.For example:r&b, slow, passionate, male vocal.
        output_directory (str, optional): Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.
    
    Returns:
        The output file and name of song generated.
    """
)
async def generate_song(lyrics: str, prompt: str = "", output_directory: str | None = None) -> \
        list[TextContent]:
    try:
        if not api_key:
            raise Exception("Can not found API key.")
        if lyrics == "":
            raise Exception("lyrics text is required.")
        model: str = "auto"
        output_path = make_output_path(output_directory, global_base_path)
        # call mureka API
        url = f"{api_url}/v1/song/generate"

        # set request parameters
        # for more parameter information, please refer to:https://platform.mureka.ai/docs/api/operations/post-v1-song-generate.html
        headers = {'Authorization': 'Bearer {}'.format(api_key),
                   'Content-Type': 'application/json'}
        params = {'lyrics': lyrics,
                  'model': model,
                  'prompt': prompt}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=params, headers=headers)
            response.raise_for_status()
            result = response.json()

        # parse result, pick up task id to query task result
        task_id = result.get("id", "")
        if len(task_id) == 0:
            raise Exception(f"generate song failed")

        current_timestamp = datetime.now().timestamp()
        while True:
            if (datetime.now().timestamp() - current_timestamp) > default_time_out:
                raise Exception(f"generate song time out {default_time_out} seconds")
            song_urls, status = await query_song_task(task_id)
            if status == "failed" or status == "cancelled" or status == "timeouted":
                raise Exception(f"generate song:{status}")
            elif status == "succeeded":
                break
            else:
                time.sleep(1)
        # downloads songs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_path_group = []
        for song_url_item in song_urls:
            filename = extract_filename_from_url(song_url_item)
            response = requests.get(song_url_item)
            if response.status_code == 200:
                song_bytes = response.content
            else:
                raise Exception(f"generate song failed! Can't download songs")
            save_path = output_path / filename
            with open(save_path, "wb") as f:
                f.write(song_bytes)
            save_path_group.append(save_path)

        return [TextContent(
            type="text",
            text=f"Success. File saved as: {save_path}",
        ) for save_path in save_path_group]
    except httpx.HTTPError as e:
        raise Exception(f"HTTP request failed: {str(e)}") from e
    except KeyError as e:
        raise Exception(f"Failed to parse response: {str(e)}") from e


async def query_song_task(task_id: str) -> ([], str):
    try:
        url = f"{api_url}/v1/song/query/{task_id}"
        headers = {'Authorization': 'Bearer {}'.format(api_key)}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()

        status = result.get("status", "failed")
        if status == "succeeded":
            # pick songs url
            ret_songs_list = []
            choices_list = result.get("choices", [])
            for choice in choices_list:
                song_url = choice.get("url", "")
                if len(song_url) > 0:
                    ret_songs_list.append(song_url)
            if len(ret_songs_list) == 0:
                return [], "failed"
            return ret_songs_list, status
        else:
            return [], status
    except httpx.HTTPError as e:
        raise Exception(f"HTTP request failed: {str(e)}") from e
    except KeyError as e:
        raise Exception(f"Failed to parse response: {str(e)}") from e


async def query_instrumental_task(task_id: str) -> ([], str):
    try:
        url = f"{api_url}/v1/instrumental/query/{task_id}"
        headers = {'Authorization': 'Bearer {}'.format(api_key)}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()

        status = result.get("status", "failed")
        if status == "succeeded":
            # pick songs url
            ret_songs_list = []
            choices_list = result.get("choices", [])
            for choice in choices_list:
                song_url = choice.get("url", "")
                if len(song_url) > 0:
                    ret_songs_list.append(song_url)
            if len(ret_songs_list) == 0:
                return [], "failed"
            return ret_songs_list, status
        else:
            return [], status
    except httpx.HTTPError as e:
        raise Exception(f"HTTP request failed: {str(e)}") from e
    except KeyError as e:
        raise Exception(f"Failed to parse response: {str(e)}") from e


@mcp.tool(
    description="""Generate background music(instrumental) based on the prompt text and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    
    ⚠️ COST WARNING: This tool makes an API call to mureka.ai which may incur costs. Only use when explicitly requested by the user.
    
    Args:
        prompt (str, optional): Control music generation by inputting a prompt.For example:r&b, slow, passionate, male vocal.
        output_directory (str, optional): Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.
            
    Returns:
        The output file and name of background music(instrumental) generated.
    """
)
async def generate_instrumental(prompt: str = "", output_directory: str | None = None) -> list[TextContent]:
    try:
        if not api_key:
            raise Exception("Can not found API key.")
        if len(prompt) == 0:
            raise Exception("Prompt is needed.For example:r&b, slow, passionate, male vocal")
        model: str = "auto"  # model: The model to use. Use auto to select the latest model.Valid values:auto,
        # mureka-5.5 or mureka-6.
        output_path = make_output_path(output_directory, global_base_path)
        # call mureka API
        url = f"{api_url}/v1/instrumental/generate"

        # set request parameters
        # for more parameter information, please refer to:https://platform.mureka.ai/docs/api/operations/post-v1-instrumental-generate.html
        headers = {'Authorization': 'Bearer {}'.format(api_key),
                   'Content-Type': 'application/json'}
        params = {'model': model,
                  'prompt': prompt}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=params, headers=headers)
            response.raise_for_status()
            result = response.json()

        # parse result, pick up task id to query task result
        task_id = result.get("id", "")
        if len(task_id) == 0:
            raise Exception(f"generate instrumental failed")

        current_timestamp = datetime.now().timestamp()
        while True:
            if (datetime.now().timestamp() - current_timestamp) > default_time_out:
                raise Exception(f"generate instrumental time out {default_time_out} seconds")
            song_urls, status = await query_instrumental_task(task_id)
            if status == "failed" or status == "cancelled" or status == "timeouted":
                raise Exception(f"generate instrumental:{status}")
            elif status == "succeeded":
                break
            else:
                time.sleep(1)
        # downloads songs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_path_group = []
        for song_url_item in song_urls:
            filename = extract_filename_from_url(song_url_item)
            response = requests.get(song_url_item)
            if response.status_code == 200:
                song_bytes = response.content
            else:
                raise Exception(f"generate instrumental failed! Can't download songs")
            save_path = output_path / filename
            with open(save_path, "wb") as f:
                f.write(song_bytes)
            save_path_group.append(save_path)

        return [TextContent(
            type="text",
            text=f"Success. File saved as: {save_path}",
        ) for save_path in save_path_group]

    except httpx.HTTPError as e:
        raise Exception(f"HTTP request failed: {str(e)}") from e
    except KeyError as e:
        raise Exception(f"Failed to parse response: {str(e)}") from e


def play(
        audio: Union[bytes, Iterator[bytes]]
) -> None:
    if isinstance(audio, Iterator):
        audio = b"".join(audio)

    try:
        import io

        import sounddevice as sd  # type: ignore
        import soundfile as sf  # type: ignore
    except ModuleNotFoundError:
        message = (
            "`pip install sounddevice soundfile` required when `use_ffmpeg=False` "
        )
        raise ValueError(message)
    sd.play(*sf.read(io.BytesIO(audio)))
    sd.wait()


@mcp.tool(description="Play an audio file. Supports WAV and MP3 formats.")
def play_audio(input_file_path: str) -> TextContent:
    file_path = handle_input_file(input_file_path)
    play(open(file_path, "rb").read())
    return TextContent(type="text", text=f"Successfully played audio file: {file_path}")


def main():
    print("Starting MCP server")
    """Run the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
