import os
import sys

import requests
import warnings

from ._channel import TextureChannel, BufferChannel, CubeTextureChannel, VolumeTextureChannel
from ._audio import AudioChannel
from ._shadertoy import Shadertoy


HEADERS = {"user-agent": "https://github.com/panxinmiao/shadertoy"}

def _get_api_key() -> str:
    key = os.environ.get("SHADERTOY_API_KEY", None)
    if key is None:
        raise ValueError(
            '''Can not find SHADERTOY_API_KEY environment variable. \nSee: https://www.shadertoy.com/howto#q2'''
        )
    test_url = "https://www.shadertoy.com/api/v1/shaders/query/test"
    response = requests.get(test_url, params={"key": key}, headers=HEADERS)
    response.raise_for_status()
    response = response.json()
    if "Error" in response:
        raise ValueError(
            f"Failed to use ShaderToy API: {response['Error']}"
        )
    return key

def _get_cache_dir(subdir="media") -> os.PathLike:
    if sys.platform.startswith("win"):
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], "shadertoy")
    elif sys.platform.startswith("darwin"):
        cache_dir = os.path.join(os.environ["HOME"], "Library", "Caches", "shadertoy")
    else:
        if "XDG_CACHE_HOME" in os.environ:
            cache_dir = os.path.join(os.environ["XDG_CACHE_HOME"], "shadertoy")
        else:
            cache_dir = os.path.join(os.environ["HOME"], ".cache", "shadertoy")
    cache_dir = os.path.join(cache_dir, subdir)
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create cache directory at {cache_dir}, due to {e}")
    return cache_dir


def _get_media_resource_uri(src: str, use_cache: bool=True):
    media_url = "https://www.shadertoy.com"
    cache_dir = _get_cache_dir("media")
    cache_path = os.path.join(cache_dir, src.split("/")[-1])
    if use_cache and os.path.exists(cache_path):
        return cache_path
    else:
        response = requests.get(
            media_url + src, headers=HEADERS, stream=True
        )
        response.raise_for_status()
        if use_cache:
            with open(cache_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return cache_path
        else:
            return response.content


def load_data_from(uri) -> dict:
    if "/" in uri:
        shader_id = uri.rstrip("/").split("/")[-1]
    else:
        shader_id = uri
    url = f"https://www.shadertoy.com/api/v1/shaders/{shader_id}"
    response = requests.get(url, params={"key": _get_api_key()}, headers=HEADERS)
    response.raise_for_status()
    shader_data = response.json()
    if "Error" in shader_data:
        raise RuntimeError(
            f"Error: {shader_data['Error']} when load from https://www.shadertoy.com/view/{shader_id}"
        )
    return shader_data


def _solve_input_channels(inputs, size, channel_cache={}, use_cache=True):
    channels = [None] * 4
    for input in inputs:
        sampler = input["sampler"]
        ctype = input["ctype"]
        channel_idx = input["channel"]
        _id = input["id"]

        if _id in channel_cache:
            channel = channel_cache[_id]
        else:
            channel = None
            if ctype == "buffer":
                channel = BufferChannel(size=size, filter=sampler["filter"], wrap=sampler["wrap"])
            else:
                src = input["src"]
                if ctype == "texture":
                    resource_uri = _get_media_resource_uri(src, use_cache=use_cache)
                    channel = TextureChannel(resource_uri, filter=sampler["filter"], wrap=sampler["wrap"], vflip=sampler["vflip"])
                elif ctype == "music":
                    resource_uri = _get_media_resource_uri(src, use_cache=use_cache)
                    channel = AudioChannel(resource_uri, filter=sampler["filter"], wrap=sampler["wrap"])
                elif ctype == "musicstream":
                    warnings.warn("soundcloud is not supported, use a shadertoy media instead.")
                    src = "/media/a/a6a1cf7a09adfed8c362492c88c30d74fb3d2f4f7ba180ba34b98556660fada1.mp3"
                    resource_uri = _get_media_resource_uri(src, use_cache=use_cache)
                    channel = AudioChannel(resource_uri, filter=sampler["filter"], wrap=sampler["wrap"])
                elif ctype == "video":
                    from ._video import VideoChannel
                    resource_uri = _get_media_resource_uri(src, use_cache=use_cache)
                    channel = VideoChannel(resource_uri, filter=sampler["filter"], wrap=sampler["wrap"], vflip=sampler["vflip"])
                elif ctype == "cubemap":
                    uris = [_get_media_resource_uri(src, use_cache=use_cache)]
                    for i in range(1, 6):
                        ext = src.split(".")[-1]
                        uri = src.replace(f".{ext}", f"_{i}.{ext}")
                        uris.append(_get_media_resource_uri(uri, use_cache=use_cache))
                    channel = CubeTextureChannel(uris, filter=sampler["filter"], wrap=sampler["wrap"], vflip=sampler["vflip"])
                elif ctype == "volume":
                    resource_uri = _get_media_resource_uri(src, use_cache=use_cache)
                    channel = VolumeTextureChannel(resource_uri, filter=sampler["filter"], wrap=sampler["wrap"])
                else:
                    warnings.warn(
                        f"Unsupported channel type: {ctype}, id: {_id}"
                    )
            
            channel_cache[_id] = channel
        
        channels[channel_idx] = channel
    return channels


def load_from_json(shader_data, **kwargs) -> dict:
    use_cache = kwargs.pop("use_cache", True)

    if not isinstance(shader_data, dict):
        raise TypeError("shader_data must be a dict")


    if "Shader" not in shader_data:
        raise ValueError(
            "shader_data must have a 'Shader' key, following Shadertoy export format."
        )
    
    author = shader_data["Shader"]["info"]["username"]
    title = shader_data["Shader"]["info"]["name"]

    shadertoy_title = f"{title} by {author}"

    renderpass = shader_data["Shader"]["renderpass"]
    codes = {}
    buffer_idx = 0
    for r_pass in renderpass:
        pass_type = r_pass["type"]
        if pass_type == "buffer":
            r_pass["_name"] = f"buffer_{buffer_idx}"
            buffer_idx += 1
        elif pass_type in ("image", "sound", "common"):
            r_pass["_name"] = pass_type
        else:
            warnings.warn(f"Unsupported pass type: {pass_type}")
        
        pass_name = r_pass["_name"]
        codes[pass_name] = r_pass["code"]

    shadertoy = Shadertoy(
        main_code = codes["image"],
        common_code = codes.get("common", None),
        buffer_a_code = codes.get("buffer_0", None),
        buffer_b_code = codes.get("buffer_1", None),
        buffer_c_code = codes.get("buffer_2", None),
        buffer_d_code = codes.get("buffer_3", None),
        sound_code = codes.get("sound", None),
        title = shadertoy_title,
    )

    passes = {
        "image": shadertoy.main_pass,
        "buffer_0": shadertoy.buffer_a_pass,
        "buffer_1": shadertoy.buffer_b_pass,
        "buffer_2": shadertoy.buffer_c_pass,
        "buffer_3": shadertoy.buffer_d_pass,
        "sound": shadertoy.sound_pass,
    }

    channels_cache = {}
    # solve output channels
    for r_pass in renderpass:
        pass_type = r_pass["type"]
        if pass_type == "buffer":
            pass_name = r_pass["_name"]
            output_id = r_pass["outputs"][0]["id"]
            buffer_pass = passes[pass_name]
            channels_cache[output_id] = buffer_pass.render_target

    # solve input channels
    for r_pass in renderpass:
        pass_name = r_pass["_name"]
        if pass_name in passes:
            _pass = passes[pass_name]
            inputs = r_pass["inputs"]
            input_channels = _solve_input_channels(inputs, size=(800, 450), channel_cache=channels_cache, use_cache=use_cache)
            _pass.channel_0, _pass.channel_1, _pass.channel_2, _pass.channel_3 = input_channels

    return shadertoy

def load_shadertoy(uri, **kwargs) -> Shadertoy:
    shader_data = load_data_from(uri)
    shadertoy = load_from_json(shader_data, **kwargs)
    return shadertoy