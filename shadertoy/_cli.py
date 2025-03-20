import argparse

from ._api import load_shadertoy

argument_parser = argparse.ArgumentParser(
    description="Download and render Shadertoy shaders"
)

argument_parser.add_argument(
    "id", type=str, help="The ID of the shader"
)
argument_parser.add_argument(
    "--resolution",
    type=int,
    nargs=2,
    help="The resolution to render the shader at",
    default=(800, 450),
)

def start():
    args = argument_parser.parse_args()
    shader_id = args.id
    resolution = args.resolution
    shader = load_shadertoy(shader_id, resolution=resolution)
    shader.show()


if __name__ == "__main__":
    start()