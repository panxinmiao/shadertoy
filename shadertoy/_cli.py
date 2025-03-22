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
    default=(1280, 720),
)

argument_parser.add_argument(
    "--output",
    type=str,
    help="The output video path to save the rendered shader to",
)

argument_parser.add_argument(
    "--duration",
    type=int,
    help="The duration of the output video in seconds",
    default=10,
)

argument_parser.add_argument(
    "--fps",
    type=int,
    help="The frames per second of the output video",
    default=60,
)


def start():
    args = argument_parser.parse_args()
    shader_id = args.id
    resolution = tuple(args.resolution)

    shader = load_shadertoy(shader_id)

    output = args.output
    if output:
        shader.to_video(output, duration=args.duration, fps=args.fps, resolution=resolution)
    else:
        shader.show(resolution = resolution)


if __name__ == "__main__":
    start()