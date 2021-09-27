#!/usr/bin/env python3

import docker, argparse, sys, os

parser = argparse.ArgumentParser(description="Compile and test template-tensors in different docker containers")
parser.add_argument("--image", type=str, default=None, help="Name of docker image to be run.")
parser.add_argument("--logdir", type=str, default=None, help="Directory where logs are stored.")
args = parser.parse_args()

if not args.logdir is None and not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)

root_path = os.path.dirname(os.path.abspath(sys.argv[0]))

client = docker.from_env()

if args.image is None:
    docker_paths = [os.path.join(root_path, f) for f in os.listdir(root_path)]
    docker_paths = [f for f in docker_paths if os.path.isdir(f) and os.path.isfile(os.path.join(f, "Dockerfile"))]
else:
    docker_path = os.path.join(root_path, image)
    if not os.path.isdir(docker_path) or not os.path.isfile(os.path.join(docker_path, "Dockerfile")):
        print(f"{docker_path} does not point to a valid docker directory")
        sys.exit(-1)
    docker_paths = [docker_path]

print(f"Running for {len(docker_paths)} docker images")

results = []
for docker_path in docker_paths:
    tag = f"fferflo/template-tensors-deps:{os.path.basename(docker_path)}"
    tag_file = tag.replace("/", ".").replace(":", ".")
    print(f"Building docker image {tag}")
    image, docker_build_logs = client.images.build(path=docker_path, tag=tag)
    docker_build_logs = "".join([b["stream"] for b in docker_build_logs if "stream" in b])
    if not args.logdir is None:
        with open(os.path.join(args.logdir, f"docker_{tag_file}.log"), "w") as f:
            f.write(docker_build_logs)

    try:
        print(f"Starting container {tag}")
        volumes = {os.path.dirname(root_path): {"bind": "/template-tensors", "mode": "ro"}}
        device_requests = [docker.types.DeviceRequest(capabilities=[["gpu"]])]
        container = client.containers.run(image=image, command="/bin/bash", stdin_open=True, detach=True, tty=True, volumes=volumes, device_requests=device_requests)

        print(f"Compiling template-tensors in container {tag}")
        build_exit_code, build_logs = container.exec_run("bash -c 'mkdir build && cd build && cmake -DBUILD_APPS=ON -DUPDATE_GIT_SUBMODULE=OFF /template-tensors && make -j && make -j tests'")
        build_logs = build_logs.decode()
        if not args.logdir is None:
            with open(os.path.join(args.logdir, f"build_{tag_file}.log"), "w") as f:
                f.write(build_logs)

        if build_exit_code == 0:
            print(f"Testing template-tensors in container {tag}")
            test_exit_code, test_logs = container.exec_run("bash -c 'cd /build && ctest'")
            test_logs = test_logs.decode()
            if not args.logdir is None:
                with open(os.path.join(args.logdir, f"test_{tag_file}.log"), "w") as f:
                    f.write(test_logs)
        else:
            test_exit_code = None

        results.append((tag, build_exit_code, test_exit_code))
    except KeyboardInterrupt:
        print("Stopping...")
        container.stop()
        container.remove()
        sys.exit(-1)
    container.stop()
    container.remove()


print()
def line(tag, build_exit_code, test_exit_code):
    print(f"{tag:>60}{build_exit_code:>7}{test_exit_code:>7}")
line("Tag", "Build", "Test")
for args in results:
    line(*args)
