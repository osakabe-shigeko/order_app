import subprocess

output = subprocess.check_output("conda list -e", shell=True)
packages = [line.split("=")[0] for line in output.decode().split("\n") if line]
requirements = "\n".join(sorted(packages))

with open("requirements.txt", "w") as f:
    f.write(requirements)