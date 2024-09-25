import pkg_resources
import subprocess
import sys
import platform

# Determine the correct pip command based on the environment
if platform.system() == 'Windows':
    pip_command = "pip"
    skip_uvloop = True
else:
    pip_command = "pip3.12"
    skip_uvloop = False

def install_package(package_name, version_specifier):
    """
    Installs the specified package with the given version specifier.
    """
    install_command = [sys.executable, '-m', 'pip', 'install', f"{package_name}{version_specifier}"]
    try:
        subprocess.check_call(install_command)
        print(f"Successfully installed {package_name}{version_specifier}.\n")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}{version_specifier}.\n")

def main():
    # Path to your requirements.txt file
    requirements_file = 'requirements.txt'

    # Read the requirements.txt file
    with open(requirements_file, 'r') as f:
        requirements = f.read().splitlines()

    # Parse the requirements
    parsed_requirements = pkg_resources.parse_requirements(requirements)

    # Check each requirement
    for req in parsed_requirements:
        package_name = req.project_name
        required_version = str(req.specifier)  # Get the version specified in requirements.txt

        if skip_uvloop and package_name.lower() == 'uvloop':
            print(f"Skipping installation of {package_name} on Windows.")
            continue

        try:
            # Check if the package is installed
            dist = pkg_resources.get_distribution(package_name)
            installed_version = dist.version

            if required_version:
                if dist.version not in req.specifier:
                    print(f"{package_name} is installed with version {installed_version}, but required version is {required_version}.")
                    print(f"Upgrading {package_name} to {required_version}...")
                    install_package(package_name, required_version)
            #     else:
            #         print(f"{package_name} is installed with the correct version {installed_version}.\n")
            # else:
            #     print(f"{package_name} is installed with version {installed_version}. No specific version required.\n")

        except pkg_resources.DistributionNotFound:
            if required_version:
                print(f"{package_name} is NOT installed. Required version: {required_version}.")
                print(f"Installing {package_name}{required_version}...")
                install_package(package_name, required_version)
            else:
                print(f"{package_name} is NOT installed.")
                print(f"Installing latest version of {package_name}...")
                install_package(package_name, "")

if __name__ == "__main__":
    print("Start pip version Check!")
    main()
    print("Complete pip version Check!")
