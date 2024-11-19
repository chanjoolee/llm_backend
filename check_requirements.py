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

def uninstall_package(package_name):
    """
    Uninstalls the specified package.
    """
    uninstall_command = [sys.executable, '-m', 'pip', 'uninstall', '-y', package_name]
    try:
        subprocess.check_call(uninstall_command)
        print(f"Successfully uninstalled {package_name}.\n")
    except subprocess.CalledProcessError:
        print(f"Failed to uninstall {package_name}.\n")

def main():
    # Path to your requirements.txt file
    requirements_file = 'requirements.txt'

    # Read the requirements.txt file
    with open(requirements_file, 'r') as f:
        requirements = f.read().splitlines()

    # Parse the requirements from requirements.txt into a dictionary (case-insensitive)
    required_packages = {}
    for line in requirements:
        if line.strip() and not line.startswith('#'):
            package, _, version = line.partition('==')
            required_packages[package.strip().lower()] = version.strip()

    # Get the list of currently installed packages (all keys in lowercase)
    installed_packages = {pkg.key.lower(): pkg.version for pkg in pkg_resources.working_set}

    # Install or update packages listed in requirements.txt
    for package, required_version in required_packages.items():
        if package in installed_packages:
            if installed_packages[package] != required_version:
                print(f"Updating {package} to version {required_version}...")
                install_package(package, f"=={required_version}")
        else:
            print(f"Installing {package} version {required_version}...")
            install_package(package, f"=={required_version}")

    # Uninstall packages that are installed but not listed in requirements.txt
    # for package in installed_packages.keys():
    #     if package not in required_packages:
    #         print(f"Uninstalling {package} as it is not listed in requirements.txt...")
    #         uninstall_package(package)

if __name__ == "__main__":
    print("Start pip version Check!")
    main()
    print("Complete pip version Check!")
