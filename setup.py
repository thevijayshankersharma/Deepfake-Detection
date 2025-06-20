import subprocess
import sys
import pkg_resources

required_packages = {
    'flask': '2.3.3',
    'flask-login': '0.6.3',
    'flask-sqlalchemy': '3.1.1',
    'werkzeug': '3.0.1',
    'pillow': '10.2.0',
    'torch': '2.1.2',
    'torchvision': '0.16.2',
    'numpy': '1.24.3',
    'opencv-python': '4.9.0.80',
    'face-recognition': '1.3.0',
    'librosa': '0.10.1',
    'matplotlib': '3.7.4',
    'transformers': '4.36.2',
    'PyWavelets': '1.4.1',
    'scikit-image': '0.21.0'
}

def check_and_install_packages():
    print("Checking and installing required packages...")
    
    for package, version in required_packages.items():
        try:
            pkg_resources.require(f"{package}=={version}")
            print(f"✓ {package} {version} is already installed")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"Installing {package} {version}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
            print(f"✓ {package} {version} has been installed")

if __name__ == "__main__":
    check_and_install_packages()
    print("\nAll required packages have been installed successfully!") 