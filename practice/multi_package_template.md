my_project/
├── package_a/
│   ├── package_a/
│   │   └── __init__.py
│   ├── setup.py
│   └── VERSION
├── package_b/
│   ├── package_b/
│   │   └── __init__.py
│   ├── setup.py
│   └── VERSION
├── shared_utils/
│   ├── shared_utils/
│   │   └── __init__.py
│   └── setup.py
└── README.md


my_project/
├── README.md
├── pyproject.toml               # Optional: use if managing global config or tool configs
├── package_a/
│   ├── package_a/
│   │   └── __init__.py
│   ├── setup.py
│   └── VERSION
├── package_b/
│   ├── package_b/
│   │   └── __init__.py
│   ├── setup.py
│   └── VERSION
├── shared_utils/
│   ├── shared_utils/
│   │   └── __init__.py
│   ├── setup.py
│   └── VERSION
└── requirements.txt            # Optional: for dev or test dependencies

# package_a/setup.py
from setuptools import setup, find_packages

with open('VERSION') as version_file:
    version = version_file.read().strip()

setup(
    name='package_a',
    version=version,
    packages=find_packages(),
    install_requires=[],
)

# package_b/setup.py
from setuptools import setup, find_packages

with open('VERSION') as version_file:
    version = version_file.read().strip()

setup(
    name='package_b',
    version=version,
    packages=find_packages(),
    install_requires=['package_a @ file://localhost/${PROJECT_ROOT}/package_a'],  # If depending on package_a
)

# shared_utils/setup.py
from setuptools import setup, find_packages

with open('VERSION') as version_file:
    version = version_file.read().strip()

setup(
    name='shared_utils',
    version=version,
    packages=find_packages(),
    install_requires=[],
)

# Example requirements.txt
-e ./package_a
-e ./package_b
-e ./shared_utils


===================
How to Support Different Programs Bound to Specific Versions
Let’s say:

Program X depends on package_a==1.0.0

Program Y depends on package_a==2.0.0

📁 Options:
1. Release and Tag Each Sub-Package
Use Git tags like:

package_a@1.0.0

package_a@2.0.0

Your deployment scripts or Docker builds can check out specific tags:

bash
Copy
Edit
git checkout tags/package_a@1.0.0 -- package_a
2. Publish Each Package to PyPI/Artifactory with Version
Each program’s requirements.txt or pyproject.toml pins the needed version:

txt
Copy
Edit
package_a==1.0.0
3. Use Local Caching
Build each package as a wheel and install the version locally per environment:

bash
Copy
Edit
cd package_a && python setup.py bdist_wheel
pip install dist/package_a-1.0.0-py3-none-any.whl
⚙️ Final Suggestion
For full automation and clean dependency tracking, I recommend:

Using VERSION files + bump_version.py

Using Git tags for each package version

Tagging releases in CI

Optionally integrating with Poetry or Hatch if you want modern tools