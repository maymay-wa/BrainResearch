:nerd_face:
A well-written README provides essential information about your project,
such as installation instructions, usage examples and contributions guidelines.

# List of usefull commands:
all command should run under project root/working-directory

## On first run:
```bash 
#install Virtualenv is - a tool to set up your Python environments
pip install virtualenv
#create virtual environment (serve only this project):
python -m venv venv
#activate virtual environment
source ./venv/bin/activate   
+ (venv) should appear as prefix to all command (run next command just after activating venv)
#update venv's python package-installer (pip) to its latest version
pip install --upgrade pip
#install projects packages
pip install -e .[dev]     
``` 

## Modify package dependencies (add/remove/update external modules/packages):
#### Add new module:
1. Add package to pyproject.toml
2. Run:
```bash 
pip install -e .[dev]
``` 

#### Remove new module:
1. Remove the package from pyproject.toml
2. Run:
```bash 
pip uninstall <package-name>
```
note: if you're don't remember the exact package name copy it from: 
```bash
pip list
```

#### Steps in case of a package failure:
Cases like package installation interuppted in the middle or something like that
1. Try to remove package and install it again.
2. If it doesn't help delete venv folder 
3. repeat 'On first run' steps


## Health check (Lint/Tests/Tests-Coverage):
#### Lint Project:
Check formatting, type hinting, lint code & docstrings
```bash
tox run -e lint
```
#### Test Project: 
Run all tests on diffrent python versions and produce tests-coverage and tests-results reports (make sure tested versions installed before run):
```bash
tox run -f test
```
#### Lint and Test project (on diffrent python versions):
```bash
tox run
```

## Build/pack the project (should run on every version change)
```bash
python -m build
```

### $${\color{red} TBD}$$:
#### Package documentation:
```bash
tox run -e docs
```

