# Repo setup instructions

### 1) Make directories

Create new repo folder with structure in one go
```console
$ pip install cookiecutter
$ cookiecutter gh:patrickmineault/true-neutral-cookiecutter
```

### 2) Install packages
Create conda environment without adding packages
```console
$ conda create -n myenv python=3.10
```

Install all normal packages you need.
```console
$ conda install pip
$ pip install ipykernel
$ pip install setuptools
$ pip install pytest
$ pip install numpy
$ pip install pandas
$ pip install matplotlib
$ pip install seaborn
```

Make requirements.txt file
```console
$ pip freeze > requirements.txt
```
Add ```-e .``` to requirements.txt
Delete all the bulk packages you don't need.

Run setup.py
```console
$ python setup.py install
```

### 3) Add black formatter
```console
$ pip install black
```
Go to settings and add this to settings.json
Type `cmd + shift + p` and type `Open Folder Settings (JSON)`
```json
"editor.formatOnSave": true,
"python.formatting.provider": "black",
"python.formatting.blackArgs": ["--line-length", "120"],
```
For testing later also add: 
```json
"python.testing.pytestEnabled": true,
"python.testing.pytestArgs": [
   "--verbose"
]
```
TIP: If you want to use autopep8 instead of black, you can do that by changing the settings.json to this:
```json
"editor.formatOnSave": true,
"python.formatting.provider": "autopep8",
"python.formatting.autopep8Args": "[-i", "-a", "-a", "--max-line-length", "120"],
```
and then run this command in the terminal:
```console      
$ autopep8 -i -a -a --max-line-length 120 -r src/
```

### 4) Add testing
configure testing in VSCode

Add empty init file to tests folder
```console
$ touch tests/__init__.py
```

Add test file to tests folder
```console
$ touch tests/test_main.py
```

Add test code to test file
```python
import pytest

def main():  # import actual main function
    return "Hello, world!"


def test_output():
    expected_output = "Hello, world!"  # replace with the expected output of your main function
    actual_output = main()  # call your main function and get its output
    assert actual_output == expected_output, "Main function output does not match expected output"
```

Get VSCode extension for easier running of tests: "Python Test Explorer for Visual Studio Code"