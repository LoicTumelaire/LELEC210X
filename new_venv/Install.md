# venv setup

used for the tensorflow models

## 1 : install python 3.10

[Python 3.10.11 Download](https://www.python.org/downloads/release/python-31011/)

## 2 : create a virtual env

```python
py -3.10 -m venv new_venv
```

## 3 : change the PATH

Linux:

```bash
source new_venv/Scripts/activate
```

Windows:

```bat
new_venv/Scripts/activate
```

## 4 : update pip

```python
python -m ensurepip --upgrade

```

## 5 : install requirements

```python
pip install -r new_venv/requirements.txt
```
