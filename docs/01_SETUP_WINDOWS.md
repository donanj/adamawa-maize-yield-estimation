# First session: Windows + VS Code

Use VS Code as the main project workspace; Colab is optional for computation. No paid software or GPU is required for the installation check. Research tuning time depends on your CPU.

1. Install VS Code from https://code.visualstudio.com/download . In Extensions install **Python** and **Jupyter**, both from Microsoft. VS Code differs from the Visual Studio IDE you used for DigitalPersona.
2. Check existing Python before installing another version: open PowerShell and run `python --version`, `py -0p`, and `git --version`. An unavailable `py` command does not mean Python is absent. This starter was validated under Python 3.12.14. Use Python 3.12 if starting fresh; if Python is already installed, check its version before replacing it. Record the version actually used in Chapter Four (the earlier draft names Python 3.11). Official Windows downloads: https://www.python.org/downloads/windows/ . Git: https://git-scm.com/downloads/win . Use 64-bit Python and select Add Python to PATH when the installer offers it.
3. Extract the ZIP to a stable folder such as `C:\Research\adamawa-maize-yield`. Open the folder containing README.md and pipeline.py in VS Code (File > Open Folder).
4. Open Terminal > New Terminal. Verify that the terminal is in this project folder. Create a local environment using the Python you chose:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe pipeline.py check
```

If Python 3.12 is installed but `python` selects another version, use `py -3.12 -m venv .venv` for the first command. Direct `.venv\Scripts\python.exe` calls work without changing PowerShell execution policy.

Record the resolved packages after installation:

```powershell
.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 requirements-lock.txt
```

5. In VS Code press Ctrl+Shift+P, choose **Python: Select Interpreter**, and choose the project's `.venv\Scripts\python.exe`. This must match the interpreter printed by the check command.
6. Pause here and review the package versions and PASS message. Then run the following stages separately, checking output after each:

```powershell
.venv\Scripts\python.exe pipeline.py features
.venv\Scripts\python.exe pipeline.py prepare
.venv\Scripts\python.exe pipeline.py smoke
```

The smoke check trains just one RF held-out fold with two trials in its own folder. It is not a thesis result and will not be reused by the 20-trial research run.

7. After we review these checks, the next research stages are linear benchmarks and tree tuning. They are intentionally separate commands:

```powershell
.venv\Scripts\python.exe pipeline.py linear
.venv\Scripts\python.exe pipeline.py trees
.venv\Scripts\python.exe pipeline.py verify-trees
```

Default tree concurrency is two processes. For a computer with enough CPU/RAM, adjust it before running, e.g. `$env:ADAMAWA_WORKERS = '4'`. Leave the laptop connected to power and prevent sleep during long jobs. Stop with Ctrl+C; completed tasks can be reused when you resume with unchanged source inputs and settings.


## Official references

- VS Code Python setup: https://code.visualstudio.com/docs/languages/python
- Environment selection: https://code.visualstudio.com/docs/python/environments
- Python venv: https://docs.python.org/3/library/venv.html
- GitHub existing-code workflow: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
