# comparison-of-LLM-prompting-techniques


## Manual for Setting Up the Virtual Environment
### Create a Virtual Environment: Open your terminal or command prompt and run:

+ Use Python 3.12   
+ Replace myenv with your preferred name for the environment.

```bash
python -m venv myenv
```

## Activate the Virtual Environment:

On Windows:
```bash
myenv\\Scripts\\activate
```
On macOS/Linux:
```bash
source myenv/bin/activate
```

## Install Required Packages: Use the requirements.txt file to install all necessary dependencies:

```bash
pip install -r requirements.tx
```

For problems with the in installation of llama-cpp-python, please refer to the official documentation:

https://llama-cpp-python.readthedocs.io/en/latest/

## Verify Installation: Ensure all packages are installed correctly:

```bash
pip list
```

# Setup Environment
## Create Environment File
Copy the example environment file to create your own:
```bash
cp .env.example .env
```

## Configure Environment Variables
Edit the .env file and set the following required model paths to the locally stored model or to the path on huggingface.

```bash
MODEL_PATH_GEMMA=      # Path to Gemma 2B model
MODEL_PATH_LLAMA3_1=   # Path to Llama 3.1 model
MODEL_PATH_LLAMA3_2=   # Path to Llama 3.2 model
MODEL_PATH_AYA_23=     # Path to Aya 2.3 model
```

# Start mlflow Server

```bash
mlflow ui
```

Open browser
Navigate to http://127.0.0.1:5000/