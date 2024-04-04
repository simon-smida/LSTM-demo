# SFC-project
Long Short-Term Memory (LSTM) demonstration

## Installation
Follow these steps to set up your project environment and run the application.

### Setting Up the Virtual Environment
First, create and activate a virtual environment to isolate the project dependencies:
```sh
python3 -m venv venv
```
### Activate the Virtual Environment:
- On Unix or MacOS, use:
```sh
source venv/bin/activate
```
- On Windows, use:
```Powershell
.\venv\Scripts\activate
```
### Installing Dependencies
Install the required packages specified in `requirements.txt`:
```sh
pip install -r requirements.txt
```

### Running the Application
With the environment set up and dependencies installed, you can now run the application:
```sh
python3 ./main.py
```

## Application demonstration 
- entering input sequence: `0,1,0,1,0,1`
- started training
- next symbol prediction
  
<p align="center">
  <img src="img/demo_v2.gif"/>
</p>

## Training visualization
![](img/demoviz-v2.gif)

## TODOs
[ ] add dependencies and usage\
[ ] integrate `./others/custom_lstm.py` model\
[ ] update docs
