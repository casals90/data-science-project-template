## Project

A short description of the project.

Project Organization
------------
    ├── LICENSE
    ├── Makefile             
    ├── README.md             
    ├── config
    │   ├── requirements.txt  
    │   ├── settings.yaml     
    ├── data
    │   ├── external         
    │   ├── interim           
    │   ├── logs           
    │   ├── models           
    │   ├── processed        
    │   └── raw              
    │
    ├── docs
    │
    ├── notebooks
    │
    ├── reports               
    │   └── figures
    ├── src
    │   ├── data
    │   │
    │   ├── models
    │   │
    │   └── tools  
    │       └── startup.py
    │       └── utils.py
    │   │
    │   └── visualization     
    │       └── plot_data.py

--------

How to run the code
------------

### Dataset

The input dataset has to be in 'data/raw'.

### Create environment file

The command to generate the environment file (.env) is the following:

```commandline
printf "UID=$(id -u)\nGID=$(id -g)\n" > .env 
```

This command writes the user ID and group ID to a file. By default, Docker
reads the environment variables from this file.

### Docker image

From the root of the project do the following:

```commandline
docker build -t jupyter_environment ./
```

### Check the volumes

The system needs to interact with some folders in order to be able to take
necessary data, train models or predict. From the root folder, the volumes 
are mounted as follows:

```commandline
volumes:
  - ./data/models:/data/models
  - ./data/processed:/data/processed
  - ./data/raw:/data/raw
  - ./reports:/reports
```

Hence, the volumes are located inside the data folder.
Should the above locations be inconvenient, feel free to change
the 'docker-compose.yml' file to change them.
E.g. /myownpath/some_raw_data:/data/raw

### Start/shutdown the system

When ready, start the system from the root folder as follows:

```commandline
docker compose up
```

Shutdown the system as follows:

```commandline
docker compose down
```

### Utility to start/shutdown the system

To facilitate system startup and shutdown, there is a script available.
Please execute the following command from the root folder:

```commandline
make deploy_jupyter
```

This command initiates the Jupyterlab environment and displays the logs in
the command line. To shut down the system, press "CTRL + C".
