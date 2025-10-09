
# OpenFAIR SereWay: Toward SEcurity and REliability benchmarking for the RailWAY IoT.

This is the official implementation of the papers:

 - ## _Open-FARI: An Open-source testbed for Federated Anomaly detection in the Railway IIoT_

and 

 - ## _SereWay: Toward SEcurity and REliability benchmarking for the RailWAY IoT_

from _Alessandra Rizzardi, Raffaele Della Corte, Jesus F. Cevallos M., Vittorio Orbinato, Simona De Vivo, Sabrina Sicari, Domenico Cotroneo, and Alberto Coen-Porisini_

If you find this code useful, please cite us!:

- Alessandra Rizzardi, Raffaele Della Corte, Jesus F. Cevallos M., Vittorio Orbinato, Simona De Vivo, Sabrina Sicari, Domenico Cotroneo, and Alberto Coen-Porisini, Open-FARI: An Open-source testbed for Federated Anomaly detection in the Railway IIoT, IWCMW 2025, Accepted, In press

- Alessandra Rizzardi, Raffaele Della Corte, Jesus F. Cevallos M., Vittorio Orbinato, Simona De Vivo, Sabrina Sicari, Domenico Cotroneo, and Alberto Coen-Porisini, SereWay: Toward SEcurity and REliability benchmarking for the RailWAY IoT, Under review

## Usage

### Clone the repo:

This is a nested repo, so be sure to clone the repo recursively, using the `--recursive` flag with the `git clone` command. Here's the syntax:

```bash
git clone --recursive https://github.com/DIETI-DISTA-IoT/SereWay
```

This will clone the repository and all its submodules.

Alternatively, if you've already cloned the repository without the `--recursive` flag, you can use the following command to initialize and update the submodules:

```bash
git submodule update --init --recursive
```

This will fetch and checkout the submodules recursively.


## Set up:

### Installing Docker

**On Windows:**

Download [Docker Desktop](https://www.docker.com/products/docker-desktop/) and run the program.

**On WSL:**

Follow [this guide](https://docs.docker.com/engine/install/ubuntu/) to install Docker CE for the Ubuntu distribution (commands for other distributions can be found on the same website).
Perform the post installation steps, as shown [here](https://docs.docker.com/engine/install/linux-postinstall/).
(Optional) You can also make Docker Daemon start on WSL initialization, by adding:
```
[boot]
systemd=true
```
to your `/etc/wsl.conf` within your WSL distribution and restart it with `wsl.exe --shutdown`.

To verify that Docker works, run `docker version` and `docker compose version` to verify that Docker Compose got installed as well.

**On Linux:**

Run `sudo apt-get update` and `sudo apt-get install docker.io docker-compose`.
Verify it works by executing `docker --version` and `docker compose --version`.

## Installing Make
**On Windows:**

Download Make from the [GnuWin32](https://gnuwin32.sourceforge.net/packages/make.htm) project.
Run the installer and add the `bin` directory to your `PATH` (inside Windows' Environment Variables).
Verify it works by opening a command prompt and executing `make --version`.

**On WSL or Linux:**

Run `sudo apt-get update` and `sudo apt-get install make`.
Verify it works by executing `make --version`.


## Building vehicle images and dashboard:

Use `make all` to build the producer and consumer docker images:

    make all

> **After the first build, to force a fast-rebuild of the image using fresh code:** You can use selective caching to accelerate building:

    make all-scache

It will only update the container's code from the repository and not the dependencies when building the images.


### Launching:

> **IMPORTANT** For using the wandb logging dashboard, you should have a file called **".env"** in the project's root folder containing your wandb api key under the WANDB_API_KEY voice:

```.env file
# .env file content:

WANDB_API_KEY=your_wandb_api_key_here

```

Use `docker compose up` to start the whole cluster:
    docker compose up -d

If you prefer you can start launching only some services:

    docker compose up -d zookeeper kafka dashboard


Adjust configurations for this script in the `config/default.yaml` or create an ovverride `*.yaml` configuration on the `config/override` directory that you can use to override a subset of params. To launch an override conf, use:

    python dashboard/app.py override=my_conf_filename

Comand-line args can be sent also using the hydra syntax (i.e. no hyphens) and created appending `+` 

    python dashboard/app.py +foo=bar

