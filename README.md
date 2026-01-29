An Intelligent Anycast Routing Method Integrating Software-Defined Networking and Parallel Hierarchical Graph Reinforcement Learning

This repository contains the implementation of the paper **"An Intelligent Anycast Routing Method Integrating Software-Defined Networking and Parallel Hierarchical Graph Reinforcement Learning."**

The framework utilizes **Mininet** and **Ryu** to simulate an Anycast TCP environment, collecting network path characteristics to train a Parallel Hierarchical Graph Reinforcement Learning model for optimizing routing decisions.

---

## 📋 Table of Contents

* System Architecture
* Prerequisites
* Installation & Setup
* Usage Workflow
* 1. Network Simulation (VM)
* 2. Data Collection (VM)
* 3. Host Bridge Configuration
* 4. Model Training & Server Processing
---

## 🏗 System Architecture

The experimental setup consists of three logical components:

1. **Virtual Machine (VM):** Runs the Mininet topology and Ryu SDN controller.
2. **Host Machine:** Acts as a communication bridge between the simulation environment and the processing server.
3. **Remote Server:** Executes the core PHGRL algorithm and processes real-time network state data.

---

## 🛠 Prerequisites

Ensure you have the following environments configured:

* **OS:** Ubuntu 18.04/20.04 (Recommended for Mininet)
* **Python:** 3.6+
* **Network Emulator:** Mininet
* **SDN Controller:** Ryu
* **Dependencies:**
* `networkx`
* `torch` / `tensorflow` (depending on your backend)
* `numpy`, `scipy`



---

## 🚀 Usage Workflow

Follow these steps strictly in order to initialize the environment, collect data, and run the main model.

### 1. Network Simulation (VM)

**Step 1.1: Start Mininet Topology**
Navigate to the project root in your Virtual Machine and initialize the topology script.

```bash
cd ~/Mininet
sudo run_mininet.sh

```

**Step 1.2: Start Ryu Controller**
Open a new terminal window in the same directory and launch the controller.

```bash
cd ~/Mininet
sudo run_ryu.sh

```

---

### 2. Data Collection (VM)

Once the Mininet topology is fully initialized, configure the stations to generate traffic.

**Step 2.1: Open Station Terminals**
Inside the Mininet CLI (`mininet>`), open xterm windows for the client node (`sta1`) and service nodes (`sta15` - `sta18`).

```bash
mininet> xterm sta1 sta15 sta16 sta17 sta18

```

**Step 2.2: Initialize Service Nodes**
In the terminals for **sta15, sta16, sta17, and sta18**, run the TCP server to listen for Anycast requests.

```bash
python anycast_tcp_server.py

```

**Step 2.3: Start Traffic Generation**
In the terminal for **sta1** (Client), run the client script to generate traffic and capture path metrics.

```bash
python anycast_tcp_client.py --sto_anycast

```

---

### 3. Host Bridge Configuration

To enable communication between the isolated VM network and the external computation server, use the Host Machine as a bridge.

**On the Host Machine:**

```bash
python http_data_server.py

```

*Ensure the Host Machine is reachable by both the VM and the Remote Server.*

---

### 4. Model Training & Server Processing

The core intelligent routing algorithm runs on the high-performance remote server.

**Step 4.1: Run the Main Algorithm**
Execute the main Reinforcement Learning loop in the background. Logs will be saved for monitoring.

```bash
nohup python3 -u train_main_thread.py >> ./logging/train_main_thread.log 2>&1 &

```

**Step 4.2: Monitor Progress**
View real-time training and routing logs:

```bash
tail -f ./logging/train_main_thread.log

```

---

## 📖 Citation

If you find this code useful in your research, please consider citing our paper:

**An Intelligent Anycast Routing Method Integrating Software-Defined Networking and Parallel Hierarchical Graph Reinforcement Learning**
