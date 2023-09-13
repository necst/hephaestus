# Hephaestus: Codesigning and Automating 3D Image Registration on Reconfigurable Architectures
Hephaestus generates efficient 3D image registration pipelines combined with reconfigurable accelerators. To alleviate the burden from the software, Hephaestus is a codesign of software-programmable accelerators that can adapt at run-time to the image volume dimensions. Hephaestus features a cross-platform abstraction layer that enables transparently high-performance and embedded systems deployment. However, given the computational complexity of 3D image registration, the embedded devices
become a relevant and complex setting being constrained in memory; thus, they require further attention and tailoring of the accelerators and registration application to reach satisfactory results. Therefore, Hephaestus also proposes an approximation mechanism that enables such devices to perform the 3D image registration and even achieve, in some cases, the accuracy of the high-performance ones. Overall, Hephaestus demonstrates 1.85× of maximum speedup, 2.35× of efficiency improvement with respect to the State of the Art, a maximum speedup of 2.51× and 2.76× efficiency improvements against our software, while attaining state-of-the-art accuracy on 3D registrations.
## Testing Environment
1. We tested the hardware code generation on two different machines based on Ubuntu 18.04 and Centos OS 7.6 respectively.
2. We used Xilinx Vitis Unified Platform and Vivado HLx toolchains 2019.2
3. We used python 3.6 with `argparse` `numpy` `math` packets on the generation machine
4. a) As host machines, or hardware design machines, we used Pynq 2.7 on the Zynq based platforms (Pynq-ZU, Zcu104), where we employ `cv2`, `numpy`, `pandas`, `torch`, `kornia`, `multiprocessing`, `statistics`, `argparse`, and `pydicom` packets.
4. b) We tested the Alveo u200 on a machine with CentOS 7.6, i7-4770 CPU @ 3.40GHz, and 16 GB of RAM, and the Alveo U280 on a machine with Ubuntu 20.04.6 LTS, AMD Ryzen 7 3700X @ 3.60 GHz, and 32 GB of RAM.

## Code organization
* `src/` source code for HLS based design, miscellaneous utilities, python host and testing code, and various scripts
 * `hls/` HLS source code for both design and testbench.
 * `python/` python host source code for single MI test and complete image registration, both with hardware acceleration and without it. 
 * `scripts/` miscellaneous scripts for the design generation, from tcl for Vivado and Vivado HLS to design configurator and design results extractions
* `platforms/` specific platforms makefile for the current supported boards: Pynq-Z2, Ultra96, Zcu104, Alveo u200


## FPGA-based Mutual Information (MI) accelerator generation flow

1. Source the necessary scripts, for example: `source <my_path_to_vitis>/settings64.sh`; for Alveo you will need to source xrt, e.g., `source /opt/xilinx/xrt/setup.sh`
2. Install the board definition files
3. Just do a `make`, or `make help` in the top folder for viewing an helper (print all helpers  `make helpall` )
4. use/modify the design space exploration script (i.e., `dse.sh` or `top_build.sh`) or generate your designs or use single instance specific generation 
4. a) `make hw_gen TRGT_PLATFORM=<trgt_zynq>` for generating an instance of a Zynq-based design, where `trgt_zynq=zcu104|pynqzu`
4. b) `make hw_gen TARGET=hw OPT_LVL=3 CLK_FRQ=$FREQZ TRGT_PLATFORM=alveo_u200 ncm=$NCM`  for generating an instance of the design on the Alveo u200 with target clock frequency `CLK_FRQ=$FREQZ` and maximum number of supported depth `$NCM`
5. [Optional] Generate other instances changing the design parameters. Look at Makefile parameters section for details.

## Testing designs

1. Complete at least one design in the previous section
2. `make sw` creates a deploy folder for the python code
3. `make deploy BRD_IP=<target_ip> BRD_USR=<user_name_on_remote_host> BRD_DIR=<path_to_copy>` copy onto the deploy folders the needed files
4. connect to the remote device, i.e., via ssh `ssh <user_name_on_remote_host>@<target_ip>`
5. [Optional] install all needed python packages as above, or the pynq package on the Alveo host machine
6. Refer to src/python/launch_tests.sh for clear examples on how to use the host code

### Makefile parameters

Follows some makefile parameters

#### General makefile parameters, and design configuration parameter
* TRGT_PLATFORM=`pynqzu|zcu104|alveo_u200|alveo_u280`
* Histogram Computation type HT=`float|fixed`
* Histogram PE Number PE=`1|2|4|8|16|32|64` 
* Entropy PE Number PE_ENTROP=`1|2|4|8|16|32`
* Core Number CORE_NR=`1|2`
* Maximum Supported Depth NCM=1|2|4|8|16|32|64|128|256|512|...

#### Vivado and Zynq specific parameters flow
* HLS_CLK=`default 10` clock period for hls synthesis
* FREQ_MHZ=`150` clock frequency for vivado block design and bitstream generation
* TOP_LVL_FN=`mutual_information_master` target top function for HLS
* HLS_OPTS=`5` HLS project flow. Supported options: 0 for only project build; 1 for sim only; 2 synth; 3 cosim; 4 synth and ip downto impl; 5 synth and ip export; 6 for ip export

#### Alveo specific parameters flow
* REPORT_FLAG=`R0|R1|R2` to report detail levels
* OPT_LVL=`0|1|2|3|s|quick` to optimization levels
* CLK_FRQ=`<target_mhz>` to ClockID 0 (board) target frequency, should be PCIe clock

## Extracting resources results

1. a) `make resyn_extr_zynq_<trgt_zynq>` e.g., trgt_zynq=`zcu104|pynqzu`, set FREQ_MHZ parameter if different from default.
1. b) `make resyn_extr_vts_<trgt_alveo>` e.g., trgt_alveo=`alveo_u200|alveo_u280`
2. You will find in the `build/` folder a new folder with all the generated bitstreams, and in the `build/<TRGT_PLATFORM>/` directory you will find a .csv with all the synthesis results


#### Credits and Contributors

Contributors: Sorrentino, Giuseppe and Venere, Marco and Conficconi, Davide and D'Arnese, Eleonora and Santambrogio, Marco D.

If you find this repository useful, please use the following citation(s):

```
@inproceedings{sorrentino2023hephaestus,
title = {HEPHAESTUS : Codesigning and Automating 3D Image Registration on Reconfigurable Architectures},
author = {Sorrentino, Giuseppe, and Venere, Marco and Conficconi, Davide and D'Arnese, Eleonora and Santambrogio, Marco D},
booktitle = {ACM Transactions on Embedded Computing Systems (TECS)},
journal={ACM Transactions on Embedded Computing Systems (TECS)},
year={2023},
volume={22},
number={5s},
issn = {1539-9087},
url = {https://doi.org/10.1145/3607928},
doi = {10.1145/3607928},
numpages = {24}
}
```
