# PCRC: Parallel Cyclic Redundancy Check

## Usage

The build process tool of the project is CMake. The following are the steps to successfully build and run the project:

1. Clone this repository or copy the source folder into a device with CUDA compiler or Nvidia toolkit installed.

```
git clone https://github.com/mircodemarchi/PCRC.git
cd PCRC
```

2. Create a *build* folder and enter it.

```
mkdir build
cd build
```

3. Configure the CMake build process and start the compilation:

```
cmake ..
make
```

4. The compilation process produces two executable files: *PCRC.x* and *PCRC-test.x*. 
    - Run *PCRC.x* to start the CUDA implementation of PCRC.

    ```
    make run # equals to ./PCRC.x
    ```

    - Run *PCRC-test.x* to start the correctness tests of sequential CRC algorithms.
    ```
    make run-test # equals to ./PCRC-test.x
    ```

NOTE: CUDA implementation of PCRC runs multiple tests in sequence with different parameters in input, comparing it with the sequential CRC algorithm. The first execution logs appreared after about 60 seconds. The total execution time of all tests is about 22 minutes, then stop it when satisfied of the result obtained.

## Sources

The implementation of Parallel CRC can be found in [/src/device/pcrc](./src/device/pcrc) directory. 

The PCRC source code is divided in 3 different files, one for each width of the generated CRC code:
- [pcrc8.cu](./src/device/pcrc/pcrc8.cu);
- [pcrc16.cu](./src/device/pcrc/pcrc16.cu);
- [pcrc32.cu](./src/device/pcrc/pcrc32.cu);

For each CRC code width, therefore for each file in the folder, there are 3 kind of kernel implementation:
- The standard implementation, with more divergence, in which the parallel execution is done over `M` bits chuncks of the original message, where `M` is the length of the resulting CRC code. The name of this kernel function is *pcrc\<M\>_kernel*.
- A less divergent implementation, with a reduction algorithm during the over the entire length of the block. The name of this kernel function is *pcrc\<M\>_kernel_reduction*.
- A task parallelism implementation, that can be found in *pcrc\<M\>_parallel_task_parallelism* function.