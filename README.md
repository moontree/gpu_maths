# gpu_maths
math functions for tensorrt
## Usage
```
mkdir build 
cd build
cmake ..
make
```
## Env
Program can run on gt1080.

There is a strange bug when running on TX2(28.1), asum function will cause `bus error (core dumped)`.
