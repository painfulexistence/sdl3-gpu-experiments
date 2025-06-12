# SDL3 GPU Experiments
[![macOS CI badge](https://github.com/painfulexistence/sdl3-gpu-experiments/actions/workflows/ci-macos.yml/badge.svg?branch=main)](https://github.com/painfulexistence/sdl3-gpu-experiments/actions/workflows/ci-macos.yml)
[![Linux CI badge](https://github.com/painfulexistence/sdl3-gpu-experiments/actions/workflows/ci-linux.yml/badge.svg?branch=main)](https://github.com/painfulexistence/sdl3-gpu-experiments/actions/workflows/ci-linux.yml)
[![Windows CI badge](https://github.com/painfulexistence/sdl3-gpu-experiments/actions/workflows/ci-windows.yml/badge.svg?branch=main)](https://github.com/painfulexistence/sdl3-gpu-experiments/actions/workflows/ci-windows.yml)
![gltf viewer screenshot](.github/media/pbr-sponza-demo.png)
My playground project for exploring and experimenting with SDL3 GPU APIs.

### Demo
[Sponza](https://www.youtube.com/watch?v=7ok396p7lmg)
[Compute Particles](https://www.youtube.com/watch?v=-IVx_nz232Y)

## Run the Project

### Prerequsites
- [CMake](https://cmake.org/download/) (required)

### Building
1. Clone the repo
```sh
git clone --recurse-submodules https://github.com/painfulexistence/sdl3-gpu-experiments.git
```
2. Setup Vcpkg
```sh
cd sdl3-gpu-experiments
./vcpkg/bootstrap-vcpkg.sh -disableMetrics
```
3. Run CMake
```sh
cmake --preset=dev
cmake --build --preset=dev
```

### Running
```sh
./build/Debug/main
```


## License
This project is distributed under the MIT License (except for the Sponza model).
See `LICENSE` for more information.