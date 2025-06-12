# SDL3 GPU Experiments
![gltf viewer screenshot](.github/media/pbr-sponza-demo.png)
My playground project for exploring and experimenting with SDL3 GPU APIs.

### Demo

### Supported Platforms
- Apple Silicon
- Windows
- Linux


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

## Running
```sh
./build/Debug/main
```


## License
This project is distributed under the MIT License (except for the Sponza model).
See `LICENSE` for more information.