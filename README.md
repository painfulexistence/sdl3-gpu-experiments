# SDL3 GLTF Viewer
![gltf viewer screenshot](.github/media/pbr-sponza-demo.png)

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
git clone --recurse-submodules https://github.com/painfulexistence/sdl3-gltf-viewer.git
```
2. Setup Vcpkg
```sh
cd sdl3-gltf-viewer
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