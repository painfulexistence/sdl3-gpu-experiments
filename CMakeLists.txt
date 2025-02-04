cmake_minimum_required(VERSION 3.20.0)

project(test-sdl3 VERSION 0.1.0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

set(FETCHCONTENT_QUIET FALSE)
set(BUILD_SHARED_LIBS OFF)
option(SDL_TEST_LIBRARY OFF)
include(FetchContent)
FetchContent_Declare(SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.2.0
    GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(SDL3)
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf
    GIT_TAG v2.9.5
    GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(tinygltf)
find_package(fmt CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

find_program(GLSL_VALIDATOR "glslangValidator" REQUIRED)

set(SOURCES
    src/main.cpp
    src/helper.cpp
)

add_executable(main ${SOURCES})
target_compile_features(main PRIVATE cxx_std_20)
set_target_properties(main PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(main PRIVATE fmt::fmt SDL3::SDL3 tinygltf)

file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/*.vert
	${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/*.frag
	${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/*.comp
)
foreach(GLSL ${SHADER_SOURCES})
	get_filename_component(SOURCE_NAME ${GLSL} NAME)
	set(SPIRV ${CMAKE_CURRENT_SOURCE_DIR}/res/shaders/${SOURCE_NAME}.spv)
	add_custom_command(
		OUTPUT ${SPIRV}
		COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
		DEPENDS ${GLSL}
	)
	list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(copy_assets COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/res  DEPENDS ${SPIRV_BINARY_FILES})
add_dependencies(main copy_assets)