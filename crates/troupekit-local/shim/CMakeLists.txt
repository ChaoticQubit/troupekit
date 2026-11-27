cmake_minimum_required(VERSION 3.14)
project(tk_llama C CXX)

if(NOT TK_LLAMA_SRC OR NOT TK_LIB_DIR)
  message(FATAL_ERROR "tk_llama needs TK_LLAMA_SRC and TK_LIB_DIR; build it through cargo with the llama feature")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${TK_LIB_DIR})
foreach(config RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} ${TK_LIB_DIR})
endforeach()

add_subdirectory(${TK_LLAMA_SRC} llama EXCLUDE_FROM_ALL)

add_library(tk_llama STATIC tk_llama.c)
target_link_libraries(tk_llama PRIVATE llama)
if(MSVC)
  target_compile_options(tk_llama PRIVATE /W3 /WX)
else()
  target_compile_options(tk_llama PRIVATE -Wall -Werror)
endif()
