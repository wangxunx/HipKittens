GPU_TARGET=CDNA4
COMP_LEVEL=profile

GPU_RUNTIME := HIP

ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc
HIPFLAGS=-std=c++20 -I$(HIP_INCLUDE_DIR) -I../../include -Wall -Wextra -Wl,--allow-multiple-definition -DKITTENS_CDNA4 --offload-arch=gfx950 -fopenmp

ifeq ($(COMP_LEVEL),safe)
HIPFLAGS+= -O0
else ifeq ($(COMP_LEVEL),debug)
HIPFLAGS+= -g -O0
else ifeq ($(COMP_LEVEL),profile)
HIPFLAGS+= -O3 -Rpass-analysis=kernel-resource-usage #--save-temps
endif

CXXFLAGS ?= -Wall -Wextra
CXXFLAGS := -w
ICXXFLAGS += $(CXXFLAGS) 

TARGETS=naive_qkv2 naive_qkv naive_qk
SRC_naive_qkv2=naive_qkv2.cu
SRC_naive_qkv=naive_qkv.cu
SRC_naive_qk=naive_qk.cu
BUILD_DIR=build

all: $(TARGETS)

naive_qkv2: $(SRC_naive_qkv2)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) -c $< -o $(BUILD_DIR)/$@.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/$@.o -o $@

naive_qkv: $(SRC_naive_qkv)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) -c $< -o $(BUILD_DIR)/$@.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/$@.o -o $@

naive_qk: $(SRC_naive_qk)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) -c $< -o $(BUILD_DIR)/$@.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/$@.o -o $@


clean:
	rm -rf $(BUILD_DIR) $(TARGETS) 
