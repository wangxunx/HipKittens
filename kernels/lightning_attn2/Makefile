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

TARGET=lightning_attn2
# SRC=lightning_attn2.cu
SRC=lightning_attn2_v_split.cu
BUILD_DIR=build

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) -c $(SRC) -o $(BUILD_DIR)/lightning_attn2.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/lightning_attn2.o -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) 
