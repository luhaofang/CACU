################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cuda/bit_math.cu \
../cuda/math.cu \
../cuda/matrix.cu 

OBJS += \
./cuda/bit_math.o \
./cuda/math.o \
./cuda/matrix.o 

CU_DEPS += \
./cuda/bit_math.d \
./cuda/math.d \
./cuda/matrix.d 


# Each subdirectory must supply rules for building sources it contributes
cuda/%.o: ../cuda/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -O2 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "cuda" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -O2 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


