#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <level_zero/ze_api.h>
#include "../include/ze_utils.h"

int NUM_ROUNDS = 1;

int main(int argc, char* argv[]) {
    if (argc < 3)
       exit_msg(strcat(argv[0], " groupSize numGroup" ));

    ze_result_t errno;

    // driver and device discovery
    // Initialize the driver

    //  PLATFORM AND DEVICE
    printf(">>> Initializing OpenCL Platform and Device...\n");
    errno  = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    check_error(errno, "zeInit");

    // Discover all the driver instances
    uint32_t driverCount = 0;
    errno = zeDriverGet(&driverCount, NULL);
    check_error(errno, "zeDriverGet");

    //Now where the phDrivers
    ze_driver_handle_t* phDrivers = (ze_driver_handle_t*) malloc(driverCount * sizeof(ze_driver_handle_t));
    errno = zeDriverGet(&driverCount, phDrivers);
    check_error(errno, "zeDriverGet");

    // Device who will be selected
    ze_device_handle_t hDevice = NULL;
    ze_driver_handle_t hDriver = NULL;

    for(uint32_t driver_idx = 0; driver_idx < driverCount; driver_idx++) {

        hDriver = phDrivers[driver_idx];
        /* - - - -
        Device
        - - - - */

        // if count is zero, then the driver will update the value with the total number of devices available.
        uint32_t deviceCount = 0;
        errno = zeDeviceGet(hDriver, &deviceCount, NULL);
        check_error(errno, "zeDeviceGet");

        ze_device_handle_t* phDevices = (ze_device_handle_t*) malloc(deviceCount * sizeof(ze_device_handle_t));
        errno = zeDeviceGet(hDriver, &deviceCount, phDevices);
        check_error(errno, "zeDeviceGet");

        for(uint32_t device_idx = 0;  device_idx < deviceCount; device_idx++) {
            ze_device_properties_t device_properties;
            errno = zeDeviceGetProperties(phDevices[device_idx], &device_properties);
            check_error(errno, "zeDeviceGetProperties");
            if (device_properties.type == ZE_DEVICE_TYPE_GPU){
                    printf("Running on Device #%d %s who is a GPU. \n", device_idx, device_properties.name);
                    hDevice = phDevices[device_idx];
                    break;
            }
        }

        free(phDevices);
        if (hDevice != NULL) {
            break;
        }
    }

    free(phDrivers);


    // CONTEXT
    ze_context_handle_t hContext = NULL;
    // Create context
    ze_context_desc_t context_desc = {
        ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        NULL,
        0};
    errno = zeContextCreate(hDriver, &context_desc, &hContext);
    check_error(errno, "zeContextCreate");


    // COMMAND QUEUE
    // Discover all command queue types
    uint32_t cmdqueueGroupCount = 0;
    zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, NULL);

    ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*) malloc(cmdqueueGroupCount * sizeof(ze_command_queue_group_properties_t));
    errno = zeDeviceGetCommandQueueGroupProperties(hDevice, &cmdqueueGroupCount, cmdqueueGroupProperties);
    check_error(errno, "zeDeviceGetCommandQueueGroupProperties");

    // Find a proper command queue
    uint32_t computeQueueGroupOrdinal = cmdqueueGroupCount;
    for( uint32_t i = 0; i < cmdqueueGroupCount; ++i ) {
        if( cmdqueueGroupProperties[ i ].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE ) {
            computeQueueGroupOrdinal = i;
            break;
        }
    }

    // Command queue
    ze_command_queue_desc_t commandQueueDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        NULL,
        computeQueueGroupOrdinal,
        0, // index
        0, // flags
        ZE_COMMAND_QUEUE_MODE_DEFAULT,
        ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };

    ze_command_queue_handle_t hCommandQueue;
    errno = zeCommandQueueCreate(hContext, hDevice, &commandQueueDesc, &hCommandQueue);
    check_error(errno, "zeCommandQueueCreate");


    // COMMAND LIST
    ze_command_list_desc_t commandListDesc = {
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        NULL,
        computeQueueGroupOrdinal,
        0 // flags
    };

    ze_command_list_handle_t hCommandList;
    errno =  zeCommandListCreate(hContext, hDevice, &commandListDesc, &hCommandList);
    check_error(errno, "zeCommandListCreate");


    // MODULE
    unsigned char* program_file_up; size_t program_size_up;
    errno = read_from_binary(&program_file_up, &program_size_up, "shfl_up.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_down; size_t program_size_down;
    errno = read_from_binary(&program_file_down, &program_size_down, "shfl_down.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_xor; size_t program_size_xor;
    errno = read_from_binary(&program_file_xor, &program_size_xor, "shfl_xor.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_d; size_t program_size_d;
    errno = read_from_binary(&program_file_d, &program_size_d, "shfl_d.ar");
    check_error(errno, "read_from_binary");

    // OpenCL C kernel has been compiled to Gen Binary
    ze_module_desc_t moduleDescUp = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_up,
        program_file_up,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleUp;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescUp, &hModuleUp, NULL);

    ze_module_desc_t moduleDescDown = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_down,
        program_file_down,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleDown;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescDown, &hModuleDown, NULL);

    ze_module_desc_t moduleDescXor = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_xor,
        program_file_xor,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleXor;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescXor, &hModuleXor, NULL);

    ze_module_desc_t moduleDescD = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_d,
        program_file_d,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleD;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescD, &hModuleD, NULL);
    check_error(errno, "zeModuleCreate");


    // KERNEL
    ze_kernel_desc_t kernelDescUp = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_shfl_up_sync_custom"
    };

    ze_kernel_desc_t kernelDescDown = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_shfl_down_sync_custom"
    };

    ze_kernel_desc_t kernelDescD = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_shfl_sync_custom"
    };

    ze_kernel_desc_t kernelDescXor = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_shfl_xor_sync_custom"
    };

    uint32_t groupSizeX =  (uint32_t) atoi(argv[1]);
    uint32_t numGroupsX =  (uint32_t) atoi(argv[2]);
    ze_kernel_handle_t hKernelUp, hKernelDown, hKernelXor, hKernelD;

    unsigned shfl_up_sync_shared_var_arr[32];
    unsigned shfl_up_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    unsigned shfl_down_sync_shared_var_arr[32];
    unsigned shfl_down_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    unsigned shfl_sync_shared_var_arr[32];
    unsigned shfl_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    unsigned shfl_xor_sync_shared_var_arr[32];
    unsigned shfl_xor_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};


    errno = zeKernelCreate(hModuleUp, &kernelDescUp, &hKernelUp);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleDown, &kernelDescDown, &hKernelDown);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleXor, &kernelDescXor, &hKernelXor);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleD, &kernelDescD, &hKernelD);
    check_error(errno, "zeKernelCreate");

    ze_group_count_t launchArgs = { numGroupsX, 1, 1 };
    // Append launch kernel
    zeKernelSetArgumentValue(hKernelUp, 0, sizeof(unsigned)*32, shfl_up_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelUp, 1, sizeof(unsigned)*32, shfl_up_sync_updated);
    zeKernelSetGroupSize(hKernelUp, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelDown, 0, sizeof(unsigned)*32, shfl_down_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelDown, 1, sizeof(unsigned)*32, shfl_down_sync_updated);
    zeKernelSetGroupSize(hKernelDown, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelXor, 0, sizeof(unsigned)*32, shfl_xor_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelXor, 1, sizeof(unsigned)*32, shfl_xor_sync_updated);
    zeKernelSetGroupSize(hKernelXor, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelD, 0, sizeof(unsigned)*32, shfl_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelD, 1, sizeof(unsigned)*32, shfl_sync_updated);
    zeKernelSetGroupSize(hKernelD, groupSizeX, 1, 1);

    zeCommandListAppendLaunchKernel(hCommandList, hKernelUp, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelDown, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelXor, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelD, &launchArgs, NULL, 0, NULL);

    // finished appending commands (typically done on another thread)
    errno = zeCommandListClose(hCommandList);
    check_error(errno, "zeCommandListClose");

    // SUBMISSION    
    
    // Execute command list in command queue
    double total_custom_time = 0;
    struct timeval start_custom, end_custom;
    gettimeofday(&start_custom, 0);

    errno = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, NULL);
    check_error(errno, "zeCommandQueueExecuteCommandLists");

    // synchronize host and device
    errno = zeCommandQueueSynchronize(hCommandQueue, UINT32_MAX);
    check_error(errno, "zeCommandQueueSynchronize");

    gettimeofday(&end_custom, 0);

    double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
        + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
    total_custom_time += duration_custom;

    double avg_custom_up = total_custom_time / NUM_ROUNDS;

    printf("  Time to run custom shfl = %f ms\n\n", avg_custom_up);
    // printf("  Average time to run custom shfl_down_sync() = %f ms\n\n", avg_custom_down);
    // printf("  Average time to run custom shfl_sync() = %f ms\n\n", avg_custom_d);
    // printf("  Average time to run custom shfl_xor_sync() = %f ms\n\n", avg_custom_xor);


    // CLEANING
    errno = zeKernelDestroy(hKernelUp);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelDown);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelXor);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelD);
    check_error(errno, "zeKernelDestroy");

    errno = zeModuleDestroy(hModuleUp);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleDown);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleXor);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleD);
    check_error(errno, "zeModuleDestroy");

    errno =  zeCommandListDestroy(hCommandList);
    check_error(errno, "zeCommandListDestroy");

    errno = zeCommandQueueDestroy(hCommandQueue);
    check_error(errno, "zeCommandQueueDestroy");
    return 0;
}