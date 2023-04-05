#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <level_zero/ze_api.h>
#include "include/ze_utils.h"

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
    unsigned char* program_file; size_t program_size;
    errno = read_from_binary(&program_file, &program_size, "shfl_up.ar");
    check_error(errno, "read_from_binary");

    // OpenCL C kernel has been compiled to Gen Binary
    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size,
        program_file,
        NULL,
        NULL
    };
    ze_module_handle_t hModule;
    errno = zeModuleCreate(hContext, hDevice, &moduleDesc, &hModule, NULL);
    check_error(errno, "zeModuleCreate");


    // KERNEL
    ze_kernel_desc_t kernelDesc = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "image_scaling"
    };
    
    uint32_t groupSizeX =  (uint32_t) atoi(argv[1]);
    uint32_t numGroupsX =  (uint32_t) atoi(argv[2]);
    ze_kernel_handle_t hKernel;

    zeKernelSetGroupSize(hKernel, groupSizeX, 1, 1);

    double total_custom_time = 0;

    for (int i = 0; i < 1/*NUM_ROUNDS*/; i++) {
        struct timeval start_custom, end_custom;
        
        ze_event_handle_t event;
        zeCommandListAppendSignalEvent(hCommandList, event);        
        
        errno = zeKernelCreate(hModule, &kernelDesc, &hKernel);
        check_error(errno, "zeKernelCreate");
        
        ze_group_count_t launchArgs = { numGroupsX, 1, 1 };
        // Append launch kernel
        zeCommandListAppendLaunchKernel(hCommandList, hKernel, &launchArgs, NULL, 0, NULL);
        // finished appending commands (typically done on another thread)
        errno = zeCommandListClose(hCommandList);
        check_error(errno, "zeCommandListClose");

        gettimeofday(&start_custom, 0);
        // Execute command list in command queue
        errno = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, NULL);
        check_error(errno, "zeCommandQueueExecuteCommandLists");
        
        // synchronize host and device
        errno = zeCommandQueueSynchronize(hCommandQueue, UINT32_MAX);
        check_error(errno, "zeCommandQueueSynchronize");
        gettimeofday(&end_custom, 0);


        double duration_custom = (1000000.0*(end_custom.tv_sec-start_custom.tv_sec)
            + end_custom.tv_usec-start_custom.tv_usec)/1000.0;
        total_custom_time += duration_custom;
    }

    double avg_custom = total_custom_time / NUM_ROUNDS;

    printf("  Average time to run custom shfl_xor_sync() = %f ms\n\n", avg_custom);


    // SUBMISSION
    ze_group_count_t launchArgs = { numGroupsX, 1, 1 };
  // Append launch kernel
    zeCommandListAppendLaunchKernel(hCommandList, hKernel, &launchArgs, NULL, 0, NULL);
    // finished appending commands (typically done on another thread)
    errno = zeCommandListClose(hCommandList);
    check_error(errno, "zeCommandListClose");
    // Execute command list in command queue
    errno = zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &hCommandList, NULL);
    check_error(errno, "zeCommandQueueExecuteCommandLists");

    // synchronize host and device
    errno = zeCommandQueueSynchronize(hCommandQueue, UINT32_MAX);
    check_error(errno, "zeCommandQueueSynchronize");


    // CLEANING
    errno = zeKernelDestroy(hKernel);
    check_error(errno, "zeKernelDestroy");

    errno = zeModuleDestroy(hModule);
    check_error(errno, "zeModuleDestroy");
    
    errno =  zeCommandListDestroy(hCommandList);
    check_error(errno, "zeCommandListDestroy");

    errno = zeCommandQueueDestroy(hCommandQueue);
    check_error(errno, "zeCommandQueueDestroy");
    return 0;
}
