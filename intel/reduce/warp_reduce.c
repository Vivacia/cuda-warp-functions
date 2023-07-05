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
    unsigned char* program_file_add; size_t program_size_add;
    errno = read_from_binary(&program_file_add, &program_size_add, "reduce_add.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_min; size_t program_size_min;
    errno = read_from_binary(&program_file_min, &program_size_min, "reduce_min.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_max; size_t program_size_max;
    errno = read_from_binary(&program_file_max, &program_size_max, "reduce_max.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_and; size_t program_size_and;
    errno = read_from_binary(&program_file_and, &program_size_and, "reduce_and.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_or; size_t program_size_or;
    errno = read_from_binary(&program_file_or, &program_size_or, "reduce_or.ar");
    check_error(errno, "read_from_binary");

    unsigned char* program_file_xor; size_t program_size_xor;
    errno = read_from_binary(&program_file_xor, &program_size_xor, "reduce_xor.ar");
    check_error(errno, "read_from_binary");

    // OpenCL C kernel has been compiled to Gen Binary
    ze_module_desc_t moduleDescAdd = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_add,
        program_file_add,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleAdd;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescAdd, &hModuleAdd, NULL);

    ze_module_desc_t moduleDescMin = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_min,
        program_file_min,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleMin;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescMin, &hModuleMin, NULL);

    ze_module_desc_t moduleDescMax = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_max,
        program_file_max,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleMax;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescMax, &hModuleMax, NULL);

    ze_module_desc_t moduleDescAnd = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_and,
        program_file_and,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleAnd;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescAnd, &hModuleAnd, NULL);
    check_error(errno, "zeModuleCreate");

    ze_module_desc_t moduleDescOr = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        NULL,
        ZE_MODULE_FORMAT_NATIVE,
        program_size_or,
        program_file_or,
        NULL,
        NULL
    };
    ze_module_handle_t hModuleOr;
    errno = zeModuleCreate(hContext, hDevice, &moduleDescOr, &hModuleOr, NULL);

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


    // KERNEL
    ze_kernel_desc_t kernelDescAdd = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_reduce_add_sync_custom"
    };

    ze_kernel_desc_t kernelDescMin = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_shfl_down_sync_custom"
    };

    ze_kernel_desc_t kernelDescMax = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_reduce_max_sync_custom"
    };

    ze_kernel_desc_t kernelDescAnd = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_reduce_min_sync_custom"
    };

    ze_kernel_desc_t kernelDescOr = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_reduce_or_sync_custom"
    };

    ze_kernel_desc_t kernelDescXor = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC,
        NULL,
        0, // flags
        "test_reduce_xor_sync_custom"
    };

    uint32_t groupSizeX =  (uint32_t) atoi(argv[1]);
    uint32_t numGroupsX =  (uint32_t) atoi(argv[2]);
    ze_kernel_handle_t hKernelAdd, hKernelMin, hKernelMax,
        hKernelAnd, hKernelOr, hKernelXor;

    unsigned reduce_add_sync_shared_var_arr[32];
    unsigned reduce_add_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    unsigned test_arr_add[32];
    unsigned res_add[1];
    *res_add = 0;
    for (unsigned ii = 0; ii < 32; ii++) {
        test_arr_add[ii] = rand() % 100;
        *res_add += test_arr_add[ii];
    }

    unsigned reduce_min_sync_shared_var_arr[32];
    unsigned reduce_min_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};
    
    unsigned test_arr_min[32];
    test_arr_min[0] = rand() % 1000;
    unsigned res_min[1];
    *res_min = test_arr_min[0];
    for (unsigned ii = 1; ii < 32; ii++) {
        test_arr_min[ii] = rand() % 1000;
        if (test_arr_min[ii] < *res_min) {
            *res_min = test_arr_min[ii];
        }
    }

    unsigned reduce_max_sync_shared_var_arr[32];
    unsigned reduce_max_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};
    
    unsigned test_arr_max[32];
    test_arr_max[0] = rand() % 1000;
    unsigned res_max[1];
    *res_max = test_arr_max[0] ;
    for (int ii = 1; ii < 32; ii++) {
        test_arr_max[ii] = rand() % 1000;
        if (test_arr_max[ii] > *res_max) {
            *res_max = test_arr_max[ii];
        }
    }

    unsigned reduce_and_sync_shared_var_arr[32];
    unsigned reduce_and_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};
    
    unsigned test_arr_and[32];
    unsigned res_and[1];
    *res_and = ~0x0;
    for (int ii = 0; ii < 32; ii++) {
        test_arr_and[ii] = rand() % 1000;
        *res_and &= test_arr_and[ii];
    }

    unsigned reduce_or_sync_shared_var_arr[32];
    unsigned reduce_or_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};

    unsigned test_arr_or[32];
    unsigned res_or[1];
    *res_or = 0;
    for (int ii = 0; ii < 32; ii++) {
        test_arr_or[ii] = rand() % 1000;
        *res_or |= test_arr_or[ii];
    }
    
    unsigned reduce_xor_sync_shared_var_arr[32];
    unsigned reduce_xor_sync_updated[32] = {0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0};
    
    unsigned test_arr_xor[32];
    test_arr_xor[0] = rand() % 1000;
    unsigned res_xor[1];
    *res_xor = test_arr_xor[0];
    for (int ii = 1; ii < 32; ii++) {
        test_arr_xor[ii] = rand() % 1000;
        *res_xor ^= test_arr_xor[ii];
    }


    errno = zeKernelCreate(hModuleAdd, &kernelDescAdd, &hKernelAdd);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleMin, &kernelDescMin, &hKernelMin);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleMax, &kernelDescMax, &hKernelMax);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleAnd, &kernelDescAnd, &hKernelAnd);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleOr, &kernelDescOr, &hKernelOr);
    check_error(errno, "zeKernelCreate");

    errno = zeKernelCreate(hModuleXor, &kernelDescXor, &hKernelXor);
    check_error(errno, "zeKernelCreate");

    ze_group_count_t launchArgs = { numGroupsX, 1, 1 };
    // Append launch kernel
    zeKernelSetArgumentValue(hKernelAdd, 0, sizeof(unsigned)*32, test_arr_add);
    zeKernelSetArgumentValue(hKernelAdd, 1, sizeof(unsigned), res_add);
    zeKernelSetArgumentValue(hKernelAdd, 2, sizeof(unsigned)*32, reduce_add_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelAdd, 3, sizeof(unsigned)*32, reduce_add_sync_updated);
    zeKernelSetGroupSize(hKernelAdd, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelMin, 0, sizeof(unsigned)*32, test_arr_min);
    zeKernelSetArgumentValue(hKernelMin, 1, sizeof(unsigned), res_min);
    zeKernelSetArgumentValue(hKernelMin, 2, sizeof(unsigned)*32, reduce_min_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelMin, 3, sizeof(unsigned)*32, reduce_min_sync_updated);
    zeKernelSetGroupSize(hKernelMin, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelMax, 0, sizeof(unsigned)*32, test_arr_max);
    zeKernelSetArgumentValue(hKernelMax, 1, sizeof(unsigned), res_max);
    zeKernelSetArgumentValue(hKernelMax, 2, sizeof(unsigned)*32, reduce_max_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelMax, 3, sizeof(unsigned)*32, reduce_max_sync_updated);
    zeKernelSetGroupSize(hKernelMax, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelAnd, 0, sizeof(unsigned)*32, test_arr_and);
    zeKernelSetArgumentValue(hKernelAnd, 1, sizeof(unsigned), res_and);
    zeKernelSetArgumentValue(hKernelAnd, 2, sizeof(unsigned)*32, reduce_and_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelAnd, 3, sizeof(unsigned)*32, reduce_and_sync_updated);
    zeKernelSetGroupSize(hKernelAnd, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelOr, 0, sizeof(unsigned)*32, test_arr_or);
    zeKernelSetArgumentValue(hKernelOr, 1, sizeof(unsigned), res_or);
    zeKernelSetArgumentValue(hKernelOr, 2, sizeof(unsigned)*32, reduce_or_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelOr, 3, sizeof(unsigned)*32, reduce_or_sync_updated);
    zeKernelSetGroupSize(hKernelOr, groupSizeX, 1, 1);

    zeKernelSetArgumentValue(hKernelXor, 0, sizeof(unsigned)*32, test_arr_xor);
    zeKernelSetArgumentValue(hKernelXor, 1, sizeof(unsigned), res_xor);
    zeKernelSetArgumentValue(hKernelXor, 2, sizeof(unsigned)*32, reduce_xor_sync_shared_var_arr);
    zeKernelSetArgumentValue(hKernelXor, 3, sizeof(unsigned)*32, reduce_xor_sync_updated);
    zeKernelSetGroupSize(hKernelXor, groupSizeX, 1, 1);

    zeCommandListAppendLaunchKernel(hCommandList, hKernelAdd, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelMin, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelMax, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelAnd, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelOr, &launchArgs, NULL, 0, NULL);
    zeCommandListAppendLaunchKernel(hCommandList, hKernelXor, &launchArgs, NULL, 0, NULL);

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

    double avg_custom_reduce = total_custom_time / NUM_ROUNDS;

    printf("  Time to run custom reduce = %f ms\n\n", avg_custom_reduce);


    // CLEANING
    errno = zeKernelDestroy(hKernelAdd);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelMin);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelMax);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelAnd);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelOr);
    check_error(errno, "zeKernelDestroy");

    errno = zeKernelDestroy(hKernelXor);
    check_error(errno, "zeKernelDestroy");


    errno = zeModuleDestroy(hModuleAdd);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleMin);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleMax);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleAnd);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleOr);
    check_error(errno, "zeModuleDestroy");

    errno = zeModuleDestroy(hModuleXor);
    check_error(errno, "zeModuleDestroy");


    errno =  zeCommandListDestroy(hCommandList);
    check_error(errno, "zeCommandListDestroy");

    errno = zeCommandQueueDestroy(hCommandQueue);
    check_error(errno, "zeCommandQueueDestroy");
    return 0;
}