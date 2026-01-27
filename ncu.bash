ncu \
    --kernel-name attention_kernel \
    --metrics smsp__inst_executed_op_shfl_pred_on.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__warps_launched.sum,smsp__maximum_warps_per_active_cycle_pct \
    --export kernels.ncu-rep \
    ./test_kernels
3001458308
