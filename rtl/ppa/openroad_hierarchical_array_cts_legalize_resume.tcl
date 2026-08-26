set_thread_count $::env(PPA_THREADS)
read_db $::env(PPA_CTS_SEED_ODB)
puts "MLX_ARRAY_CTS_SEED_RESUME checkpoint=$::env(PPA_CTS_SEED_ODB)"
source [file normalize [file join [pwd] rtl ppa openroad_hierarchical_array_cts_buffer_legalize.tcl]]
set cts_checkpoint_tmp "$::env(PPA_CTS_ODB).tmp"
write_db $cts_checkpoint_tmp
file rename -force $cts_checkpoint_tmp $::env(PPA_CTS_ODB)
puts "MLX_ARRAY_STOP_AFTER_CTS checkpoint=$::env(PPA_CTS_ODB)"
