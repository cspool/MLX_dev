set_thread_count 1
read_db $::env(PPA_GRT_ODB)

if {![info exists ::env(MLX_DRT_STOP_AFTER_IMPORT)]
    || ($::env(MLX_DRT_STOP_AFTER_IMPORT) != 1)} {
  error "import probe requires MLX_DRT_STOP_AFTER_IMPORT=1"
}

puts "MLX_DRT_IMPORT_PROBE checkpoint=$::env(PPA_GRT_ODB)"
detailed_route -no_pin_access -droute_end_iter 1
puts "MLX_DRT_IMPORT_PROBE_COMPLETE"
