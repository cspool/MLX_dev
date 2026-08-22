set_thread_count $::env(PPA_THREADS)
read_db $::env(PPA_INPUT_ODB)
read_liberty $::env(PPA_LIBERTY)
if {[info exists ::env(PPA_EXTRA_LIBS)] && ($::env(PPA_EXTRA_LIBS) ne "")} {
  foreach lib $::env(PPA_EXTRA_LIBS) { read_liberty $lib }
}

if {$::env(PPA_HAS_CLOCK) == 1} {
  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
  set_propagated_clock [all_clocks]
}
set_input_transition 0.05 [all_inputs]
set_load 0.01 [all_outputs]
set_wire_rc -signal -layer metal3 -clock -layer metal6

set block [ord::get_db_block]
puts "MLX_COMPONENT_REPAIR_REROUTE_BEGIN insts=[llength [$block getInsts]]"
repair_design -max_utilization $::env(PPA_REPAIR_MAX_UTILIZATION)
detailed_placement
puts "MLX_COMPONENT_REPAIR_REROUTE_REPAIRED insts=[llength [$block getInsts]]"

foreach net [$block getNets] {
  set signal_type [$net getSigType]
  if {($signal_type == "GROUND") || ($signal_type == "POWER")} {
    puts "MLX_LOGIC_CONSTANT_NET signal=[$net getName] old_type=$signal_type new_type=SIGNAL"
    $net setSigType SIGNAL
  }
}
if {$::env(PPA_GRT_ALLOW_CONGESTION) == 1} {
  global_route \
    -congestion_iterations $::env(PPA_GRT_CONGESTION_ITERATIONS) \
    -allow_congestion \
    -guide_file $::env(PPA_GUIDE)
} else {
  global_route \
    -congestion_iterations $::env(PPA_GRT_CONGESTION_ITERATIONS) \
    -guide_file $::env(PPA_GUIDE)
}
estimate_parasitics -global_routing

set output_tmp "$::env(PPA_OUTPUT_ODB).tmp"
write_db $output_tmp
file rename -force $output_tmp $::env(PPA_OUTPUT_ODB)
puts "MLX_COMPONENT_REPAIR_REROUTE_END checkpoint=$::env(PPA_OUTPUT_ODB)"
