set_thread_count $::env(PPA_THREADS)
if {$::env(PPA_RESUME_CTS) == 1} {
  read_db $::env(PPA_CTS_ODB)
  read_liberty $::env(PPA_LIBERTY)
  read_liberty $::env(PPA_PE_LIBERTY)
  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
  set_propagated_clock [all_clocks]
  set_input_transition 0.05 [all_inputs]
  set_load 0.01 [all_outputs]
  set_wire_rc -signal -layer metal3 -clock -layer metal6
  puts "MLX_ARRAY_POST_CTS_RESUME checkpoint=$::env(PPA_CTS_ODB)"
} else {
  read_db $::env(PPA_LEGAL_ODB)
  read_liberty $::env(PPA_LIBERTY)
  read_liberty $::env(PPA_PE_LIBERTY)
  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
  set_input_transition 0.05 [all_inputs]
  set_load 0.01 [all_outputs]
  set_wire_rc -signal -layer metal3 -clock -layer metal6

  puts "MLX_ARRAY_POST_LEGAL checkpoint=$::env(PPA_LEGAL_ODB)"
  clock_tree_synthesis \
    -root_buf CLKBUF_X3 \
    -buf_list {CLKBUF_X1 CLKBUF_X2 CLKBUF_X3}
  set_propagated_clock [all_clocks]
  detailed_placement
  set cts_checkpoint_tmp "$::env(PPA_CTS_ODB).tmp"
  write_db $cts_checkpoint_tmp
  file rename -force $cts_checkpoint_tmp $::env(PPA_CTS_ODB)
  puts "MLX_ARRAY_STOP_AFTER_CTS checkpoint=$::env(PPA_CTS_ODB)"
  if {$::env(PPA_STOP_AFTER_CTS) == 1} {
    exit
  }
}

foreach net [[ord::get_db_block] getNets] {
  set signal_type [$net getSigType]
  if {($signal_type == "GROUND") || ($signal_type == "POWER")} {
    puts "MLX_LOGIC_CONSTANT_NET signal=[$net getName] old_type=$signal_type new_type=SIGNAL"
    $net setSigType SIGNAL
  }
}
set_routing_layers \
  -signal $::env(PPA_SIGNAL_ROUTING_LAYERS) \
  -clock $::env(PPA_CLOCK_ROUTING_LAYERS)
foreach {layer adjustment} $::env(PPA_LAYER_CAPACITY_ADJUSTMENTS) {
  set_global_routing_layer_adjustment $layer $adjustment
  puts "MLX_GRT_LAYER_CAPACITY_ADJUSTMENT layer=$layer adjustment=$adjustment"
}
set_macro_extension $::env(PPA_MACRO_EXTENSION_GCELLS)
set grt_args [list \
  -congestion_iterations $::env(PPA_GRT_CONGESTION_ITERATIONS) \
  -critical_nets_percentage $::env(PPA_CRITICAL_NETS_PERCENTAGE) \
  -guide_file $::env(PPA_GUIDE)]
if {$::env(PPA_GRT_ALLOW_CONGESTION) == 1} {
  lappend grt_args -allow_congestion
}
if {$::env(PPA_GRT_VERBOSE) == 1} {
  lappend grt_args -verbose
}
puts "MLX_GRT_ROUTE_ARGS $grt_args"
global_route {*}$grt_args
estimate_parasitics -global_routing
set grt_checkpoint_tmp "$::env(PPA_GRT_ODB).tmp"
write_db $grt_checkpoint_tmp
file rename -force $grt_checkpoint_tmp $::env(PPA_GRT_ODB)
puts "MLX_ARRAY_STOP_AFTER_GRT checkpoint=$::env(PPA_GRT_ODB)"
if {$::env(PPA_STOP_AFTER_GRT) == 1} {
  exit
}
detailed_route \
  -droute_end_iter $::env(PPA_DROUTE_END_ITER) \
  -output_drc $::env(PPA_DRC)
extract_parasitics -ext_model_file $::env(PPA_RCX_RULES)

filler_placement {FILLCELL_X32 FILLCELL_X16 FILLCELL_X8 FILLCELL_X4 FILLCELL_X2 FILLCELL_X1}
check_placement -verbose
write_def $::env(PPA_DEF)
write_db $::env(PPA_ODB)
write_spef $::env(PPA_SPEF)

read_vcd -scope $::env(PPA_VCD_SCOPE) $::env(PPA_VCD)
puts "MLX_PPA_TIMING_BEGIN"
report_checks -path_delay max -fields {slew cap input_pins} -digits 6
puts "MLX_PPA_TIMING_END"
puts "MLX_PPA_POWER_BEGIN"
report_power
puts "MLX_PPA_POWER_END"
report_design_area

set block [ord::get_db_block]
set dbu [$block getDbUnitsPerMicron]
set die [$block getDieArea]
set core [$block getCoreArea]
puts [format "MLX_PPA_DIE_UM %.6f %.6f" \
  [expr {double([$die xMax] - [$die xMin]) / $dbu}] \
  [expr {double([$die yMax] - [$die yMin]) / $dbu}]]
puts [format "MLX_PPA_CORE_UM %.6f %.6f" \
  [expr {double([$core xMax] - [$core xMin]) / $dbu}] \
  [expr {double([$core yMax] - [$core yMin]) / $dbu}]]
puts "MLX_ARRAY_DROUTE_COMPLETE odb=$::env(PPA_ODB) spef=$::env(PPA_SPEF) drc=$::env(PPA_DRC)"
