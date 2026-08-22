set_thread_count $::env(PPA_THREADS)
read_db $::env(PPA_GRT_ODB)
read_liberty $::env(PPA_LIBERTY)
read_liberty $::env(PPA_PE_LIBERTY)

create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
set_propagated_clock [all_clocks]
set_input_transition 0.05 [all_inputs]
set_load 0.01 [all_outputs]
set_wire_rc -signal -layer metal3 -clock -layer metal6

puts "MLX_ARRAY_DROUTE_RESUME checkpoint=$::env(PPA_GRT_ODB)"
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
