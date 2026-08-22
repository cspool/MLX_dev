set_thread_count $::env(PPA_THREADS)
read_lef $::env(PPA_TECH_LEF)
read_lef $::env(PPA_MACRO_LEF)
read_liberty $::env(PPA_LIBERTY)
read_verilog $::env(PPA_NETLIST)
link_design mlx_pe_top

create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
set_input_transition 0.05 [all_inputs]
set_load 0.01 [all_outputs]
set_wire_rc -signal -layer metal3 -clock -layer metal6

initialize_floorplan \
  -site FreePDK45_38x28_10R_NP_162NW_34O \
  -utilization $::env(PPA_UTILIZATION) \
  -aspect_ratio 1.0 \
  -core_space 20
make_tracks
place_pins -random -hor_layers metal3 -ver_layers metal2

set block [ord::get_db_block]
set core [$block getCoreArea]
set x_min [$core xMin]
set y_min [$core yMin]
set width [expr {double([$core xMax] - $x_min)}]
set height [expr {double([$core yMax] - $y_min)}]
set instances [$block getInsts]
set count [llength $instances]
set columns [expr {int(ceil(sqrt($count * $width / $height)))}]
set rows [expr {int(ceil(double($count) / $columns))}]
set index 0
foreach inst $instances {
  set local_x [expr {$index % $columns}]
  set local_y [expr {$index / $columns}]
  set x [expr {int($x_min + ($local_x + 0.5) * $width / $columns)}]
  set y [expr {int($y_min + ($local_y + 0.5) * $height / $rows)}]
  $inst setLocation $x $y
  $inst setPlacementStatus PLACED
  incr index
}
puts "MLX_PE_SEED count=$count columns=$columns rows=$rows"
source $::env(PPA_TAPCELL_TCL)

global_placement \
  -skip_initial_place \
  -density $::env(PPA_DENSITY) \
  -bin_grid_count $::env(PPA_BIN_GRID_COUNT) \
  -overflow $::env(PPA_OVERFLOW_TARGET) \
  -init_density_penalty $::env(PPA_INIT_DENSITY_PENALTY) \
  -min_phi_coef $::env(PPA_MIN_PHI_COEF) \
  -max_phi_coef $::env(PPA_MAX_PHI_COEF)
detailed_placement

clock_tree_synthesis \
  -root_buf CLKBUF_X3 \
  -buf_list {CLKBUF_X1 CLKBUF_X2 CLKBUF_X3}
set_propagated_clock [all_clocks]
detailed_placement

global_route -guide_file $::env(PPA_GUIDE)
estimate_parasitics -global_routing
detailed_route \
  -droute_end_iter $::env(PPA_DROUTE_END_ITER) \
  -output_drc $::env(PPA_DRC)
extract_parasitics -ext_model_file $::env(PPA_RCX_RULES)

filler_placement {FILLCELL_X32 FILLCELL_X16 FILLCELL_X8 FILLCELL_X4 FILLCELL_X2 FILLCELL_X1}
check_placement -verbose
write_def $::env(PPA_DEF)
write_db $::env(PPA_ODB)
write_spef $::env(PPA_SPEF)
write_abstract_lef -bloat_occupied_layers $::env(PPA_ABSTRACT_LEF)
write_timing_model \
  -library_name mlx_pe_macro_lib \
  -cell_name mlx_pe_top \
  $::env(PPA_TIMING_LIB)

read_vcd -scope $::env(PPA_VCD_SCOPE) $::env(PPA_VCD)
puts "MLX_PE_TIMING_BEGIN"
report_checks -path_delay max -fields {slew cap input_pins} -digits 6
puts "MLX_PE_TIMING_END"
puts "MLX_PE_POWER_BEGIN"
report_power
puts "MLX_PE_POWER_END"
report_design_area

set dbu [$block getDbUnitsPerMicron]
set die [$block getDieArea]
set core [$block getCoreArea]
puts [format "MLX_PE_DIE_UM %.6f %.6f" \
  [expr {double([$die xMax] - [$die xMin]) / $dbu}] \
  [expr {double([$die yMax] - [$die yMin]) / $dbu}]]
puts [format "MLX_PE_CORE_UM %.6f %.6f" \
  [expr {double([$core xMax] - [$core xMin]) / $dbu}] \
  [expr {double([$core yMax] - [$core yMin]) / $dbu}]]
