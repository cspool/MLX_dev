set_thread_count $::env(PPA_THREADS)
read_lef $::env(PPA_TECH_LEF)
read_lef $::env(PPA_MACRO_LEF)
read_liberty $::env(PPA_LIBERTY)
read_verilog $::env(PPA_NETLIST)
link_design $::env(PPA_TOP)

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

proc mlx_seed_physical_array {} {
  set block [ord::get_db_block]
  set core [$block getCoreArea]
  set x_min [$core xMin]
  set y_min [$core yMin]
  set width [expr {double([$core xMax] - $x_min)}]
  set height [expr {double([$core yMax] - $y_min)}]
  set tile_width [expr {$width / 4.0}]
  set tile_height [expr {$height / 4.0}]
  set columns [expr {int($::env(PPA_SEED_GRID_COLUMNS))}]
  set expected_pe [expr {int($::env(PPA_PE_CELL_COUNT))}]
  set expected_top [expr {int($::env(PPA_TOP_CELL_COUNT))}]
  set pe_rows [expr {($expected_pe + $columns - 1) / $columns}]
  set top_rows [expr {($expected_top + $columns - 1) / $columns}]
  for {set pe 0} {$pe < 16} {incr pe} {
    set pe_count($pe) 0
  }
  set top_count 0
  foreach inst [$block getInsts] {
    set name [$inst getName]
    if {[regexp {^GENERATE_PES\\\[([0-9]+)\\\]\.physical_pe/} $name -> pe]} {
      set index $pe_count($pe)
      set local_x [expr {$index % $columns}]
      set local_y [expr {$index / $columns}]
      set pe_x [expr {$pe % 4}]
      set pe_y [expr {$pe / 4}]
      set x [expr {int($x_min + $pe_x * $tile_width
          + ($local_x + 0.5) * $tile_width / $columns)}]
      set y [expr {int($y_min + $pe_y * $tile_height
          + ($local_y + 0.5) * $tile_height / $pe_rows)}]
      incr pe_count($pe)
    } else {
      set local_x [expr {$top_count % $columns}]
      set local_y [expr {$top_count / $columns}]
      set x [expr {int($x_min + ($local_x + 0.5) * $width / $columns)}]
      set y [expr {int($y_min + ($local_y + 0.5) * $height / $top_rows)}]
      incr top_count
    }
    $inst setLocation $x $y
    $inst setPlacementStatus PLACED
  }
  set observed_pe_counts {}
  for {set pe 0} {$pe < 16} {incr pe} {
    lappend observed_pe_counts $pe_count($pe)
  }
  puts "MLX_PPA_SEED_OBSERVED pe_counts=$observed_pe_counts top=$top_count total=[llength [$block getInsts]]"
  for {set pe 0} {$pe < 16} {incr pe} {
    if {$pe_count($pe) != $expected_pe} {
      error "PE $pe seed count $pe_count($pe) != $expected_pe"
    }
  }
  if {$top_count != $expected_top} {
    error "top seed count $top_count != $expected_top"
  }
  puts "MLX_PPA_SEED_COUNTS pe_each=$expected_pe top=$top_count total=[llength [$block getInsts]]"
}

if {$::env(PPA_PHYSICAL_SEED) == 1} {
  mlx_seed_physical_array
}
source $::env(PPA_TAPCELL_TCL)

set gpl_args [list \
  -density $::env(PPA_DENSITY) \
  -bin_grid_count $::env(PPA_BIN_GRID_COUNT) \
  -overflow $::env(PPA_OVERFLOW_TARGET) \
  -initial_place_max_iter $::env(PPA_INITIAL_PLACE_MAX_ITER) \
  -init_density_penalty $::env(PPA_INIT_DENSITY_PENALTY) \
  -min_phi_coef $::env(PPA_MIN_PHI_COEF) \
  -max_phi_coef $::env(PPA_MAX_PHI_COEF)]
if {$::env(PPA_PHYSICAL_SEED) == 1} {
  lappend gpl_args -skip_initial_place
}
global_placement {*}$gpl_args
if {$::env(PPA_PRE_CTS_REPAIR) == 1} {
  estimate_parasitics -placement
  repair_design
}
detailed_placement

clock_tree_synthesis \
  -root_buf CLKBUF_X3 \
  -buf_list {CLKBUF_X1 CLKBUF_X2 CLKBUF_X3}
set_propagated_clock [all_clocks]
if {$::env(PPA_POST_CTS_REPAIR) == 1} {
  estimate_parasitics -placement
  repair_timing -setup
}
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
set die_width [expr {double([$die xMax] - [$die xMin]) / $dbu}]
set die_height [expr {double([$die yMax] - [$die yMin]) / $dbu}]
set core_width [expr {double([$core xMax] - [$core xMin]) / $dbu}]
set core_height [expr {double([$core yMax] - [$core yMin]) / $dbu}]
puts [format "MLX_PPA_DIE_UM %.6f %.6f" $die_width $die_height]
puts [format "MLX_PPA_CORE_UM %.6f %.6f" $core_width $core_height]
