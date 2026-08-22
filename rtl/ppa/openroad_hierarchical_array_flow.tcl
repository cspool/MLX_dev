if {$::env(PPA_RESUME_GPL) == 1} {
  set_thread_count $::env(PPA_THREADS)
  read_db $::env(PPA_GPL_ODB)
  read_liberty $::env(PPA_LIBERTY)
  read_liberty $::env(PPA_PE_LIBERTY)
  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
  set_input_transition 0.05 [all_inputs]
  set_load 0.01 [all_outputs]
  set_wire_rc -signal -layer metal3 -clock -layer metal6
  puts "MLX_ARRAY_POST_GPL_RESUME checkpoint=$::env(PPA_GPL_ODB)"
} else {
  set_thread_count $::env(PPA_THREADS)
  read_lef $::env(PPA_TECH_LEF)
  read_lef $::env(PPA_MACRO_LEF)
  read_lef $::env(PPA_PE_LEF)
  read_liberty $::env(PPA_LIBERTY)
  read_liberty $::env(PPA_PE_LIBERTY)
  read_verilog $::env(PPA_NETLIST)
  link_design mlx_array_4x4

  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
  set_input_transition 0.05 [all_inputs]
  set_load 0.01 [all_outputs]
  set_wire_rc -signal -layer metal3 -clock -layer metal6

proc mlx_snap_macro_coordinate {value} {
  return [expr {int(floor((double($value) + 5.0) / 10.0)) * 10}]
}

initialize_floorplan \
  -site FreePDK45_38x28_10R_NP_162NW_34O \
  -utilization $::env(PPA_UTILIZATION) \
  -aspect_ratio $::env(PPA_ASPECT_RATIO) \
  -core_space 20
make_tracks
place_pins -random \
  -hor_layers {metal3 metal5 metal7 metal9} \
  -ver_layers {metal2 metal4 metal6 metal8 metal10}

set block [ord::get_db_block]
set core [$block getCoreArea]
set x_min [$core xMin]
set y_min [$core yMin]
set core_width [expr {double([$core xMax] - $x_min)}]
set core_height [expr {double([$core yMax] - $y_min)}]
set first_pe [$block findInst {GENERATE_PES\[0\].physical_pe}]
if {$first_pe == "NULL"} {
  error "missing PE macro instance 0"
}
set master [$first_pe getMaster]
set macro_width [$master getWidth]
set macro_height [$master getHeight]
set x_gap [expr {($core_width - 4.0 * $macro_width) / 5.0}]
set y_gap [expr {($core_height - 4.0 * $macro_height) / 5.0}]
if {($x_gap <= 0) || ($y_gap <= 0)} {
  error "floorplan cannot fit 4x4 PE macros"
}
for {set pe 0} {$pe < 16} {incr pe} {
  set name [format {GENERATE_PES\[%d\].physical_pe} $pe]
  set inst [$block findInst $name]
  if {$inst == "NULL"} {
    error "missing PE macro $name"
  }
  set column [expr {$pe % 4}]
  set row [expr {$pe / 4}]
  set x [mlx_snap_macro_coordinate \
    [expr {$x_min + $x_gap + $column * ($macro_width + $x_gap)}]]
  set y [mlx_snap_macro_coordinate \
    [expr {$y_min + $y_gap + $row * ($macro_height + $y_gap)}]]
  $inst setLocation $x $y
  $inst setPlacementStatus FIRM
}
puts [format "MLX_ARRAY_MACROS width_dbu=%d height_dbu=%d x_gap_dbu=%.0f y_gap_dbu=%.0f" \
  $macro_width $macro_height $x_gap $y_gap]

set movable {}
foreach inst [$block getInsts] {
  if {[$inst getMaster] != $master} {
    lappend movable $inst
  }
}
set count [llength $movable]
set columns [expr {int(ceil(sqrt($count * $core_width / $core_height)))}]
set rows [expr {int(ceil(double($count) / $columns))}]
set index 0
foreach inst $movable {
  set local_x [expr {$index % $columns}]
  set local_y [expr {$index / $columns}]
  set x [expr {int($x_min + ($local_x + 0.5) * $core_width / $columns)}]
  set y [expr {int($y_min + ($local_y + 0.5) * $core_height / $rows)}]
  $inst setLocation $x $y
  $inst setPlacementStatus PLACED
  incr index
}
puts "MLX_ARRAY_STD_SEED count=$count columns=$columns rows=$rows"
if {[info exists ::env(PPA_TAPCELL_DISTANCE_UM)]} {
  tapcell \
    -distance $::env(PPA_TAPCELL_DISTANCE_UM) \
    -tapcell_master $::env(TAP_CELL_NAME) \
    -endcap_master $::env(TAP_CELL_NAME)
} else {
  source $::env(PPA_TAPCELL_TCL)
}

  if {$::env(PPA_SKIP_INITIAL_PLACE) == 1} {
    global_placement \
      -skip_initial_place \
      -density $::env(PPA_DENSITY) \
      -bin_grid_count $::env(PPA_BIN_GRID_COUNT) \
      -overflow $::env(PPA_OVERFLOW_TARGET) \
      -init_density_penalty $::env(PPA_INIT_DENSITY_PENALTY) \
      -min_phi_coef $::env(PPA_MIN_PHI_COEF) \
      -max_phi_coef $::env(PPA_MAX_PHI_COEF)
  } else {
    global_placement \
      -density $::env(PPA_DENSITY) \
      -bin_grid_count $::env(PPA_BIN_GRID_COUNT) \
      -overflow $::env(PPA_OVERFLOW_TARGET) \
      -init_density_penalty $::env(PPA_INIT_DENSITY_PENALTY) \
      -min_phi_coef $::env(PPA_MIN_PHI_COEF) \
      -max_phi_coef $::env(PPA_MAX_PHI_COEF)
  }
  set gpl_checkpoint_tmp "$::env(PPA_GPL_ODB).tmp"
  write_db $gpl_checkpoint_tmp
  file rename -force $gpl_checkpoint_tmp $::env(PPA_GPL_ODB)
}
set block [ord::get_db_block]
if {$::env(PPA_DPL_ROW_LIMIT) > 0} {
  set removed_tapcells 0
  foreach inst [$block getInsts] {
    if {[[$inst getMaster] getName] == $::env(TAP_CELL_NAME)} {
      odb::dbInst_destroy $inst
      incr removed_tapcells
    }
  }

  set max_site_count 0
  foreach row [$block getRows] {
    set site_count [$row getSiteCount]
    if {$site_count > $max_site_count} {
      set max_site_count $site_count
    }
  }
  set full_rows {}
  foreach row [$block getRows] {
    if {[$row getSiteCount] == $max_site_count} {
      lappend full_rows [list [lindex [$row getOrigin] 1] $row]
    }
  }
  set full_rows [lsort -integer -index 0 $full_rows]
  set full_row_count [llength $full_rows]
  set row_limit [expr {min($::env(PPA_DPL_ROW_LIMIT), $full_row_count)}]
  if {$row_limit <= 0} {
    error "no full-width standard-cell rows available for top legalization"
  }
  set keep_rows {}
  for {set index 0} {$index < $row_limit} {incr index} {
    set position [expr {int(floor(($index + 0.5) * $full_row_count / $row_limit))}]
    if {$position >= $full_row_count} {
      set position [expr {$full_row_count - 1}]
    }
    set row [lindex [lindex $full_rows $position] 1]
    dict set keep_rows [$row getName] 1
  }
  set removed_rows 0
  foreach row [$block getRows] {
    if {![dict exists $keep_rows [$row getName]]} {
      odb::dbRow_destroy $row
      incr removed_rows
    }
  }
  tapcell \
    -distance $::env(PPA_TAPCELL_DISTANCE_UM) \
    -tapcell_master $::env(TAP_CELL_NAME) \
    -endcap_master $::env(TAP_CELL_NAME)
  puts "MLX_ARRAY_DPL_ROWS kept=$row_limit full_available=$full_row_count max_sites=$max_site_count removed=$removed_rows removed_tapcells=$removed_tapcells"
}
detailed_placement

clock_tree_synthesis \
  -root_buf CLKBUF_X3 \
  -buf_list {CLKBUF_X1 CLKBUF_X2 CLKBUF_X3}
set_propagated_clock [all_clocks]
detailed_placement

if {$::env(PPA_REPAIR_DESIGN) == 1} {
  repair_design -max_utilization $::env(PPA_REPAIR_MAX_UTILIZATION)
  detailed_placement
}

foreach net [[ord::get_db_block] getNets] {
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
set grt_checkpoint_tmp "$::env(PPA_GRT_ODB).tmp"
write_db $grt_checkpoint_tmp
file rename -force $grt_checkpoint_tmp $::env(PPA_GRT_ODB)
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

set dbu [$block getDbUnitsPerMicron]
set die [$block getDieArea]
set core [$block getCoreArea]
puts [format "MLX_PPA_DIE_UM %.6f %.6f" \
  [expr {double([$die xMax] - [$die xMin]) / $dbu}] \
  [expr {double([$die yMax] - [$die yMin]) / $dbu}]]
puts [format "MLX_PPA_CORE_UM %.6f %.6f" \
  [expr {double([$core xMax] - [$core xMin]) / $dbu}] \
  [expr {double([$core yMax] - [$core yMin]) / $dbu}]]
