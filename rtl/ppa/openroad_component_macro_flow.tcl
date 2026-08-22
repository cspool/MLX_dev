set_thread_count $::env(PPA_THREADS)
read_lef $::env(PPA_TECH_LEF)
read_lef $::env(PPA_MACRO_LEF)
if {[info exists ::env(PPA_EXTRA_LEFS)] && ($::env(PPA_EXTRA_LEFS) ne "")} {
  foreach lef $::env(PPA_EXTRA_LEFS) { read_lef $lef }
}
read_liberty $::env(PPA_LIBERTY)
if {[info exists ::env(PPA_EXTRA_LIBS)] && ($::env(PPA_EXTRA_LIBS) ne "")} {
  foreach lib $::env(PPA_EXTRA_LIBS) { read_liberty $lib }
}
read_verilog $::env(PPA_NETLIST)
link_design $::env(PPA_TOP)

if {$::env(PPA_HAS_CLOCK) == 1} {
  create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
}
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
if {$::env(PPA_MULTI_LAYER_PINS) == 1} {
  # The PE exposes thousands of vector bits to the array shell.  Spread its
  # boundary terminals across every legal preferred-direction signal layer.
  place_pins -random \
    -hor_layers {metal3 metal5 metal7 metal9} \
    -ver_layers {metal2 metal4 metal6 metal8 metal10}
} else {
  place_pins -random -hor_layers metal3 -ver_layers metal2
}

set block [ord::get_db_block]
set core [$block getCoreArea]
set x_min [$core xMin]
set y_min [$core yMin]
set width [expr {double([$core xMax] - $x_min)}]
set height [expr {double([$core yMax] - $y_min)}]
if {$::env(PPA_HAS_MACROS) == 1} {
  set macros {}
  foreach inst [$block getInsts] {
    if {[[$inst getMaster] isBlock]} { lappend macros $inst }
  }
  set macro_count [llength $macros]
  if {$macro_count == 2} {
    set total_width 0
    set total_height 0
    set max_width 0
    set max_height 0
    foreach inst $macros {
      set master [$inst getMaster]
      set total_width [expr {$total_width + [$master getWidth]}]
      set total_height [expr {$total_height + [$master getHeight]}]
      set max_width [expr {max($max_width, [$master getWidth])}]
      set max_height [expr {max($max_height, [$master getHeight])}]
    }
    if {($total_width + 60000 <= $width) && ($max_height + 20000 <= $height)} {
      set macro_gap [expr {($width - $total_width) / 3.0}]
      set x [mlx_snap_macro_coordinate [expr {$x_min + $macro_gap}]]
      foreach inst $macros {
        set master [$inst getMaster]
        set y [mlx_snap_macro_coordinate [expr {$y_min + ($height - [$master getHeight]) / 2.0}]]
        $inst setLocation $x $y
        $inst setPlacementStatus FIRM
        set x [mlx_snap_macro_coordinate \
          [expr {$x + [$master getWidth] + $macro_gap}]]
      }
      set macro_columns 2
      set macro_rows 1
    } elseif {($max_width + 20000 <= $width) && ($total_height + 60000 <= $height)} {
      set macro_gap [expr {($height - $total_height) / 3.0}]
      set y [mlx_snap_macro_coordinate [expr {$y_min + $macro_gap}]]
      foreach inst $macros {
        set master [$inst getMaster]
        set x [mlx_snap_macro_coordinate [expr {$x_min + ($width - [$master getWidth]) / 2.0}]]
        $inst setLocation $x $y
        $inst setPlacementStatus FIRM
        set y [mlx_snap_macro_coordinate \
          [expr {$y + [$master getHeight] + $macro_gap}]]
      }
      set macro_columns 1
      set macro_rows 2
    } else {
      error "two macros cannot fit with 5um boundary halos and 10um channel"
    }
  } else {
    set macro_columns [expr {max(1, int(ceil(sqrt($macro_count * $width / $height))))}]
    set macro_rows [expr {max(1, int(ceil(double($macro_count) / $macro_columns)))}]
    set slot_width [expr {$width / $macro_columns}]
    set slot_height [expr {$height / $macro_rows}]
    set macro_index 0
    foreach inst $macros {
      set master [$inst getMaster]
      set master_width [$master getWidth]
      set master_height [$master getHeight]
      if {($master_width + 20000 > $slot_width) || ($master_height + 20000 > $slot_height)} {
        error "macro grid slot cannot fit [$inst getName] with 5um halo"
      }
      set local_x [expr {$macro_index % $macro_columns}]
      set local_y [expr {$macro_index / $macro_columns}]
      set x [expr {int($x_min + $local_x * $slot_width + ($slot_width - $master_width) / 2.0)}]
      set y [expr {int($y_min + $local_y * $slot_height + ($slot_height - $master_height) / 2.0)}]
      $inst setLocation $x $y
      $inst setPlacementStatus PLACED
      incr macro_index
    }
  }
  puts "MLX_COMPONENT_MACRO_SEED count=$macro_count columns=$macro_columns rows=$macro_rows"
  if {$macro_count != 2} {
    macro_placement -halo {5 5} -channel {10 10}
  }
}
set movable {}
foreach inst [$block getInsts] {
  if {![[$inst getMaster] isBlock]} { lappend movable $inst }
}
set count [llength $movable]
set columns [expr {max(1, int(ceil(sqrt($count * $width / $height))))}]
set rows [expr {max(1, int(ceil(double($count) / $columns)))}]
set index 0
foreach inst $movable {
  set local_x [expr {$index % $columns}]
  set local_y [expr {$index / $columns}]
  set x [expr {int($x_min + ($local_x + 0.5) * $width / $columns)}]
  set y [expr {int($y_min + ($local_y + 0.5) * $height / $rows)}]
  $inst setLocation $x $y
  $inst setPlacementStatus PLACED
  incr index
}
puts "MLX_COMPONENT_SEED top=$::env(PPA_TOP) std_cells=$count columns=$columns rows=$rows"
source $::env(PPA_TAPCELL_TCL)

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
detailed_placement

if {$::env(PPA_HAS_CLOCK) == 1} {
  clock_tree_synthesis \
    -root_buf CLKBUF_X3 \
    -buf_list {CLKBUF_X1 CLKBUF_X2 CLKBUF_X3}
  set_propagated_clock [all_clocks]
  detailed_placement
}

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
write_abstract_lef -bloat_factor 1 $::env(PPA_ABSTRACT_LEF)
write_timing_model \
  -library_name $::env(PPA_LIBRARY_NAME) \
  -cell_name $::env(PPA_TOP) \
  $::env(PPA_TIMING_LIB)

if {[info exists ::env(PPA_VCD)] && ($::env(PPA_VCD) ne "")} {
  read_vcd -scope $::env(PPA_VCD_SCOPE) $::env(PPA_VCD)
}
puts "MLX_COMPONENT_TIMING_BEGIN"
report_checks -path_delay max -fields {slew cap input_pins} -digits 6
puts "MLX_COMPONENT_TIMING_END"
puts "MLX_COMPONENT_POWER_BEGIN"
report_power
puts "MLX_COMPONENT_POWER_END"
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
