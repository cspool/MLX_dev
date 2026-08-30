set_thread_count $::env(PPA_THREADS)
read_lef $::env(PPA_TECH_LEF)
read_lef $::env(PPA_MACRO_LEF)
read_lef $::env(PPA_PE_LEF)
read_liberty $::env(PPA_LIBERTY)
read_liberty $::env(PPA_PE_LIBERTY)
read_verilog $::env(PPA_NETLIST)
link_design mlx_array_pe_tile

create_clock -name clk -period $::env(PPA_CLOCK_PERIOD_NS) [get_ports clk]
set_input_transition 0.05 [all_inputs]
set_load 0.01 [all_outputs]
set_wire_rc -signal -layer metal3 -clock -layer metal6

set die_width $::env(PPA_TILE_DIE_WIDTH_UM)
set die_height $::env(PPA_TILE_DIE_HEIGHT_UM)
set core_margin $::env(PPA_TILE_CORE_MARGIN_UM)
initialize_floorplan \
  -site FreePDK45_38x28_10R_NP_162NW_34O \
  -die_area [list 0 0 $die_width $die_height] \
  -core_area [list \
    $core_margin \
    $core_margin \
    [expr {$die_width - $core_margin}] \
    [expr {$die_height - $core_margin}]]
make_tracks
place_pins -random \
  -hor_layers {metal3 metal5 metal7 metal9} \
  -ver_layers {metal2 metal4 metal6 metal8 metal10}

set block [ord::get_db_block]
set pe [$block findInst physical_pe]
if {$pe == "NULL"} {
  error "missing physical_pe macro in autonomous tile"
}
set pe_master [$pe getMaster]
$pe setLocation $::env(PPA_PE_ORIGIN_X_DBU) $::env(PPA_PE_ORIGIN_Y_DBU)
$pe setPlacementStatus FIRM

set core [$block getCoreArea]
set x_min [$core xMin]
set y_min [$core yMin]
set width [expr {double([$core xMax] - $x_min)}]
set height [expr {double([$core yMax] - $y_min)}]
set movable {}
foreach inst [$block getInsts] {
  if {$inst != $pe} {
    lappend movable $inst
  }
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
puts "MLX_TILE_SEED std_cells=$count columns=$columns rows=$rows"
puts "MLX_TILE_PE_ORIGIN x_dbu=$::env(PPA_PE_ORIGIN_X_DBU) y_dbu=$::env(PPA_PE_ORIGIN_Y_DBU)"

tapcell \
  -distance $::env(PPA_TAPCELL_DISTANCE_UM) \
  -tapcell_master $::env(TAP_CELL_NAME) \
  -endcap_master $::env(TAP_CELL_NAME)

global_placement \
  -density $::env(PPA_DENSITY) \
  -bin_grid_count $::env(PPA_BIN_GRID_COUNT) \
  -overflow $::env(PPA_OVERFLOW_TARGET) \
  -init_density_penalty $::env(PPA_INIT_DENSITY_PENALTY) \
  -min_phi_coef $::env(PPA_MIN_PHI_COEF) \
  -max_phi_coef $::env(PPA_MAX_PHI_COEF)
write_db $::env(PPA_GPL_ODB)
puts "MLX_TILE_STOP_AFTER_GPL checkpoint=$::env(PPA_GPL_ODB)"
if {$::env(PPA_STOP_AFTER_GPL) == 1} {
  exit
}

detailed_placement
write_db $::env(PPA_LEGAL_ODB)
puts "MLX_TILE_LEGAL_COMPLETE checkpoint=$::env(PPA_LEGAL_ODB)"
if {$::env(PPA_STOP_AFTER_LEGAL) == 1} {
  exit
}

clock_tree_synthesis \
  -root_buf CLKBUF_X3 \
  -buf_list {CLKBUF_X1 CLKBUF_X2 CLKBUF_X3}
set_propagated_clock [all_clocks]
detailed_placement
write_db $::env(PPA_CTS_ODB)
puts "MLX_TILE_CTS_COMPLETE checkpoint=$::env(PPA_CTS_ODB)"
if {$::env(PPA_STOP_AFTER_CTS) == 1} {
  exit
}

foreach net [$block getNets] {
  set signal_type [$net getSigType]
  if {($signal_type == "GROUND") || ($signal_type == "POWER")} {
    $net setSigType SIGNAL
  }
}
set_routing_layers \
  -signal $::env(PPA_SIGNAL_ROUTING_LAYERS) \
  -clock $::env(PPA_CLOCK_ROUTING_LAYERS)
set_macro_extension $::env(PPA_MACRO_EXTENSION_GCELLS)
set grt_args [list \
  -congestion_iterations $::env(PPA_GRT_CONGESTION_ITERATIONS) \
  -critical_nets_percentage 0 \
  -guide_file $::env(PPA_GUIDE)]
if {[info exists ::env(PPA_GRT_CONGESTION_REPORT_ITER_STEP)] &&
    ($::env(PPA_GRT_CONGESTION_REPORT_ITER_STEP) > 0)} {
  lappend grt_args \
    -congestion_report_file $::env(PPA_GRT_CONGESTION_REPORT_FILE) \
    -congestion_report_iter_step $::env(PPA_GRT_CONGESTION_REPORT_ITER_STEP)
}
if {$::env(PPA_GRT_ALLOW_CONGESTION) == 1} {
  lappend grt_args -allow_congestion
}
if {[info exists ::env(PPA_GRT_VERBOSE)] && ($::env(PPA_GRT_VERBOSE) == 1)} {
  lappend grt_args -verbose
}
global_route {*}$grt_args
estimate_parasitics -global_routing
write_db $::env(PPA_GRT_ODB)
puts "MLX_TILE_STOP_AFTER_GRT checkpoint=$::env(PPA_GRT_ODB)"
if {$::env(PPA_STOP_AFTER_GRT) == 1} {
  exit
}

detailed_route \
  -droute_end_iter $::env(PPA_DROUTE_END_ITER) \
  -output_drc $::env(PPA_DRC)
extract_parasitics -ext_model_file $::env(PPA_RCX_RULES)
write_def $::env(PPA_DEF)
write_db $::env(PPA_ODB)
write_spef $::env(PPA_SPEF)
write_abstract_lef -bloat_factor 1 $::env(PPA_ABSTRACT_LEF)
write_timing_model \
  -library_name mlx_array_pe_tile_macro_lib \
  -cell_name mlx_array_pe_tile \
  $::env(PPA_TIMING_LIB)

if {[info exists ::env(PPA_VCD)] && ($::env(PPA_VCD) ne "")} {
  read_vcd -scope $::env(PPA_VCD_SCOPE) $::env(PPA_VCD)
}
puts "MLX_TILE_TIMING_BEGIN"
report_checks -path_delay max -fields {slew cap input_pins} -digits 6
puts "MLX_TILE_TIMING_END"
puts "MLX_TILE_POWER_BEGIN"
report_power
puts "MLX_TILE_POWER_END"
report_design_area

set dbu [$block getDbUnitsPerMicron]
set die [$block getDieArea]
set core [$block getCoreArea]
puts [format "MLX_TILE_DIE_UM %.6f %.6f" \
  [expr {double([$die xMax] - [$die xMin]) / $dbu}] \
  [expr {double([$die yMax] - [$die yMin]) / $dbu}]]
puts [format "MLX_TILE_CORE_UM %.6f %.6f" \
  [expr {double([$core xMax] - [$core xMin]) / $dbu}] \
  [expr {double([$core yMax] - [$core yMin]) / $dbu}]]
puts "MLX_TILE_DROUTE_COMPLETE odb=$::env(PPA_ODB) spef=$::env(PPA_SPEF) drc=$::env(PPA_DRC)"
