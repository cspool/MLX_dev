read_db $::env(PPA_INPUT_ODB)

set snap_grid $::env(PPA_MANUFACTURING_GRID_DBU)
proc mlx_snap_coordinate {value grid} {
  return [expr {int(floor((double($value) + $grid / 2.0) / $grid)) * $grid}]
}

set block [ord::get_db_block]
set bpin_count 0
set box_count 0
set changed_count 0
set macro_count 0
set changed_macro_count 0
foreach bterm [$block getBTerms] {
  set old_bpins [$bterm getBPins]
  foreach old_bpin $old_bpins {
    set new_bpin [odb::dbBPin_create $bterm]
    $new_bpin setPlacementStatus [$old_bpin getPlacementStatus]
    foreach old_box [$old_bpin getBoxes] {
      if {[$old_box isVia]} {
        error "unexpected via box on BTerm [$bterm getName]"
      }
      set x_min [mlx_snap_coordinate [$old_box xMin] $snap_grid]
      set y_min [mlx_snap_coordinate [$old_box yMin] $snap_grid]
      set x_max [mlx_snap_coordinate [$old_box xMax] $snap_grid]
      set y_max [mlx_snap_coordinate [$old_box yMax] $snap_grid]
      if {$x_max <= $x_min} { set x_max [expr {$x_min + $snap_grid}] }
      if {$y_max <= $y_min} { set y_max [expr {$y_min + $snap_grid}] }
      if {($x_min != [$old_box xMin]) || ($y_min != [$old_box yMin]) ||
          ($x_max != [$old_box xMax]) || ($y_max != [$old_box yMax])} {
        incr changed_count
      }
      odb::dbBox_create \
        $new_bpin [$old_box getTechLayer] $x_min $y_min $x_max $y_max
      incr box_count
    }
    odb::dbBPin_destroy $old_bpin
    incr bpin_count
  }
}

foreach inst [$block getInsts] {
  if {[[$inst getMaster] isBlock]} {
    lassign [$inst getLocation] x y
    set snapped_x [mlx_snap_coordinate $x $snap_grid]
    set snapped_y [mlx_snap_coordinate $y $snap_grid]
    if {($snapped_x != $x) || ($snapped_y != $y)} {
      set placement_status [$inst getPlacementStatus]
      $inst setPlacementStatus PLACED
      $inst setLocation $snapped_x $snapped_y
      $inst setPlacementStatus $placement_status
      incr changed_macro_count
    }
    incr macro_count
  }
}

puts "MLX_BTERM_GRID_SNAP grid_dbu=$snap_grid bpins=$bpin_count boxes=$box_count changed=$changed_count macros=$macro_count changed_macros=$changed_macro_count"
set output_tmp "$::env(PPA_OUTPUT_ODB).tmp"
write_db $output_tmp
file rename -force $output_tmp $::env(PPA_OUTPUT_ODB)
