set_thread_count $::env(PPA_THREADS)
read_db $::env(PPA_GPL_ODB)

set block [ord::get_db_block]

set macro_grid $::env(PPA_MACRO_ORIGIN_GRID_DBU)
set aligned_macros 0
set maximum_macro_displacement 0
foreach inst [$block getInsts] {
  if {[[$inst getMaster] getName] != $::env(PPA_PE_MACRO_MASTER)} {
    continue
  }
  set location [$inst getLocation]
  set old_x [lindex $location 0]
  set old_y [lindex $location 1]
  set new_x [expr {int(floor((double($old_x) + $macro_grid / 2.0) / $macro_grid)) * $macro_grid}]
  set new_y [expr {int(floor((double($old_y) + $macro_grid / 2.0) / $macro_grid)) * $macro_grid}]
  set displacement [expr {abs($new_x - $old_x) + abs($new_y - $old_y)}]
  if {$displacement > $maximum_macro_displacement} {
    set maximum_macro_displacement $displacement
  }
  $inst setPlacementStatus PLACED
  $inst setLocation $new_x $new_y
  $inst setPlacementStatus FIRM
  if {(($new_x % $macro_grid) != 0) || (($new_y % $macro_grid) != 0)} {
    error "failed to align macro [$inst getName] to routing-track supergrid"
  }
  puts "MLX_MACRO_TRACK_LOCATION name=[$inst getName] old=$old_x,$old_y new=$new_x,$new_y displacement_dbu=$displacement"
  incr aligned_macros
}
if {$aligned_macros != $::env(PPA_MACRO_INSTANCE_COUNT)} {
  error "aligned $aligned_macros macros, expected $::env(PPA_MACRO_INSTANCE_COUNT)"
}
puts "MLX_MACRO_TRACK_ALIGNMENT macros=$aligned_macros grid_dbu=$macro_grid max_displacement_dbu=$maximum_macro_displacement"

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
  error "no full-width standard-cell rows available for channel legalization"
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

set row_records {}
foreach row [$block getRows] {
  lappend row_records [list [lindex [$row getOrigin] 1] $row]
}
set row_records [lsort -integer -index 0 $row_records]
set row_ys {}
foreach record $row_records {
  lappend row_ys [lindex $record 0]
}
set selected_row_count [llength $row_records]
if {$selected_row_count != $row_limit} {
  error "selected row count changed after tap insertion"
}

proc mlx_nearest_row_index {row_ys target_y} {
  set low 0
  set high [expr {[llength $row_ys] - 1}]
  while {$low <= $high} {
    set middle [expr {($low + $high) / 2}]
    set value [lindex $row_ys $middle]
    if {$value < $target_y} {
      set low [expr {$middle + 1}]
    } elseif {$value > $target_y} {
      set high [expr {$middle - 1}]
    } else {
      return $middle
    }
  }
  if {$high < 0} {
    return 0
  }
  if {$low >= [llength $row_ys]} {
    return [expr {[llength $row_ys] - 1}]
  }
  set lower_distance [expr {$target_y - [lindex $row_ys $high]}]
  set upper_distance [expr {[lindex $row_ys $low] - $target_y}]
  return [expr {$lower_distance <= $upper_distance ? $high : $low}]
}

array set buckets {}
array set bucket_sites {}
array set bucket_counts {}
for {set index 0} {$index < $selected_row_count} {incr index} {
  set buckets($index) {}
  set bucket_sites($index) 0
  set bucket_counts($index) 0
}

set movable_count 0
foreach inst [$block getInsts] {
  if {[$inst getPlacementStatus] != "PLACED"} {
    continue
  }
  set location [$inst getLocation]
  set original_x [lindex $location 0]
  set original_y [lindex $location 1]
  set row_index [mlx_nearest_row_index $row_ys $original_y]
  set row [lindex [lindex $row_records $row_index] 1]
  set spacing [$row getSpacing]
  set master [$inst getMaster]
  set width [$master getWidth]
  set height [$master getHeight]
  if {$height > [[$row getSite] getHeight]} {
    error "multi-height cell [$inst getName] is unsupported by channel legalization"
  }
  set width_sites [expr {int(ceil(double($width) / $spacing))}]
  lappend buckets($row_index) [list $original_x $original_y $width_sites $inst]
  incr bucket_sites($row_index) $width_sites
  incr bucket_counts($row_index)
  incr movable_count
}

set fixed_by_y [dict create]
set locked_tapcells 0
foreach inst [$block getInsts] {
  if {[[$inst getMaster] getName] != $::env(TAP_CELL_NAME)} {
    continue
  }
  set location [$inst getLocation]
  set x_min [lindex $location 0]
  set y_min [lindex $location 1]
  set x_max [expr {$x_min + [[$inst getMaster] getWidth]}]
  dict lappend fixed_by_y $y_min [list $x_min $x_max]
  incr locked_tapcells
}

proc mlx_channel_entry_compare {left right} {
  set left_y [lindex $left 1]
  set right_y [lindex $right 1]
  if {$left_y < $right_y} {
    return -1
  }
  if {$left_y > $right_y} {
    return 1
  }
  set left_x [lindex $left 0]
  set right_x [lindex $right 0]
  if {$left_x < $right_x} {
    return -1
  }
  if {$left_x > $right_x} {
    return 1
  }
  return 0
}

set target_sites [expr {int($max_site_count * $::env(PPA_CHANNEL_TARGET_UTILIZATION))}]
set spill_entries {}
set spill_sites 0
set maximum_prebalance_sites 0
set rebalanced_rows 0
for {set row_index 0} {$row_index < $selected_row_count} {incr row_index} {
  set original_sites $bucket_sites($row_index)
  if {$original_sites > $maximum_prebalance_sites} {
    set maximum_prebalance_sites $original_sites
  }
  set combined [concat $spill_entries $buckets($row_index)]
  set combined [lsort -command mlx_channel_entry_compare $combined]
  set assigned {}
  set next_spill {}
  set assigned_sites 0
  set next_spill_sites 0
  foreach entry $combined {
    set width_sites [lindex $entry 2]
    if {$assigned_sites + $width_sites <= $target_sites} {
      lappend assigned $entry
      incr assigned_sites $width_sites
    } else {
      lappend next_spill $entry
      incr next_spill_sites $width_sites
    }
  }
  if {$spill_sites > 0 || $next_spill_sites > 0} {
    incr rebalanced_rows
  }
  set buckets($row_index) $assigned
  set bucket_sites($row_index) $assigned_sites
  set bucket_counts($row_index) [llength $assigned]
  set spill_entries $next_spill
  set spill_sites $next_spill_sites
}
if {$spill_sites > 0} {
  error "channel rebalance leaves $spill_sites sites after the final row"
}
puts "MLX_CHANNEL_REBALANCE target_sites=$target_sites max_prebalance_sites=$maximum_prebalance_sites rows=$rebalanced_rows"

set placed_count 0
set maximum_displacement_dbu 0
set minimum_capacity_ratio 1.0e30
for {set row_index 0} {$row_index < $selected_row_count} {incr row_index} {
  set row [lindex [lindex $row_records $row_index] 1]
  set origin [$row getOrigin]
  set row_x [lindex $origin 0]
  set row_y [lindex $origin 1]
  set spacing [$row getSpacing]
  set row_end [expr {$row_x + [$row getSiteCount] * $spacing}]
  set intervals {}
  if {[dict exists $fixed_by_y $row_y]} {
    set intervals [lsort -integer -index 0 [dict get $fixed_by_y $row_y]]
  }
  set segments {}
  set segment_start $row_x
  foreach interval $intervals {
    set fixed_start [expr {max($row_x, [lindex $interval 0])}]
    set fixed_end [expr {min($row_end, [lindex $interval 1])}]
    if {$fixed_start > $segment_start} {
      lappend segments [list $segment_start $fixed_start]
    }
    if {$fixed_end > $segment_start} {
      set segment_start $fixed_end
    }
  }
  if {$segment_start < $row_end} {
    lappend segments [list $segment_start $row_end]
  }

  set free_sites 0
  foreach segment $segments {
    incr free_sites [expr {int(([lindex $segment 1] - [lindex $segment 0]) / $spacing)}]
  }
  set required_sites $bucket_sites($row_index)
  if {$required_sites > $free_sites} {
    error "row $row_index requires $required_sites sites but only $free_sites are free"
  }
  if {$required_sites > 0} {
    set capacity_ratio [expr {double($free_sites) / $required_sites}]
    if {$capacity_ratio < $minimum_capacity_ratio} {
      set minimum_capacity_ratio $capacity_ratio
    }
  }
  set entries [lsort -integer -index 0 $buckets($row_index)]
  set entry_count [llength $entries]
  if {$entry_count == 0} {
    continue
  }
  set gap_sites [expr {int(($free_sites - $required_sites) / ($entry_count + 1))}]
  set segment_index 0
  set cursor [lindex [lindex $segments 0] 0]
  foreach entry $entries {
    set remaining_gap $gap_sites
    while {$remaining_gap > 0} {
      if {$segment_index >= [llength $segments]} {
        error "row $row_index ran out of free segments while applying gaps"
      }
      set segment [lindex $segments $segment_index]
      set start [lindex $segment 0]
      set end [lindex $segment 1]
      if {$cursor < $start} {
        set cursor $start
      }
      set available [expr {int(($end - $cursor) / $spacing)}]
      if {$available >= $remaining_gap} {
        set cursor [expr {$cursor + $remaining_gap * $spacing}]
        set remaining_gap 0
      } else {
        set remaining_gap [expr {$remaining_gap - $available}]
        incr segment_index
        if {$segment_index < [llength $segments]} {
          set cursor [lindex [lindex $segments $segment_index] 0]
        }
      }
    }

    set width_sites [lindex $entry 2]
    set inst [lindex $entry 3]
    set placed 0
    while {!$placed} {
      if {$segment_index >= [llength $segments]} {
        error "row $row_index ran out of free segments while placing [$inst getName]"
      }
      set segment [lindex $segments $segment_index]
      set start [lindex $segment 0]
      set end [lindex $segment 1]
      if {$cursor < $start} {
        set cursor $start
      }
      set offset [expr {$cursor - $row_x}]
      set cursor [expr {$row_x + int(ceil(double($offset) / $spacing)) * $spacing}]
      set width [expr {$width_sites * $spacing}]
      if {$cursor + $width <= $end} {
        set original_x [lindex $entry 0]
        set original_y [lindex $entry 1]
        set displacement [expr {abs($cursor - $original_x) + abs($row_y - $original_y)}]
        if {$displacement > $maximum_displacement_dbu} {
          set maximum_displacement_dbu $displacement
        }
        set row_orient [$row getOrient]
        set placement_y $row_y
        if {$row_orient == "MX"} {
          set placement_y [expr {$row_y + [[$inst getMaster] getHeight]}]
        } elseif {$row_orient != "R0"} {
          error "unsupported row orientation $row_orient"
        }
        $inst setLocation $cursor $placement_y
        $inst setOrient $row_orient
        $inst setPlacementStatus FIRM
        set cursor [expr {$cursor + $width}]
        set placed 1
        incr placed_count
      } else {
        incr segment_index
        if {$segment_index < [llength $segments]} {
          set cursor [lindex [lindex $segments $segment_index] 0]
        }
      }
    }
  }
  if {(($row_index + 1) % 16) == 0} {
    puts "MLX_CHANNEL_ROW index=$row_index cells=$entry_count required_sites=$required_sites free_sites=$free_sites"
  }
}

if {$placed_count != $movable_count} {
  error "channel legalizer placed $placed_count of $movable_count cells"
}
check_placement
set legal_checkpoint_tmp "$::env(PPA_LEGAL_ODB).tmp"
write_db $legal_checkpoint_tmp
file rename -force $legal_checkpoint_tmp $::env(PPA_LEGAL_ODB)
puts "MLX_CHANNEL_LEGALIZER cells=$placed_count rows=$selected_row_count taps=$locked_tapcells removed_rows=$removed_rows removed_tapcells=$removed_tapcells max_displacement_dbu=$maximum_displacement_dbu min_capacity_ratio=$minimum_capacity_ratio checkpoint=$::env(PPA_LEGAL_ODB)"
