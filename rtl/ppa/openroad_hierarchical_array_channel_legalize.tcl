set_thread_count $::env(PPA_THREADS)
set resume_rows [expr {[info exists ::env(PPA_RESUME_ROWS)] && ($::env(PPA_RESUME_ROWS) == 1)}]
if {$resume_rows} {
  read_db $::env(PPA_ROWS_ODB)
  puts "MLX_CHANNEL_ROWS_RESUME checkpoint=$::env(PPA_ROWS_ODB)"
} else {
  read_db $::env(PPA_GPL_ODB)
}

set block [ord::get_db_block]

set macro_grid $::env(PPA_MACRO_ORIGIN_GRID_DBU)
set aligned_macros 0
set maximum_macro_displacement 0
set macro_bboxes {}
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
  set bbox [$inst getBBox]
  lappend macro_bboxes [list [$bbox xMin] [$bbox yMin] [$bbox xMax] [$bbox yMax]]
  incr aligned_macros
}
if {$aligned_macros != $::env(PPA_MACRO_INSTANCE_COUNT)} {
  error "aligned $aligned_macros macros, expected $::env(PPA_MACRO_INSTANCE_COUNT)"
}
puts "MLX_MACRO_TRACK_ALIGNMENT macros=$aligned_macros grid_dbu=$macro_grid max_displacement_dbu=$maximum_macro_displacement"

if {$resume_rows} {
  set removed_tapcells 0
  set removed_rows 0
  set requested_row_count $::env(PPA_DPL_ROW_LIMIT)
  set row_limit $requested_row_count
} else {
  set removed_tapcells 0
  foreach inst [$block getInsts] {
    if {[[$inst getMaster] getName] == $::env(TAP_CELL_NAME)} {
      odb::dbInst_destroy $inst
      incr removed_tapcells
    }
  }

  set rows_by_y_before_tap [dict create]
  set physical_row_ys {}
  foreach row [$block getRows] {
    set row_y [lindex [$row getOrigin] 1]
    if {![dict exists $rows_by_y_before_tap $row_y]} {
      lappend physical_row_ys $row_y
    }
    dict lappend rows_by_y_before_tap $row_y $row
  }
  set physical_row_ys [lsort -integer $physical_row_ys]
  set physical_row_count [llength $physical_row_ys]
  set requested_row_count $::env(PPA_DPL_ROW_LIMIT)
  set row_limit [expr {$requested_row_count > 0 ? min($requested_row_count, $physical_row_count) : $physical_row_count}]
  if {$row_limit <= 0} {
    error "no physical standard-cell rows available for channel legalization"
  }
  set keep_row_ys [dict create]
  for {set index 0} {$index < $row_limit} {incr index} {
    set position [expr {int(floor(($index + 0.5) * $physical_row_count / $row_limit))}]
    if {$position >= $physical_row_count} {
      set position [expr {$physical_row_count - 1}]
    }
    dict set keep_row_ys [lindex $physical_row_ys $position] 1
  }
  set removed_rows 0
  foreach row [$block getRows] {
    set row_y [lindex [$row getOrigin] 1]
    if {![dict exists $keep_row_ys $row_y]} {
      odb::dbRow_destroy $row
      incr removed_rows
    }
  }

  tapcell \
    -distance $::env(PPA_TAPCELL_DISTANCE_UM) \
    -tapcell_master $::env(TAP_CELL_NAME) \
    -endcap_master $::env(TAP_CELL_NAME)
}

proc mlx_row_record_compare {left right} {
  set left_y [lindex $left 0]
  set right_y [lindex $right 0]
  if {$left_y < $right_y} { return -1 }
  if {$left_y > $right_y} { return 1 }
  set left_x [lindex $left 1]
  set right_x [lindex $right 1]
  if {$left_x < $right_x} { return -1 }
  if {$left_x > $right_x} { return 1 }
  return 0
}

set row_records {}
foreach row [$block getRows] {
  set origin [$row getOrigin]
  lappend row_records [list [lindex $origin 1] [lindex $origin 0] $row]
}
set row_records [lsort -command mlx_row_record_compare $row_records]
set row_ys {}
set row_indices_by_y [dict create]
set max_site_count 0
for {set row_index 0} {$row_index < [llength $row_records]} {incr row_index} {
  set row [lindex [lindex $row_records $row_index] 2]
  if {[$row getSiteCount] > $max_site_count} {
    set max_site_count [$row getSiteCount]
  }
  set row_y [lindex [lindex $row_records $row_index] 0]
  if {![dict exists $row_indices_by_y $row_y]} {
    lappend row_ys $row_y
  }
  dict lappend row_indices_by_y $row_y $row_index
}
set full_width_row_ys {}
foreach row_y $row_ys {
  foreach row_index [dict get $row_indices_by_y $row_y] {
    set row [lindex [lindex $row_records $row_index] 2]
    if {[$row getSiteCount] == $max_site_count} {
      lappend full_width_row_ys $row_y
      break
    }
  }
}
set selected_row_count [llength $row_records]
set selected_physical_row_count [llength $row_ys]
if {$selected_physical_row_count != $row_limit} {
  error "selected physical row count changed after tap insertion"
}
puts "MLX_CHANNEL_ROW_SELECTION physical_rows=$selected_physical_row_count row_segments=$selected_row_count removed_segments=$removed_rows"

set audited_row_segments 0
set previous_row_y -1
set previous_row_end -1
foreach row_record $row_records {
  set row_y [lindex $row_record 0]
  set row_x [lindex $row_record 1]
  set row [lindex $row_record 2]
  set row_end [expr {$row_x + [$row getSiteCount] * [$row getSpacing]}]
  set row_top [expr {$row_y + [[$row getSite] getHeight]}]
  if {$row_y == $previous_row_y && $row_x < $previous_row_end} {
    error "overlapping database row segments at y=$row_y x=$row_x"
  }
  foreach macro_bbox $macro_bboxes {
    if {$row_x < [lindex $macro_bbox 2] && $row_end > [lindex $macro_bbox 0] &&
        $row_y < [lindex $macro_bbox 3] && $row_top > [lindex $macro_bbox 1]} {
      error "database row segment at $row_x,$row_y crosses a PE macro"
    }
  }
  set previous_row_y $row_y
  set previous_row_end $row_end
  incr audited_row_segments
}
puts "MLX_CHANNEL_ROW_AUDIT nonoverlapping_macro_clear_segments=$audited_row_segments"
if {[info exists ::env(PPA_ROWS_ODB)] && ($::env(PPA_ROWS_ODB) ne "")} {
  write_db $::env(PPA_ROWS_ODB)
  puts "MLX_CHANNEL_ROWS_CHECKPOINT checkpoint=$::env(PPA_ROWS_ODB)"
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
set full_width_escape_count 0
foreach inst [$block getInsts] {
  if {[$inst getPlacementStatus] != "PLACED"} {
    continue
  }
  set location [$inst getLocation]
  set original_x [lindex $location 0]
  set original_y [lindex $location 1]
  set master [$inst getMaster]
  set width [$master getWidth]
  set height [$master getHeight]
  set nearest_y_index [mlx_nearest_row_index $row_ys $original_y]
  set nearest_y [lindex $row_ys $nearest_y_index]
  set candidate_ys [list $nearest_y]
  set nearest_y_has_x_fit 0
  foreach candidate_index [dict get $row_indices_by_y $nearest_y] {
    set candidate_row [lindex [lindex $row_records $candidate_index] 2]
    set candidate_origin [$candidate_row getOrigin]
    set candidate_x [lindex $candidate_origin 0]
    set candidate_end [expr {$candidate_x + [$candidate_row getSiteCount] * [$candidate_row getSpacing]}]
    if {$candidate_x <= $original_x && $candidate_end >= $original_x + $width} {
      set nearest_y_has_x_fit 1
      break
    }
  }
  if {[llength $full_width_row_ys] > 0} {
    set full_width_y_index [mlx_nearest_row_index $full_width_row_ys $original_y]
    set full_width_y [lindex $full_width_row_ys $full_width_y_index]
    if {!$nearest_y_has_x_fit && $full_width_y != $nearest_y} {
      lappend candidate_ys $full_width_y
    }
  }
  set row_index -1
  set selected_y $nearest_y
  set best_distance 0x7fffffffffffffff
  foreach candidate_y $candidate_ys {
    set y_distance [expr {abs($candidate_y - $original_y)}]
    foreach candidate_index [dict get $row_indices_by_y $candidate_y] {
      set candidate_row [lindex [lindex $row_records $candidate_index] 2]
      set candidate_origin [$candidate_row getOrigin]
      set candidate_x [lindex $candidate_origin 0]
      set candidate_end [expr {$candidate_x + [$candidate_row getSiteCount] * [$candidate_row getSpacing]}]
      if {$original_x < $candidate_x} {
        set x_distance [expr {$candidate_x - $original_x}]
      } elseif {$original_x + $width > $candidate_end} {
        set x_distance [expr {$original_x + $width - $candidate_end}]
      } else {
        set x_distance 0
      }
      set distance [expr {$x_distance + $y_distance}]
      if {$distance < $best_distance} {
        set best_distance $distance
        set row_index $candidate_index
        set selected_y $candidate_y
      }
    }
  }
  if {$row_index < 0} {
    error "no row segment found for [$inst getName] near y=$nearest_y"
  }
  if {$selected_y != $nearest_y} {
    incr full_width_escape_count
  }
  set row [lindex [lindex $row_records $row_index] 2]
  set spacing [$row getSpacing]
  if {$height > [[$row getSite] getHeight]} {
    error "multi-height cell [$inst getName] is unsupported by channel legalization"
  }
  set width_sites [expr {int(ceil(double($width) / $spacing))}]
  lappend buckets($row_index) [list $original_x $original_y $width_sites $inst]
  incr bucket_sites($row_index) $width_sites
  incr bucket_counts($row_index)
  incr movable_count
}
puts "MLX_CHANNEL_ASSIGNMENT cells=$movable_count full_width_y_escapes=$full_width_escape_count"

set fixed_by_y [dict create]
set locked_tapcells 0
foreach inst [$block getInsts] {
  if {[[$inst getMaster] getName] != $::env(TAP_CELL_NAME)} {
    continue
  }
  set bbox [$inst getBBox]
  set x_min [$bbox xMin]
  set y_min [$bbox yMin]
  set x_max [$bbox xMax]
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

proc mlx_channel_width_compare {left right} {
  set left_width [lindex $left 2]
  set right_width [lindex $right 2]
  if {$left_width > $right_width} {
    return -1
  }
  if {$left_width < $right_width} {
    return 1
  }
  return [mlx_channel_entry_compare $left $right]
}

set spill_entries {}
set spill_sites 0
set maximum_prebalance_sites 0
set minimum_target_sites 0x7fffffff
set maximum_target_sites 0
set rebalanced_rows 0
for {set row_index 0} {$row_index < $selected_row_count} {incr row_index} {
  set row [lindex [lindex $row_records $row_index] 2]
  set target_sites [expr {max(1, int([$row getSiteCount] * $::env(PPA_CHANNEL_TARGET_UTILIZATION)))}]
  if {$target_sites < $minimum_target_sites} { set minimum_target_sites $target_sites }
  if {$target_sites > $maximum_target_sites} { set maximum_target_sites $target_sites }
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
puts "MLX_CHANNEL_REBALANCE min_target_sites=$minimum_target_sites max_target_sites=$maximum_target_sites max_prebalance_sites=$maximum_prebalance_sites segments=$rebalanced_rows"

set placed_count 0
set maximum_displacement_dbu 0
set maximum_x_displacement_dbu 0
set maximum_y_displacement_dbu 0
set total_displacement_dbu 0
set minimum_capacity_ratio 1.0e30
set backward_compactions 0
set site_aligned_checks 0
set segment_containment_checks 0
set standard_cell_nonoverlap_checks 0
for {set row_index 0} {$row_index < $selected_row_count} {incr row_index} {
  set row [lindex [lindex $row_records $row_index] 2]
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
    set interval_start [lindex $interval 0]
    set interval_end [lindex $interval 1]
    if {$interval_end <= $row_x || $interval_start >= $row_end} {
      continue
    }
    set fixed_start [expr {max($row_x, $interval_start)}]
    set fixed_end [expr {min($row_end, $interval_end)}]
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

  set aligned_segments {}
  foreach segment $segments {
    set start_offset [expr {[lindex $segment 0] - $row_x}]
    set end_offset [expr {[lindex $segment 1] - $row_x}]
    set aligned_start [expr {$row_x + int(ceil(double($start_offset) / $spacing)) * $spacing}]
    set aligned_end [expr {$row_x + int(floor(double($end_offset) / $spacing)) * $spacing}]
    if {$aligned_end > $aligned_start} {
      lappend aligned_segments [list $aligned_start $aligned_end]
    }
  }
  set segments $aligned_segments

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
  array unset segment_entries
  array unset segment_sites
  array unset segment_capacity_sites
  for {set segment_index 0} {$segment_index < [llength $segments]} {incr segment_index} {
    set segment [lindex $segments $segment_index]
    set segment_entries($segment_index) {}
    set segment_sites($segment_index) 0
    set segment_capacity_sites($segment_index) [expr {int(([lindex $segment 1] - [lindex $segment 0]) / $spacing)}]
  }
  foreach entry [lsort -command mlx_channel_width_compare $entries] {
    set width_sites [lindex $entry 2]
    set inst [lindex $entry 3]
    set original_x [lindex $entry 0]
    set width [expr {$width_sites * $spacing}]
    set best_segment_index -1
    set best_distance 0x7fffffffffffffff
    for {set segment_index 0} {$segment_index < [llength $segments]} {incr segment_index} {
      if {$segment_sites($segment_index) + $width_sites > $segment_capacity_sites($segment_index)} {
        continue
      }
      set segment [lindex $segments $segment_index]
      set start [lindex $segment 0]
      set end [lindex $segment 1]
      set maximum_x [expr {$end - $width}]
      if {$original_x < $start} {
        set distance [expr {$start - $original_x}]
      } elseif {$original_x > $maximum_x} {
        set distance [expr {$original_x - $maximum_x}]
      } else {
        set distance 0
      }
      if {$distance < $best_distance} {
        set best_distance $distance
        set best_segment_index $segment_index
      }
    }
    if {$best_segment_index < 0} {
      error "no free segment has $width_sites sites for [$inst getName] in row $row_index"
    }
    lappend segment_entries($best_segment_index) $entry
    incr segment_sites($best_segment_index) $width_sites
  }

  set row_orient [$row getOrient]
  if {$row_orient != "R0" && $row_orient != "MX"} {
    error "unsupported row orientation $row_orient"
  }
  for {set segment_index 0} {$segment_index < [llength $segments]} {incr segment_index} {
    set assigned_entries [lsort -integer -index 0 $segment_entries($segment_index)]
    if {[llength $assigned_entries] == 0} {
      continue
    }
    set segment [lindex $segments $segment_index]
    set start [lindex $segment 0]
    set end [lindex $segment 1]
    set cursor $start
    set positions {}
    foreach entry $assigned_entries {
      set width [expr {[lindex $entry 2] * $spacing}]
      set original_x [lindex $entry 0]
      set maximum_x [expr {$end - $width}]
      set desired_offset [expr {$original_x - $row_x}]
      set desired_x [expr {$row_x + int(floor((double($desired_offset) + $spacing / 2.0) / $spacing)) * $spacing}]
      set desired_x [expr {max($start, min($maximum_x, $desired_x))}]
      set placement_x [expr {max($cursor, $desired_x)}]
      lappend positions $placement_x
      set cursor [expr {$placement_x + $width}]
    }
    if {$cursor > $end} {
      incr backward_compactions
      set next_x $end
      for {set entry_index [expr {[llength $assigned_entries] - 1}]} {$entry_index >= 0} {incr entry_index -1} {
        set entry [lindex $assigned_entries $entry_index]
        set width [expr {[lindex $entry 2] * $spacing}]
        set placement_x [expr {min([lindex $positions $entry_index], $next_x - $width)}]
        lset positions $entry_index $placement_x
        set next_x $placement_x
      }
      if {[lindex $positions 0] < $start} {
        error "backward compaction crossed segment start in row $row_index segment $segment_index"
      }
    }
    set previous_cell_end $start
    for {set entry_index 0} {$entry_index < [llength $assigned_entries]} {incr entry_index} {
      set entry [lindex $assigned_entries $entry_index]
      set inst [lindex $entry 3]
      set placement_x [lindex $positions $entry_index]
      set master [$inst getMaster]
      set actual_width [$master getWidth]
      set actual_height [$master getHeight]
      if {(($placement_x - $row_x) % $spacing) != 0} {
        error "off-site placement for [$inst getName] in row $row_index"
      }
      incr site_aligned_checks
      if {$placement_x < $start || $placement_x + $actual_width > $end ||
          $actual_height > [[$row getSite] getHeight]} {
        error "placement for [$inst getName] escapes row $row_index segment $segment_index"
      }
      incr segment_containment_checks
      if {$placement_x < $previous_cell_end} {
        error "standard-cell overlap for [$inst getName] in row $row_index segment $segment_index"
      }
      set previous_cell_end [expr {$placement_x + $actual_width}]
      incr standard_cell_nonoverlap_checks
      $inst setOrient $row_orient
      $inst setLocation $placement_x $row_y
      $inst setPlacementStatus PLACED
      incr placed_count
    }
  }
  if {(($row_index + 1) % 512) == 0} {
    puts "MLX_CHANNEL_ROW index=$row_index cells=$entry_count required_sites=$required_sites free_sites=$free_sites"
  }
}

if {$placed_count != $movable_count} {
  error "channel legalizer placed $placed_count of $movable_count cells"
}
puts "MLX_CHANNEL_1D_LEGALIZATION backward_compactions=$backward_compactions"
puts "MLX_CHANNEL_CONSTRUCTIVE_AUDIT cells=$placed_count site_aligned=$site_aligned_checks segment_contained=$segment_containment_checks standard_nonoverlap=$standard_cell_nonoverlap_checks row_segments=$audited_row_segments"
if {[info exists ::env(PPA_SEED_ODB)] && ($::env(PPA_SEED_ODB) ne "")} {
  write_db $::env(PPA_SEED_ODB)
  puts "MLX_CHANNEL_SEED_CHECKPOINT checkpoint=$::env(PPA_SEED_ODB)"
}

set maximum_displacement_dbu 0
set maximum_x_displacement_dbu 0
set maximum_y_displacement_dbu 0
set total_displacement_dbu 0
for {set row_index 0} {$row_index < $selected_row_count} {incr row_index} {
  foreach entry $buckets($row_index) {
    set original_x [lindex $entry 0]
    set original_y [lindex $entry 1]
    set inst [lindex $entry 3]
    set location [$inst getLocation]
    set x_displacement [expr {abs([lindex $location 0] - $original_x)}]
    set y_displacement [expr {abs([lindex $location 1] - $original_y)}]
    set displacement [expr {$x_displacement + $y_displacement}]
    if {$displacement > $maximum_displacement_dbu} {
      set maximum_displacement_dbu $displacement
    }
    if {$x_displacement > $maximum_x_displacement_dbu} {
      set maximum_x_displacement_dbu $x_displacement
    }
    if {$y_displacement > $maximum_y_displacement_dbu} {
      set maximum_y_displacement_dbu $y_displacement
    }
    set total_displacement_dbu [expr {$total_displacement_dbu + $displacement}]
  }
}
if {[info exists ::env(PPA_PRECHECK_ODB)] && ($::env(PPA_PRECHECK_ODB) ne "")} {
  write_db $::env(PPA_PRECHECK_ODB)
  puts "MLX_CHANNEL_PRECHECK checkpoint=$::env(PPA_PRECHECK_ODB)"
}
foreach inst [$block getInsts] {
  if {[$inst getPlacementStatus] == "PLACED" && ![[$inst getMaster] isBlock]} {
    $inst setPlacementStatus FIRM
  }
}
set legal_checkpoint_tmp "$::env(PPA_LEGAL_ODB).tmp"
write_db $legal_checkpoint_tmp
file rename -force $legal_checkpoint_tmp $::env(PPA_LEGAL_ODB)
puts "MLX_CHANNEL_LEGALIZER cells=$placed_count rows=$selected_physical_row_count row_segments=$selected_row_count taps=$locked_tapcells removed_rows=$removed_rows removed_tapcells=$removed_tapcells max_displacement_dbu=$maximum_displacement_dbu min_capacity_ratio=$minimum_capacity_ratio checkpoint=$::env(PPA_LEGAL_ODB)"
puts [format "MLX_CHANNEL_LOCALITY max_x_displacement_dbu=%d max_y_displacement_dbu=%d average_displacement_dbu=%.6f" \
  $maximum_x_displacement_dbu $maximum_y_displacement_dbu \
  [expr {$placed_count > 0 ? double($total_displacement_dbu) / $placed_count : 0.0}]]
