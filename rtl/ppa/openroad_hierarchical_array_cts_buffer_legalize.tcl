# Legalize only the buffers inserted by CTS without constructing OpenDP's
# full-chip site bitmap.  The pre-CTS standard cells and tapcells are FIRM.

proc mlx_cts_row_compare {left right} {
  foreach index {0 1} {
    set left_value [lindex $left $index]
    set right_value [lindex $right $index]
    if {$left_value < $right_value} { return -1 }
    if {$left_value > $right_value} { return 1 }
  }
  return 0
}

proc mlx_cts_entry_compare {left right} {
  foreach index {1 0} {
    set left_value [lindex $left $index]
    set right_value [lindex $right $index]
    if {$left_value < $right_value} { return -1 }
    if {$left_value > $right_value} { return 1 }
  }
  return 0
}

proc mlx_cts_width_compare {left right} {
  set left_width [lindex $left 2]
  set right_width [lindex $right 2]
  if {$left_width > $right_width} { return -1 }
  if {$left_width < $right_width} { return 1 }
  return [mlx_cts_entry_compare $left $right]
}

proc mlx_cts_nearest_index {values target} {
  set low 0
  set high [expr {[llength $values] - 1}]
  while {$low <= $high} {
    set middle [expr {($low + $high) / 2}]
    set value [lindex $values $middle]
    if {$value < $target} {
      set low [expr {$middle + 1}]
    } elseif {$value > $target} {
      set high [expr {$middle - 1}]
    } else {
      return $middle
    }
  }
  if {$high < 0} { return 0 }
  if {$low >= [llength $values]} { return [expr {[llength $values] - 1}] }
  set lower_distance [expr {$target - [lindex $values $high]}]
  set upper_distance [expr {[lindex $values $low] - $target}]
  return [expr {$lower_distance <= $upper_distance ? $high : $low}]
}

set cts_block [ord::get_db_block]
set cts_row_records {}
foreach row [$cts_block getRows] {
  set origin [$row getOrigin]
  lappend cts_row_records [list [lindex $origin 1] [lindex $origin 0] $row]
}
set cts_row_records [lsort -command mlx_cts_row_compare $cts_row_records]
set cts_row_count [llength $cts_row_records]
set cts_row_ys {}
set cts_row_indices_by_y [dict create]
set cts_max_site_count 0
for {set row_index 0} {$row_index < $cts_row_count} {incr row_index} {
  set row_record [lindex $cts_row_records $row_index]
  set row_y [lindex $row_record 0]
  set row [lindex $row_record 2]
  if {![dict exists $cts_row_indices_by_y $row_y]} {
    lappend cts_row_ys $row_y
  }
  dict lappend cts_row_indices_by_y $row_y $row_index
  if {[$row getSiteCount] > $cts_max_site_count} {
    set cts_max_site_count [$row getSiteCount]
  }
}
set cts_full_width_row_ys {}
foreach row_y $cts_row_ys {
  foreach row_index [dict get $cts_row_indices_by_y $row_y] {
    set row [lindex [lindex $cts_row_records $row_index] 2]
    if {[$row getSiteCount] == $cts_max_site_count} {
      lappend cts_full_width_row_ys $row_y
      break
    }
  }
}

array set cts_buckets {}
array set cts_bucket_sites {}
for {set row_index 0} {$row_index < $cts_row_count} {incr row_index} {
  set cts_buckets($row_index) {}
  set cts_bucket_sites($row_index) 0
}
set cts_fixed_by_y [dict create]
set cts_buffer_count 0
set cts_fixed_cell_count 0
foreach inst [$cts_block getInsts] {
  set master [$inst getMaster]
  if {[$master isBlock]} {
    continue
  }
  set status [$inst getPlacementStatus]
  if {$status == "FIRM" || $status == "LOCKED"} {
    set bbox [$inst getBBox]
    dict lappend cts_fixed_by_y [$bbox yMin] [list [$bbox xMin] [$bbox xMax]]
    incr cts_fixed_cell_count
    continue
  }
  if {$status != "PLACED"} {
    error "unexpected non-fixed placement status $status for [$inst getName] after CTS"
  }
  set location [$inst getLocation]
  set original_x [lindex $location 0]
  set original_y [lindex $location 1]
  set width [$master getWidth]
  set height [$master getHeight]
  set nearest_y [lindex $cts_row_ys [mlx_cts_nearest_index $cts_row_ys $original_y]]
  set candidate_ys [list $nearest_y]
  set nearest_y_has_x_fit 0
  foreach candidate_index [dict get $cts_row_indices_by_y $nearest_y] {
    set candidate_row [lindex [lindex $cts_row_records $candidate_index] 2]
    set candidate_origin [$candidate_row getOrigin]
    set candidate_x [lindex $candidate_origin 0]
    set candidate_end [expr {$candidate_x + [$candidate_row getSiteCount] * [$candidate_row getSpacing]}]
    if {$candidate_x <= $original_x && $candidate_end >= $original_x + $width &&
        $height <= [[$candidate_row getSite] getHeight]} {
      set nearest_y_has_x_fit 1
      break
    }
  }
  if {!$nearest_y_has_x_fit && [llength $cts_full_width_row_ys] > 0} {
    set full_width_y [lindex $cts_full_width_row_ys [mlx_cts_nearest_index $cts_full_width_row_ys $original_y]]
    if {$full_width_y != $nearest_y} {
      lappend candidate_ys $full_width_y
    }
  }
  set best_row_index -1
  set best_distance 0x7fffffffffffffff
  foreach candidate_y $candidate_ys {
    set y_distance [expr {abs($candidate_y - $original_y)}]
    foreach candidate_index [dict get $cts_row_indices_by_y $candidate_y] {
      set candidate_row [lindex [lindex $cts_row_records $candidate_index] 2]
      if {$height > [[$candidate_row getSite] getHeight]} {
        continue
      }
      set candidate_origin [$candidate_row getOrigin]
      set candidate_x [lindex $candidate_origin 0]
      set candidate_end [expr {$candidate_x + [$candidate_row getSiteCount] * [$candidate_row getSpacing]}]
      if {$candidate_end - $candidate_x < $width} {
        continue
      }
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
        set best_row_index $candidate_index
      }
    }
  }
  if {$best_row_index < 0} {
    error "no row segment found for CTS buffer [$inst getName]"
  }
  set row [lindex [lindex $cts_row_records $best_row_index] 2]
  set width_sites [expr {int(ceil(double($width) / [$row getSpacing]))}]
  lappend cts_buckets($best_row_index) [list $original_x $original_y $width_sites $inst]
  incr cts_bucket_sites($best_row_index) $width_sites
  incr cts_buffer_count
}
puts "MLX_CTS_BUFFER_ASSIGNMENT buffers=$cts_buffer_count fixed_cells=$cts_fixed_cell_count physical_rows=[llength $cts_row_ys] row_segments=$cts_row_count"

set cts_placed_count 0
set cts_backward_compactions 0
set cts_site_aligned_checks 0
set cts_segment_containment_checks 0
set cts_fixed_clear_checks 0
set cts_standard_nonoverlap_checks 0
set cts_maximum_displacement_dbu 0
set cts_total_displacement_dbu 0
for {set row_index 0} {$row_index < $cts_row_count} {incr row_index} {
  set entries $cts_buckets($row_index)
  if {[llength $entries] == 0} {
    continue
  }
  set row [lindex [lindex $cts_row_records $row_index] 2]
  set origin [$row getOrigin]
  set row_x [lindex $origin 0]
  set row_y [lindex $origin 1]
  set spacing [$row getSpacing]
  set row_end [expr {$row_x + [$row getSiteCount] * $spacing}]
  set intervals {}
  if {[dict exists $cts_fixed_by_y $row_y]} {
    set intervals [lsort -integer -index 0 [dict get $cts_fixed_by_y $row_y]]
  }
  set raw_segments {}
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
      lappend raw_segments [list $segment_start $fixed_start]
    }
    if {$fixed_end > $segment_start} {
      set segment_start $fixed_end
    }
  }
  if {$segment_start < $row_end} {
    lappend raw_segments [list $segment_start $row_end]
  }
  set segments {}
  foreach segment $raw_segments {
    set start_offset [expr {[lindex $segment 0] - $row_x}]
    set end_offset [expr {[lindex $segment 1] - $row_x}]
    set start [expr {$row_x + int(ceil(double($start_offset) / $spacing)) * $spacing}]
    set end [expr {$row_x + int(floor(double($end_offset) / $spacing)) * $spacing}]
    if {$end > $start} {
      lappend segments [list $start $end]
    }
  }
  array unset cts_segment_entries
  array unset cts_segment_sites
  array unset cts_segment_capacity
  set free_sites 0
  for {set segment_index 0} {$segment_index < [llength $segments]} {incr segment_index} {
    set segment [lindex $segments $segment_index]
    set capacity [expr {int(([lindex $segment 1] - [lindex $segment 0]) / $spacing)}]
    set cts_segment_entries($segment_index) {}
    set cts_segment_sites($segment_index) 0
    set cts_segment_capacity($segment_index) $capacity
    incr free_sites $capacity
  }
  if {$cts_bucket_sites($row_index) > $free_sites} {
    error "CTS row $row_index requires $cts_bucket_sites($row_index) sites but only $free_sites are free"
  }
  foreach entry [lsort -command mlx_cts_width_compare $entries] {
    set width_sites [lindex $entry 2]
    set width [expr {$width_sites * $spacing}]
    set original_x [lindex $entry 0]
    set inst [lindex $entry 3]
    set best_segment_index -1
    set best_distance 0x7fffffffffffffff
    for {set segment_index 0} {$segment_index < [llength $segments]} {incr segment_index} {
      if {$cts_segment_sites($segment_index) + $width_sites > $cts_segment_capacity($segment_index)} {
        continue
      }
      set segment [lindex $segments $segment_index]
      set start [lindex $segment 0]
      set maximum_x [expr {[lindex $segment 1] - $width}]
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
      error "no fixed-clear segment can hold CTS buffer [$inst getName] in row $row_index"
    }
    lappend cts_segment_entries($best_segment_index) $entry
    incr cts_segment_sites($best_segment_index) $width_sites
  }
  set row_orient [$row getOrient]
  if {$row_orient != "R0" && $row_orient != "MX"} {
    error "unsupported CTS row orientation $row_orient"
  }
  for {set segment_index 0} {$segment_index < [llength $segments]} {incr segment_index} {
    set assigned_entries [lsort -integer -index 0 $cts_segment_entries($segment_index)]
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
      incr cts_backward_compactions
      set next_x $end
      for {set entry_index [expr {[llength $assigned_entries] - 1}]} {$entry_index >= 0} {incr entry_index -1} {
        set entry [lindex $assigned_entries $entry_index]
        set width [expr {[lindex $entry 2] * $spacing}]
        set placement_x [expr {min([lindex $positions $entry_index], $next_x - $width)}]
        lset positions $entry_index $placement_x
        set next_x $placement_x
      }
      if {[lindex $positions 0] < $start} {
        error "CTS backward compaction crossed row $row_index segment $segment_index"
      }
    }
    set previous_end $start
    for {set entry_index 0} {$entry_index < [llength $assigned_entries]} {incr entry_index} {
      set entry [lindex $assigned_entries $entry_index]
      set inst [lindex $entry 3]
      set placement_x [lindex $positions $entry_index]
      set master [$inst getMaster]
      set actual_width [$master getWidth]
      set actual_height [$master getHeight]
      if {(($placement_x - $row_x) % $spacing) != 0} {
        error "off-site CTS buffer [$inst getName]"
      }
      incr cts_site_aligned_checks
      if {$placement_x < $start || $placement_x + $actual_width > $end ||
          $actual_height > [[$row getSite] getHeight]} {
        error "CTS buffer [$inst getName] escapes a fixed-clear row segment"
      }
      incr cts_segment_containment_checks
      incr cts_fixed_clear_checks
      if {$placement_x < $previous_end} {
        error "CTS buffer overlap for [$inst getName]"
      }
      set previous_end [expr {$placement_x + $actual_width}]
      incr cts_standard_nonoverlap_checks
      set original_x [lindex $entry 0]
      set original_y [lindex $entry 1]
      set displacement [expr {abs($placement_x - $original_x) + abs($row_y - $original_y)}]
      if {$displacement > $cts_maximum_displacement_dbu} {
        set cts_maximum_displacement_dbu $displacement
      }
      set cts_total_displacement_dbu [expr {$cts_total_displacement_dbu + $displacement}]
      $inst setOrient $row_orient
      $inst setLocation $placement_x $row_y
      $inst setPlacementStatus FIRM
      incr cts_placed_count
    }
  }
}
if {$cts_placed_count != $cts_buffer_count} {
  error "legalized $cts_placed_count of $cts_buffer_count CTS buffers"
}
puts [format "MLX_CTS_BUFFER_LEGALIZATION buffers=%d backward_compactions=%d site_aligned=%d segment_contained=%d fixed_clear=%d standard_nonoverlap=%d max_displacement_dbu=%d average_displacement_dbu=%.6f" \
  $cts_placed_count $cts_backward_compactions $cts_site_aligned_checks \
  $cts_segment_containment_checks $cts_fixed_clear_checks \
  $cts_standard_nonoverlap_checks $cts_maximum_displacement_dbu \
  [expr {$cts_placed_count > 0 ? double($cts_total_displacement_dbu) / $cts_placed_count : 0.0}]]
