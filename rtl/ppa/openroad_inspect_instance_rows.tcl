read_db $::env(PPA_INSPECT_ODB)
set block [ord::get_db_block]
puts "MLX_INSPECT_DB checkpoint=$::env(PPA_INSPECT_ODB)"

foreach instance_name $::env(PPA_INSPECT_INSTANCES) {
  set inst [$block findInst $instance_name]
  if {$inst == "NULL"} {
    puts "MLX_INSPECT_INSTANCE_MISSING name=$instance_name"
    continue
  }
  set location [$inst getLocation]
  set bbox [$inst getBBox]
  set master [$inst getMaster]
  puts "MLX_INSPECT_INSTANCE name=$instance_name status=[$inst getPlacementStatus] orient=[$inst getOrient] location=[lindex $location 0],[lindex $location 1] bbox=[$bbox xMin],[$bbox yMin],[$bbox xMax],[$bbox yMax] master=[$master getName] size=[$master getWidth],[$master getHeight]"
  set row_matches 0
  foreach row [$block getRows] {
    set origin [$row getOrigin]
    set row_x [lindex $origin 0]
    set row_y [lindex $origin 1]
    set row_end [expr {$row_x + [$row getSiteCount] * [$row getSpacing]}]
    if {abs($row_y - [$bbox yMin]) <= 5600 &&
        $row_x <= [$bbox xMin] && $row_end >= [$bbox xMax]} {
      puts "MLX_INSPECT_ROW name=[$row getName] origin=$row_x,$row_y end=$row_end orient=[$row getOrient] spacing=[$row getSpacing] site_height=[[$row getSite] getHeight]"
      incr row_matches
      if {$row_matches >= 8} { break }
    }
  }
  puts "MLX_INSPECT_ROW_MATCHES name=$instance_name count=$row_matches"
  if {$row_matches == 0} {
    set nearby_rows 0
    foreach row [$block getRows] {
      set origin [$row getOrigin]
      set row_x [lindex $origin 0]
      set row_y [lindex $origin 1]
      set row_end [expr {$row_x + [$row getSiteCount] * [$row getSpacing]}]
      if {abs($row_y - [$bbox yMin]) <= 5600} {
        puts "MLX_INSPECT_NEAR_ROW name=[$row getName] origin=$row_x,$row_y end=$row_end orient=[$row getOrient] spacing=[$row getSpacing]"
        incr nearby_rows
        if {$nearby_rows >= 20} { break }
      }
    }
    puts "MLX_INSPECT_NEAR_ROW_COUNT name=$instance_name count=$nearby_rows"
  }
}
