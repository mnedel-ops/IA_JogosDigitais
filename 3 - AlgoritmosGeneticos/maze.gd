extends TileMapLayer

const TILE_WALL := 0
const TILE_FLOOR := 1
const TILE_TARGET := 2

const DIRECTION_NORTH := Vector2i(0, -1)
const DIRECTION_SOUTH := Vector2i(0, 1)
const DIRECTION_EAST := Vector2i(1, 0)
const DIRECTION_WEST := Vector2i(-1, 0)

var target_cell: Vector2i
var spawn_cell: Vector2i

func _ready() -> void:
    _find_special_cells()

func _find_special_cells() -> void:
    for cell in get_used_cells():
        var id := get_cell_source_id(cell)
        if id == TILE_TARGET:
            target_cell = cell

func is_target_cell(cell: Vector2i) -> bool:
    return cell == target_cell

func cell_to_world(cell: Vector2i) -> Vector2:
    return map_to_local(cell)

func world_to_cell(world: Vector2) -> Vector2i:
    return local_to_map(world)

func get_exits(cell: Vector2i, came_from_direction: Vector2i) -> Array[Vector2i]:
    var exits: Array[Vector2i] = []
    var all_directions := [DIRECTION_NORTH, DIRECTION_SOUTH, DIRECTION_EAST, DIRECTION_WEST]
    var reverse_directions := -came_from_direction

    for direction in all_directions:
        if direction == reverse_directions:
            continue
        var neighbor := cell + direction
        if not is_wall(neighbor):
                exits.append(direction)
    return exits