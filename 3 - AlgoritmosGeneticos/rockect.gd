class_name Rocket
extends Node2D

# DNA
enum Decision { LEFT, FORWARD, RIGHT}

const MAX_DECISIONS := 100
var dna: Array[int] = []

# Estados
var current_cell: Vector2i
var gene_index: int = 0
var facing_direction: Vector2i
var steps_taken: int = 0
var max_steps: int = 200

var is_alive: bool = true  
var reached_target: bool = false
var fitness: float = 0.0

# Referencia do alvo (injetada pelo Manager)
var maze: TileMapLayer

const STEP_INTERVAL := 0.05
var _step_timer: float = 0.0

func _ready() -> void:
    if dna.is_empty():
        _randomize_dna()

    facing_direction = Vector2i(0, -1)
    position = maze.map_to_local(current_cell)

func _randomize_dna() -> void:
    dna.clear()
    for i in range(MAX_DECISIONS):
        dna.append(randi() % 3)  # 0, 1 ou 2

func _process(delta: float) -> void:
    if not is_alive or reached_target:
        return

    if steps_taken >= max_steps or gene_index > MAX_DECISIONS:
        is_alive = false
        return

    _step_timer += delta
    if _step_timer >= STEP_INTERVAL:
        _step_timer = 0.0
        _move_one_step()

func _move_one_step() -> void:
    var exits := maze.get_exits(current_cell, facing_direction)
    var chosen_direction: Vector2i

    if exits.size() == 0:
        is_alive = false
        return
    elif exits.size() == 1:
        chosen_direction = exits[0]
    else:
        if gene_index >= dna.size():
            is_alive = false
            return
        chosen_direction = _decide(exits, dna[gene_index])
        gene_index += 1

    facing_direction = chosen_direction
    current_cell += facing_direction
    steps_taken += 1

    position = maze.map_to_local(current_cell)

    if maze.is_target(current_cell):
        reached_target = true
        is_alive = false

func _decide(exits: Array[Vector2i], gene: int) -> Vector2i:
    var relative := _classify_exits(exits)

    match gene:
        Decision.FORWARD:
            if relative.has("forward"):
                return exits["forward"]
        Decision.LEFT:
            if relative.has("left"):
                    return exits["left"]
        Decision.RIGHT:
            if relative.has("right"):
                return exits["right"]

    return exits[randi() % exits.size()]

func _classify_exits(exits: Array[Vector2i]) -> Dictionary:
    var result := {}

    var right_direction := Vector2i(-facing_direction.y, facing_direction.x)
    var left_direction := Vector2i(facing_direction.y, -facing_direction.x)

    for exit in exits:
        if exit == facing_direction:
            result["forward"] = exit
        elif exit == right_direction:
            result["right"] = exit
        elif exit == left_direction:
            result["left"] = exit
    return result

func calculate_fitness(target_cell: Vector2i) -> void:
    var distance := float((current_cell - target_cell).length())

    fitness := 1.0 / (distance + 1.0)

    if reached_target:
        fitness *= 20.0

    if reached_target and steps_taken > 0:
        fitness *= (1.0 / float(steps_taken)) * 100.0

    if not is_alive and not reached_target:
        fitness *= 0.05