# Model source

`bari_sim.model_builder.ModelBuilder` is the authoritative MuJoCo model definition.
It generates a complete MJCF document from `config/*.toml` for the selected robot
count and environment. Keeping guessed dimensions, masses, friction, and attachment
strength in one configuration source prevents static XML copies from drifting.

Use `python scripts/export_model.py --environment gap --robots 4` to inspect or edit
the generated `models/generated_gap.xml`. Generated XML is an inspection artifact;
changes there are not loaded back into the simulator.

Collision geoms use group 3. Visual-only geoms use group 1. Markers use group 2.
Future CAD meshes can replace group-1 geoms without changing contact or inertia.

