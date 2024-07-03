import mujoco
import mujoco.viewer
import time 

# Make model and data
m = mujoco.MjModel.from_xml_path("models/spider/scene.xml")
d = mujoco.MjData(m)

exited = False
paused = False

def key_callback(keycode):
    if chr(keycode) == 'x':
        global exited
        exited = not exited
    if chr(keycode) == ' ':
        global paused
        paused = not paused

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    while viewer.is_running() and not exited:
        if not paused:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    viewer.close()