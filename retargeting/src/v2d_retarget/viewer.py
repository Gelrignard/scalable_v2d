"""Inspect the same source, targets, final FK and inferred contacts per frame."""

import threading
import time
import re
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation
from .core.robot import G1Kinematics
from .core.contacts import TIP_NAMES
from .output import load_result
from .paths import WORKSPACE
from robotic_grounding.retarget.pinocchio_viser_visualizer import ViserVisualizer


def serve(paths, port=8081):
    records = {}
    for path in paths:
        data, metadata = load_result(path)
        records[metadata["method"]] = (data, metadata)
    if not records:
        raise ValueError("Provide at least one result file")
    first = next(iter(records.values()))[0]
    for data, _ in records.values():
        if data["timestamps"].shape != first["timestamps"].shape or not np.allclose(
            data["timestamps"], first["timestamps"]
        ):
            raise ValueError(
                "Comparisons must use identical frame selections and timing"
            )
    kin = G1Kinematics()
    for _, meta in records.values():
        if meta["joint_names"] != kin.joint_names:
            raise ValueError("Robot joint order differs from the saved result")
    server = viser.ViserServer(host="127.0.0.1", port=port)
    visual = ViserVisualizer(
        server,
        model=kin.robot.model,
        visual_model=kin.robot.visual_model,
        collision_model=kin.robot.collision_model,
    )
    visual.displayFrames(False)
    visual.displayCollisions(False)
    mesh = trimesh.load(str(first["object_mesh"]), force="mesh", process=False)
    obj = server.scene.add_mesh_trimesh("/object", mesh)
    server.scene.add_grid("/floor", width=5.0, height=5.0, plane="xy")
    support = WORKSPACE / "outputs/snack_box_support.usda"
    if (
        support.exists()
        and "snack_box" in next(iter(records.values()))[1]["sequence_id"]
    ):
        text = support.read_text()
        radius = float(re.search(r"double radius = ([\d.eE+-]+)", text).group(1))
        height = float(re.search(r"double height = ([\d.eE+-]+)", text).group(1))
        position = tuple(
            map(
                float,
                re.search(r"xformOp:translate = \(([^)]+)\)", text).group(1).split(","),
            )
        )
        server.scene.add_mesh_trimesh(
            "/support",
            trimesh.creation.cylinder(radius=radius, height=height),
            position=position,
        )
    human = server.scene.add_point_cloud(
        "/human",
        points=first["human_positions"][0].astype(np.float32),
        colors=(20, 190, 220),
        point_size=0.012,
    )
    tips = server.scene.add_point_cloud(
        "/targets/tips",
        points=first["tip_targets"][0].astype(np.float32),
        colors=(250, 170, 35),
        point_size=0.016,
    )
    body = server.scene.add_point_cloud(
        "/targets/body",
        points=first["body_targets"][0].astype(np.float32),
        colors=(230, 160, 60),
        point_size=0.012,
    )
    fk = server.scene.add_point_cloud(
        "/fk/tips",
        points=first["fk_tip_positions"][0].astype(np.float32),
        colors=(210, 65, 215),
        point_size=0.012,
    )
    anchors = server.scene.add_point_cloud(
        "/contact/anchors",
        points=np.zeros((1, 3), np.float32),
        colors=(35, 210, 80),
        point_size=0.02,
    )
    lines = server.scene.add_line_segments(
        "/errors/tips",
        points=np.zeros((6, 2, 3), np.float32),
        colors=(235, 65, 55),
        line_width=2.0,
    )
    frame_names = [
        f"{s}_{suffix}"
        for s in ("left", "right")
        for suffix in ("wrist_yaw_link", "hand_palm_link")
    ] + list(TIP_NAMES)
    axes = {
        n: server.scene.add_frame("/axes/" + n, axes_length=0.045, axes_radius=0.0015)
        for n in frame_names
    }
    with server.gui.add_folder("Retargeting comparison"):
        server.gui.add_markdown(
            "**Contact-aware retargeting**\n\nCyan: human · Orange: targets · Magenta: robot tips · Green: inferred object contacts.\n\nBaseline does not optimize the displayed fingertip targets. Object playback is prescribed. Contact and balance are not physics-validated."
        )
        selector = server.gui.add_dropdown(
            "Motion",
            options=list(records),
            initial_value="contact_aware"
            if "contact_aware" in records
            else next(iter(records)),
        )
        frame = server.gui.add_slider(
            "Frame", min=0, max=max(1, len(first["q"]) - 1), step=1, initial_value=0
        )
        play = server.gui.add_checkbox("Play", initial_value=False)
        rate = server.gui.add_number(
            "Playback FPS", initial_value=30.0, min=1.0, max=240.0, step=1.0
        )
        show_human = server.gui.add_checkbox("Human joints", initial_value=True)
        show_targets = server.gui.add_checkbox("IK targets", initial_value=True)
        show_axes = server.gui.add_checkbox(
            "Wrist / palm / tip axes", initial_value=False
        )
        closeup = server.gui.add_button("Focus on hands")
        overview = server.gui.add_button("Show whole body")
        status = server.gui.add_markdown("")
    lock = threading.Lock()

    def update(_=None):
        with lock:
            data, meta = records[selector.value]
            t = min(int(frame.value), len(data["q"]) - 1)
            with server.atomic():
                visual.display(data["q"][t])
                obj.position = data["object_positions"][t]
                obj.wxyz = data["object_wxyz"][t]
                human.points = data["human_positions"][t].astype(np.float32)
                human.visible = show_human.value
                tips.points = data["tip_targets"][t].astype(np.float32)
                tips.visible = show_targets.value
                body.points = data["body_targets"][t].astype(np.float32)
                body.visible = show_targets.value
                fk.points = data["fk_tip_positions"][t].astype(np.float32)
                lines.points = np.stack(
                    [data["fk_tip_positions"][t], data["tip_targets"][t]], axis=1
                ).astype(np.float32)
                lines.visible = show_targets.value
                active = data["contact_active"][t]
                rotation = Rotation.from_quat(
                    data["object_wxyz"][t], scalar_first=True
                ).as_matrix()
                anchor = (
                    data["contact_anchors_local"][t] @ rotation.T
                    + data["object_positions"][t]
                )
                anchors.points = anchor[active].astype(np.float32)
                anchors.visible = bool(active.any())
                for name, pose in zip(
                    frame_names, kin.poses(data["q"][t], frame_names)
                ):
                    axes[name].position = pose.translation
                    axes[name].wxyz = Rotation.from_matrix(pose.rotation).as_quat(
                        scalar_first=True
                    )
                    axes[name].visible = show_axes.value
                contact = data["contact_error_m"][t][active]
                mean = (
                    "none active"
                    if not len(contact)
                    else f"{contact.mean() * 1000:.1f} mm"
                )
                status.content = f"Time: **{data['timestamps'][t]:.3f} s** · Active tips: **{active.sum()} / 6**\n\nContact-anchor error: **{mean}**\n\nMax hand-probe hull penetration: **{data['probe_penetration_m'][t].max() * 1000:.1f} mm**"

    for control in (selector, frame, show_human, show_targets, show_axes):
        control.on_update(update)

    @server.on_client_connect
    def connect(client):
        client.camera.position = (3.2, -3.4, 2.0)
        client.camera.look_at = (0.8, 0.0, 0.8)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    @closeup.on_click
    def focus(_):
        data, _ = records[selector.value]
        t = min(int(frame.value), len(data["q"]) - 1)
        center = data["fk_tip_positions"][t].mean(axis=0)
        for client in server.get_clients().values():
            client.camera.look_at = center
            client.camera.position = center + np.array([0.65, -0.75, 0.35])

    @overview.on_click
    def overview_camera(_):
        for client in server.get_clients().values():
            connect(client)

    update()
    print(f"Viewer: http://127.0.0.1:{port}", flush=True)
    try:
        while True:
            before = time.monotonic()
            if play.value:
                frame.value = (int(frame.value) + 1) % len(first["q"])
            time.sleep(
                max(0.001, 1 / max(1.0, rate.value) - (time.monotonic() - before))
            )
    except KeyboardInterrupt:
        server.stop()
