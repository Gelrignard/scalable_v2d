import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
from ..paths import enable_upstream

enable_upstream()
from robotic_grounding.retarget.robot_config import load_robot_config
from robotic_grounding.retarget.whole_body_kinematics import WholeBodyKinematics
from .contacts import TIP_NAMES


class G1Kinematics(WholeBodyKinematics):
    """NVIDIA model/config plus explicit Dex3 fingertip and collision probe frames."""

    def __init__(self, **kwargs):
        self.probe_names = []
        self.probe_radii = []
        super().__init__(config=load_robot_config("g1"), **kwargs)
        self.joint_names = [str(n) for n in self.robot.model.names][2:]

    def load_robot_model(self):
        robot = super().load_robot_model()
        model = robot.model

        def add(name, parent_name, xyz):
            parent_id = model.getFrameId(parent_name)
            if parent_id >= model.nframes:
                raise ValueError(f"Missing robot frame {parent_name}")
            parent = model.frames[parent_id]
            placement = parent.placement * pin.SE3(
                np.eye(3), np.asarray(xyz, dtype=float)
            )
            model.addFrame(
                pin.Frame(
                    name,
                    parent.parentJoint,
                    parent_id,
                    placement,
                    pin.FrameType.OP_FRAME,
                )
            )

        for side in ("left", "right"):
            # Reuse the manufacturer's/NVIDIA's fingertip site definitions.
            root = ET.parse(
                self._config.urdf_path.parent / f"dex3_{side}.urdf"
            ).getroot()
            for joint in root.findall("joint"):
                child = joint.find("child").attrib["link"]
                if child in TIP_NAMES:
                    origin = joint.find("origin")
                    if origin.attrib.get("rpy", "0 0 0") != "0 0 0":
                        raise ValueError(
                            "Non-identity tip rotation needs explicit handling"
                        )
                    add(
                        child,
                        joint.find("parent").attrib["link"],
                        list(map(float, origin.attrib["xyz"].split())),
                    )
            sign = -1 if side == "left" else 1
            probes = [
                ("palm", [0.035, 0, 0], 0.022),
                ("index_0", [0.025, 0, 0], 0.008),
                ("index_1", [0.0175, 0, 0], 0.007),
                ("middle_0", [0.025, 0, 0], 0.008),
                ("middle_1", [0.0175, 0, 0], 0.007),
                ("thumb_1", [0, sign * 0.015, 0], 0.008),
                ("thumb_2", [0, sign * 0.0175, 0], 0.007),
            ]
            for part, xyz, radius in probes:
                name = f"{side}_{part}_probe"
                add(name, f"{side}_hand_{part}_link", xyz)
                self.probe_names.append(name)
                self.probe_radii.append(radius)
        self.probe_names += list(TIP_NAMES)
        self.probe_radii += [0.002] * len(TIP_NAMES)
        robot.data = model.createData()
        return robot

    def poses(self, q, names):
        pin.forwardKinematics(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        return [
            self.robot.data.oMf[self.robot.model.getFrameId(n)].copy() for n in names
        ]
