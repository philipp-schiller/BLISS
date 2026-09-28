
class GeometricEllipsoidFitter(BaseEllipsoidFitter):

    def fit(points: np.ndarray) -> dict:
        """
        Geometric LS ellipsoid fit via LM — minimises algebraic distance
        with geometric initialisation from Li & Griffiths.
        """

        p = init["params"]
        cx, cy, cz = p["center"]
        a, b, c = p["semi_axes"]
        # Encode rotation as axis-angle (Rodrigues) — 3 params
        from scipy.spatial.transform import Rotation as Rot
        R = project_to_so3(p["rotation_matrix"])
        rvec = Rot.from_matrix(R).as_rotvec()

