import matplotlib.pyplot as plt


def plot_person_cluster(person_name, face_points_2d, person_point_2d):
    """Plot one person's 2D face points and their person embedding (centroid)."""
    plt.figure(figsize=(7, 6))
    plt.scatter(face_points_2d[:, 0], face_points_2d[:, 1], alpha=0.7, label="faces")
    plt.scatter(person_point_2d[0], person_point_2d[1], marker="X", s=250,
                edgecolors="black", label="centroid")
    plt.title(f"Face embeddings for {person_name}")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def draw_person_clusters(groups_2d):
    """Draw every person's face points and centroid in that person's color on the current figure."""
    for color_index, (person_name, (face_points_2d, person_point_2d)) in enumerate(groups_2d.items()):
        color = f"C{color_index}"
        plt.scatter(face_points_2d[:, 0], face_points_2d[:, 1], color=color, alpha=0.6, label=person_name)
        plt.scatter(person_point_2d[0], person_point_2d[1], color=color, marker="X", s=250,
                    edgecolors="black", label=f"{person_name} centroid")


def plot_person_clusters(groups_2d):
    """Plot several people's clusters on one figure to compare them.
    groups_2d: {person_name: (face_points_2d (N, 2), person_point_2d (2,))}."""
    plt.figure(figsize=(8, 7))
    draw_person_clusters(groups_2d)
    plt.title("Face embedding clusters per person")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_face_against_clusters(face_label, face_point_2d, groups_2d):
    """Plot one face's 2D point against every person's cluster and centroid.
    groups_2d: {person_name: (face_points_2d (N, 2), person_point_2d (2,))}."""
    plt.figure(figsize=(8, 7))
    draw_person_clusters(groups_2d)
    plt.scatter(face_point_2d[0], face_point_2d[1], color="red", marker="*", s=400,
                edgecolors="black", label=face_label, zorder=3)
    plt.title(f"{face_label} against person clusters")
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
