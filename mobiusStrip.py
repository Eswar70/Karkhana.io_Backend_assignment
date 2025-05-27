import numpy as np
import matplotlib
matplotlib.use("Qt5Agg") 
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.spatial.distance import euclidean


class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=200):
        self.R = R            # Radius from center
        self.w = w            # Width of the strip
        self.n = n            # Resolution (number of points)
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._compute_coordinates()

    def _compute_coordinates(self):
        U, V = self.U, self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def compute_surface_area(self):
        """Approximate surface area using numerical integration."""
        dU = self.u[1] - self.u[0]
        dV = self.v[1] - self.v[0]
        
        # Calculate partial derivatives
        Xu = np.gradient(self.X, dU, axis=1)
        Yu = np.gradient(self.Y, dU, axis=1)
        Zu = np.gradient(self.Z, dU, axis=1)

        Xv = np.gradient(self.X, dV, axis=0)
        Yv = np.gradient(self.Y, dV, axis=0)
        Zv = np.gradient(self.Z, dV, axis=0)

        # Cross product magnitude |Xu x Xv| for each (u,v)
        cross_x = Yu * Zv - Zu * Yv
        cross_y = Zu * Xv - Xu * Zv
        cross_z = Xu * Yv - Yu * Xv
        dA = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        # Integrate over the surface
        surface_area = simpson(simpson(dA, self.v), self.u)
        return surface_area

    def compute_edge_length(self):
        """Approximate the length of one boundary edge."""
        # Use edge v = +w/2
        edge_x = (self.R + self.w / 2 * np.cos(self.u / 2)) * np.cos(self.u)
        edge_y = (self.R + self.w / 2 * np.cos(self.u / 2)) * np.sin(self.u)
        edge_z = (self.w / 2) * np.sin(self.u / 2)

        length = 0.0
        for i in range(1, len(self.u)):
            p1 = np.array([edge_x[i - 1], edge_y[i - 1], edge_z[i - 1]])
            p2 = np.array([edge_x[i], edge_y[i], edge_z[i]])
            length += euclidean(p1, p2)
        return length

    def plot(self):
        """3D Plot of the Möbius strip using matplotlib."""
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='k', alpha=0.9)
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.4, n=300)

    area = mobius.compute_surface_area()
    edge_length = mobius.compute_edge_length()

    print(f"Surface Area ≈ {area:.5f}")
    print(f"Edge Length ≈ {edge_length:.5f}")

    mobius.plot()
