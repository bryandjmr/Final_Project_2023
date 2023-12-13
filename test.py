"""Final Project"""

import math
import numpy as np
import matplotlib.pyplot as plt 
import os


class particle:
    """
    Class object that describes how a particle
    moves and how it interacts with other particles.
    """

    def __init__(self, r, v, rho = 0.01):
        """
        intrinsic properties of particles

        Args:
            r (list): position
            v (list): velocity
            rho (float, optional): radius. Defaults to 0.1.
        """        
        self.r = np.array(r)
        self.v = np.array(v)
        self.rho = np.array(rho)

    def __eq__(self, other):
        """defines equality of particles

        Args:
            other (particle): other particle

        Returns:
            boolean: if particles equal each other
        """
        if type(other) == particle:
            return list(self.r) == list(other.r)
        return False

    
    def dist(self, other):
        """generates distance between two particles

        Args:
            other (particle): othe particle

        Returns:
            int or float: distance between particles
        
        """

        return math.sqrt(sum((self.r - other.r)**2))

    def overlap(self, other):
        """determines if two particles are overlapping

        Args:
            other (particle): other particle

        Returns:
            boolean: if particle overlap
        """
        return self.dist(other) < (self.rho + other.rho)

    def overlap_corr(self, other):
        """corrects the velocity of overlapping particles

        Args:
            other (particle): other particle
        """
        v1, v2 = self.v.copy(), other.v.copy()
        r1, r2 = self.r - other.r, other.r - self.r
        self.v = v1 - sum((v1 - v2) * r1) / sum(r1**2) * r1
        other.v = v2 - sum((v2 - v1) * r2) / sum(r2**2) * r2

    def step(self, t):
        self.r += self.v * t

class box:
    """
    Class object that describes the box that the particles live in
    and has a function to do a simulation and create a boltzman 
    distribution.
    """

    def __init__(self, x = [-1, 1], y = [-1, 1]):
        """intrinsic properties of particles


        Args:
            x (list, optional): x bounds. Defaults to [-1, 1].
            y (list, optional): y bounds. Defaults to [-1, 1].
        """


        self.xb = x
        self.yb = y
        self.particles = []
        self.avg_vel = 0

    def average_velocity(self):
        """generates average velocity of particles
        """

        avg = 0
        for p in self.particles:
            avg += (math.sqrt(sum(p.v**2)))
        return avg/len(self.particles)

    def add_particle(self, p):
        """adds particle to box

        Args:
            p (particle): particle or list of particles
            being added
        """
        if type(p) == list:
            self.particles += p
        else:
            self.particles.append(p)

    def push(self, t):
        """push particles forward by time step
        t

        Args:
            t (int or float): time input
        """
        for p in self.particles:
            p.step(t)

    def check_boundary(self):
        """check if particles crossed boundary
        """
        for p in self.particles:
            x, y = p.r
            if x > self.xb[1] or x < self.xb[0]:
                p.v[0] = -p.v[0]
            if y > self.yb[1] or y < self.yb[0]:
                p.v[1] = -p.v[1]

    def check_overlap(self):
        """checks if particles are overlapping in box
        """
        for p1 in self.particles:
            for p2 in self.particles:
                if p1 != p2:
                    if p1.overlap(p2):
                        p1.overlap_corr(p2)


    def record(self):
        """records the current position of the particles

        Returns:
            _type_: _description_
        """
        ans = [0] * len(self.particles)
        for i, p in enumerate(self.particles):
            ans[i] = p.r
        return np.array(ans)

    def simulation(self, time, dt=0.05):
        """creates 3d array that houses all the x and y coordinates
        of the particles during the simulation

        Args:
            time (int or float): time input
            dt (float, optional): time step. Defaults to 0.05.

        Returns:
            array: 3d array of all the particles positions during
            simulation
        """
        self.avg_vel = self.average_velocity()
        sim = [0] * (int(time / dt) + 1)
        time_step = np.arange(0, time, dt)
        sim[0] = self.record()
        for i in range(len(time_step)):
            self.push(dt)
            self.check_boundary()
            self.check_overlap()
            sim[i+1] = self.record()
        return sim

    def animate(self, records):
        """creates video of simulation

        Args:
            records (array): 3d array of all 
            past positions of particles
        """
        for i, iter in enumerate(records):
            figure, axes = plt.subplots()
            for xi, yi in iter:
                cir = plt.Circle((xi,yi), self.particles[0].rho)
                axes.add_artist(cir)
            plt.xlim(self.xb[0]-0.2, self.xb[1]+0.2)
            plt.ylim(self.yb[0]-0.2, self.yb[1]+0.2)
            plt.savefig('fig_{0:03}.png'.format(i))
            plt.close()

        os.system('convert -delay 25 ./fig_*.png -loop 1 movie_test.gif')
        [os.remove(file) for file in os.listdir('./') if file.endswith('.png')]

    def velocity(self):
        "generates list of magnitude of velocity of particles"
        vel = [0] * len(self.particles)
        for i, p in enumerate(self.particles):
            vel[i] = math.sqrt(sum(p.v**2))
        return vel


    def boltzmann(self, data):
        """creates graph to show the fit of velocity to 
        boltzman distribution

        Args:
            data (list): list of all past positions of 
            the particles
        """
        v = np.linspace(0, 2000, 1000)
        coeff = 2/self.avg_vel**2
        f = coeff*v*np.exp(-coeff*v**2/2)
        v_part = self.velocity()
        plt.hist(v_part, density=True)
        plt.plot(v, f)
        plt.xlabel('Velocity')
        plt.ylabel('Particle Count')
        plt.title('Boltzmann Distribution')
        plt.savefig('Boltzmann_Distribution')
        plt.close()



if __name__ == "__main__":
    p_list = []
    for i in range(20):
        p_list.append(particle(np.random.uniform(0, 1, 2), np.random.uniform(0,1,2)))
    d = box()
    d.add_particle(p_list)
    data = d.simulation(8)
    d.animate(data)
    d.boltzmann(data)