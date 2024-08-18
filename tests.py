import orekit
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.data import DataProvidersManager, ZipJarCrawler
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, OrbitType, PositionAngle
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.utils import Constants, PVCoordinates
from org.orekit.propagation.analytical.keplerian import KeplerianPropagator
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation.numerical import ForceModel
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity.potential import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.gravity import GravityFieldFactory
from org.orekit.frames import Frame
from org.orekit.orbits import Orbit
from org.orekit.propagation.sampling import OrekitFixedStepHandler
from math import radians, degrees

# Setup Orekit
orekit.initVM()

# Load Orekit data
data_file = 'orekit-data.zip'  # Adjust path if necessary
manager = DataProvidersManager.getInstance()
crawler = ZipJarCrawler(data_file)
manager.clearProviders()
manager.addProvider(crawler)
setup_orekit_curdir()

# Constants
mu = Constants.EIGEN5C_EARTH_MU  # Earth's gravitational parameter (m^3/s^2)

# Define initial orbit parameters (Example: Low Earth Orbit)
a = 7000000.0             # Semi-major axis (m)
e = 0.001                 # Eccentricity
i = 98.6                  # Inclination (degrees)
omega = 0.0               # Argument of perigee (degrees)
raan = 0.0                # Right Ascension of Ascending Node (degrees)
lM = 0.0                  # Mean anomaly (degrees)

# Reference frame and epoch
inertial_frame = FramesFactory.getEME2000()
epoch = AbsoluteDate.J2000_EPOCH
time_scale = TimeScalesFactory.getUTC()

# Create initial orbit
initial_orbit = KeplerianOrbit(a, e, radians(i), radians(omega), radians(raan), radians(lM), PositionAngle.MEAN, inertial_frame, epoch, mu)

# Integrator parameters
min_step = 0.001
max_step = 300.0
position_tolerance = 10.0

# Orbit type for the propagator
propagator_type = OrbitType.CARTESIAN

# Numerical propagator setup
integrator = DormandPrince853Integrator(min_step, max_step, position_tolerance)
propagator = NumericalPropagator(integrator)
propagator.setOrbitType(propagator_type)

# Gravity field model
gravity_provider = GravityFieldFactory.getNormalizedProvider(8, 8)
propagator.addForceModel(HolmesFeatherstoneAttractionModel(inertial_frame, gravity_provider))

# Set initial state
propagator.setInitialState(propagator.getInitialState().withOrbit(initial_orbit))

# Define the radial thrust maneuver
thrust = 0.5  # Thrust (N)
isp = 300.0   # Specific Impulse (s)
direction = [1.0, 0.0, 0.0]  # Radial direction in LVLH frame (X-axis)

# Create thrust maneuver
maneuver = ConstantThrustManeuver(inertial_frame, epoch, thrust, isp, direction)

# Add maneuver to the propagator
propagator.addForceModel(maneuver)

# Propagation duration (in seconds)
duration = 600.0

# Set up step handler to get results
class MyStepHandler(OrekitFixedStepHandler):
    def __init__(self):
        self.orbits = []

    def handleStep(self, current_state, is_last):
        self.orbits.append(current_state.getOrbit())

step_handler = MyStepHandler()
propagator.setStepHandler(step_handler)

# Propagate
final_state = propagator.propagate(epoch.shiftedBy(duration))

# Extract final orbit
final_orbit = final_state.getOrbit()

# Print results
print("Initial Orbit:")
print(f"Semi-major axis: {initial_orbit.getA():.3f} m")
print(f"Eccentricity: {initial_orbit.getE():.6f}")
print(f"Inclination: {degrees(initial_orbit.getI()):.3f} degrees")

print("\nFinal Orbit after Radial Thrust Maneuver:")
print(f"Semi-major axis: {final_orbit.getA():.3f} m")
print(f"Eccentricity: {final_orbit.getE():.6f}")
print(f"Inclination: {degrees(final_orbit.getI()):.3f} degrees")
